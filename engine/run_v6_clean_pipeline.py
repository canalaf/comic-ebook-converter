# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
PIPELINE_SOURCE = ROOT / "v55" / "marvel_kobo_v58_smart_merge.py"
PUBLIC_MODEL = ROOT / "models" / "comic_base.pt"
V6_MODEL = PUBLIC_MODEL
V3_RESCUE_MODEL = PUBLIC_MODEL
BASE_MODEL = PUBLIC_MODEL


def require_release_files(*paths: Path) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Required release files are missing. Download the model pack from "
            f"the GitHub Releases page and restore these paths:\n{formatted}"
        )


def load_pipeline(primary_model: Path = V6_MODEL, rescue_model: Path = V3_RESCUE_MODEL):
    require_release_files(PIPELINE_SOURCE, primary_model, rescue_model, BASE_MODEL)
    spec = importlib.util.spec_from_file_location("comic_clean_v6_pipeline", PIPELINE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.CLEAN_MODEL_PATH = primary_model
    # The old model is never trusted as the primary detector. It is allowed to
    # contribute at most two candidates and only inside regions V6 left empty.
    module.LEGACY_MODEL_PATH = rescue_model
    module.BASE_MODEL_PATH = BASE_MODEL
    # A few Torn panels were 50-80 pixels away from the true gutter. Widen
    # the existing visual search window without changing the acceptance score.
    module.BORDER_SNAP_PAGE_RANGE = 0.04
    module.BORDER_SNAP_BOX_RANGE = 0.12

    original_detect_hybrid = module.detect_hybrid
    original_get_frame_detections = module.get_frame_detections
    original_expand_box = module.expand_box
    original_expand_box_for_text = module.expand_box_for_text

    def clamp_safe_box_to_near_neighbors(safe_box, panel_box, neighbors, image_w, image_h):
        """Keep padding/text expansion from leaking a thin strip of a neighbor."""
        sx1, sy1, sx2, sy2 = safe_box
        px1, py1, px2, py2 = panel_box
        panel_w = max(1, px2 - px1)
        panel_h = max(1, py2 - py1)
        panel_cx = (px1 + px2) / 2
        panel_cy = (py1 + py2) / 2
        bleed_guard = max(3, round(min(image_w, image_h) * 0.006))

        for other in neighbors or []:
            if tuple(other) == tuple(panel_box):
                continue
            ox1, oy1, ox2, oy2 = other
            other_w = max(1, ox2 - ox1)
            other_h = max(1, oy2 - oy1)
            other_cx = (ox1 + ox2) / 2
            other_cy = (oy1 + oy2) / 2
            horizontal_overlap = max(0, min(px2, ox2) - max(px1, ox1)) / max(
                1, min(panel_w, other_w)
            )
            vertical_overlap = max(0, min(py2, oy2) - max(py1, oy1)) / max(
                1, min(panel_h, other_h)
            )

            if horizontal_overlap >= 0.55:
                near_y = max(10, round(min(panel_h, other_h) * 0.08))
                if other_cy < panel_cy and abs(py1 - oy2) <= near_y:
                    sy1 = max(sy1, round((py1 + oy2) / 2) + bleed_guard)
                elif other_cy > panel_cy and abs(py2 - oy1) <= near_y:
                    sy2 = min(sy2, round((py2 + oy1) / 2) - bleed_guard)

            if vertical_overlap >= 0.55:
                near_x = max(10, round(min(panel_w, other_w) * 0.08))
                if other_cx < panel_cx and abs(px1 - ox2) <= near_x:
                    sx1 = max(sx1, round((px1 + ox2) / 2) + bleed_guard)
                elif other_cx > panel_cx and abs(px2 - ox1) <= near_x:
                    sx2 = min(sx2, round((px2 + ox1) / 2) - bleed_guard)

        if sx2 <= sx1 or sy2 <= sy1:
            return safe_box
        return (
            module.clamp(sx1, 0, image_w),
            module.clamp(sy1, 0, image_h),
            module.clamp(sx2, 0, image_w),
            module.clamp(sy2, 0, image_h),
        )

    def expand_box_v6(box, image_w, image_h, neighbors=None):
        expanded = original_expand_box(box, image_w, image_h, neighbors=neighbors)
        return clamp_safe_box_to_near_neighbors(
            expanded, box, neighbors, image_w, image_h
        )

    def expand_box_for_text_v6(
        safe_box, panel_box, analysis, image_w, image_h, neighbors
    ):
        expanded = original_expand_box_for_text(
            safe_box, panel_box, analysis, image_w, image_h, neighbors
        )
        expanded = clamp_safe_box_to_near_neighbors(
            expanded, panel_box, neighbors, image_w, image_h
        )
        sx1, sy1, sx2, sy2 = expanded
        px1, py1, px2, py2 = panel_box
        margin = max(3, round(min(image_w, image_h) * 0.008))

        # A wide edge panel can contain a speech balloon that reaches the
        # physical page edge. The former 16% expansion cap could still cut it.
        if (px2 - px1) >= 0.70 * image_w:
            for tx1, ty1, tx2, ty2 in analysis["text_regions"]:
                vertical_overlap = max(0, min(py2, ty2) - max(py1, ty1))
                if vertical_overlap <= 0:
                    continue
                if tx1 <= 0.02 * image_w and tx2 >= px1 - 0.03 * image_w:
                    sx1 = 0
                if tx2 >= 0.98 * image_w and tx1 <= px2 + 0.03 * image_w:
                    sx2 = image_w

        # If a balloon genuinely crosses the geometric panel edge, let the
        # owning crop keep it unless another panel already contains it whole.
        for text_box in analysis["text_regions"]:
            tx1, ty1, tx2, ty2 = text_box
            vertical_overlap = max(0, min(py2, ty2) - max(py1, ty1))
            vertical_overlap /= max(1, min(py2 - py1, ty2 - ty1))
            horizontal_overlap = max(0, min(px2, tx2) - max(px1, tx1))
            horizontal_overlap /= max(1, min(px2 - px1, tx2 - tx1))
            covered_elsewhere = any(
                tuple(other) != tuple(panel_box)
                and module.containment_ratio(text_box, other) >= 0.94
                for other in neighbors or []
            )
            if covered_elsewhere:
                continue
            if vertical_overlap >= 0.18 and tx1 < px2 < tx2:
                sx2 = max(sx2, min(tx2 + margin, px2 + round((px2 - px1) * 0.22)))
            if vertical_overlap >= 0.18 and tx1 < px1 < tx2:
                sx1 = min(sx1, max(tx1 - margin, px1 - round((px2 - px1) * 0.22)))
            if horizontal_overlap >= 0.18 and ty1 < py2 < ty2:
                sy2 = max(sy2, min(ty2 + margin, py2 + round((py2 - py1) * 0.18)))
            if horizontal_overlap >= 0.18 and ty1 < py1 < ty2:
                sy1 = min(sy1, max(ty1 - margin, py1 - round((py2 - py1) * 0.18)))

        # OCR often reports a speech balloon as several separate line boxes.
        # The first line can cross the panel edge and trigger the normal 22%
        # expansion, while the following lines begin just outside that limit
        # and remain clipped. Grow through only those connected text fragments,
        # stopping before the midpoint to a real neighboring panel.
        panel_w = max(1, px2 - px1)
        panel_h = max(1, py2 - py1)
        chain_gap = max(4, round(min(image_w, image_h) * 0.012))
        right_limit = min(image_w, px2 + round(panel_w * 0.60))
        left_limit = max(0, px1 - round(panel_w * 0.60))
        bottom_limit = min(image_h, py2 + round(panel_h * 0.42))
        top_limit = max(0, py1 - round(panel_h * 0.42))
        bleed_guard = max(3, round(min(image_w, image_h) * 0.006))

        for other in neighbors or []:
            if tuple(other) == tuple(panel_box):
                continue
            ox1, oy1, ox2, oy2 = other
            vertical_overlap = max(0, min(py2, oy2) - max(py1, oy1))
            vertical_overlap /= max(1, min(panel_h, oy2 - oy1))
            horizontal_overlap = max(0, min(px2, ox2) - max(px1, ox1))
            horizontal_overlap /= max(1, min(panel_w, ox2 - ox1))
            if vertical_overlap >= module.NEIGHBOR_AXIS_OVERLAP:
                if ox1 >= px2:
                    right_limit = min(right_limit, ox1 - bleed_guard)
                elif ox2 <= px1:
                    left_limit = max(left_limit, ox2 + bleed_guard)
            if horizontal_overlap >= module.NEIGHBOR_AXIS_OVERLAP:
                if oy1 >= py2:
                    bottom_limit = min(bottom_limit, oy1 - bleed_guard)
                elif oy2 <= py1:
                    top_limit = max(top_limit, oy2 + bleed_guard)

        def covered_by_another_panel(text_box):
            return any(
                tuple(other) != tuple(panel_box)
                and module.containment_ratio(text_box, other) >= 0.94
                for other in neighbors or []
            )

        # Iterate because adjacent OCR line fragments can become reachable only
        # after the preceding fragment has extended the current crop boundary.
        for _ in range(4):
            previous = (sx1, sy1, sx2, sy2)
            for text_box in analysis["text_regions"]:
                if covered_by_another_panel(text_box):
                    continue
                tx1, ty1, tx2, ty2 = text_box
                vertical_overlap = max(0, min(py2, ty2) - max(py1, ty1))
                vertical_overlap /= max(1, min(panel_h, ty2 - ty1))
                horizontal_overlap = max(0, min(px2, tx2) - max(px1, tx1))
                horizontal_overlap /= max(1, min(panel_w, tx2 - tx1))

                if vertical_overlap >= 0.18:
                    if sx2 < tx2 and tx1 <= sx2 + chain_gap:
                        sx2 = min(right_limit, max(sx2, tx2 + margin))
                    if tx1 < sx1 and tx2 >= sx1 - chain_gap:
                        sx1 = max(left_limit, min(sx1, tx1 - margin))
                if horizontal_overlap >= 0.18:
                    if sy2 < ty2 and ty1 <= sy2 + chain_gap:
                        sy2 = min(bottom_limit, max(sy2, ty2 + margin))
                    if ty1 < sy1 and ty2 >= sy1 - chain_gap:
                        sy1 = max(top_limit, min(sy1, ty1 - margin))
            if (sx1, sy1, sx2, sy2) == previous:
                break

        return (
            module.clamp(sx1, 0, image_w),
            module.clamp(sy1, 0, image_h),
            module.clamp(sx2, 0, image_w),
            module.clamp(sy2, 0, image_h),
        )

    module.expand_box = expand_box_v6
    module.expand_box_for_text = expand_box_for_text_v6

    def represented_fraction(box, existing_boxes):
        area = max(1, module.box_area(box))
        return min(
            1.0,
            sum(module.intersection_area(box, other) for other in existing_boxes) / area,
        )

    def conservative_legacy_rescues(primary, legacy_candidates, image_w, image_h):
        if not legacy_candidates:
            return list(primary), []
        comparison_pool = list(primary) + list(legacy_candidates)
        accepted = list(primary)
        rescued = []
        for candidate in sorted(legacy_candidates, key=lambda item: item["conf"], reverse=True):
            if len(rescued) >= 2:
                break
            box = candidate["box"]
            confidence = float(candidate.get("conf", 0.0))
            area_ratio = module.box_area(box) / max(1, image_w * image_h)
            if confidence < 0.42 or not 0.012 <= area_ratio <= 0.44:
                continue
            if module.is_group_detection(candidate, comparison_pool, image_w, image_h):
                continue
            existing_boxes = [item["box"] for item in accepted]
            if any(module.detections_are_duplicates(box, other) for other in existing_boxes):
                continue
            represented = represented_fraction(box, existing_boxes)
            maximum_overlap = max(
                (module.intersection_area(box, other) / max(1, module.box_area(box)) for other in existing_boxes),
                default=0.0,
            )
            # Old models may fill genuinely empty regions, but may not add a
            # second interpretation of an already represented panel.
            if represented >= 0.34 or maximum_overlap >= 0.30:
                continue
            accepted.append(candidate)
            rescued.append(candidate)
        return accepted, rescued

    module.add_legacy_rescues = conservative_legacy_rescues

    def support_for(box, captured_raw, minimum_iou=0.55):
        support = {}
        for source, detections in captured_raw.items():
            best = None
            best_iou = 0.0
            for item in detections:
                overlap = module.box_iou(box, item["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best = item
            if best is not None and best_iou >= minimum_iou:
                support[source] = {
                    "iou": round(best_iou, 4),
                    "conf": round(float(best.get("conf", 0.0)), 4),
                }
        return support

    def relaxed_containment(smaller, larger):
        return module.intersection_area(smaller, larger) / max(1, module.box_area(smaller))

    def aligned_subset_kind(smaller, larger):
        if relaxed_containment(smaller, larger) < 0.86:
            return None
        sx1, sy1, sx2, sy2 = smaller
        lx1, ly1, lx2, ly2 = larger
        sw, sh = max(1, sx2 - sx1), max(1, sy2 - sy1)
        lw, lh = max(1, lx2 - lx1), max(1, ly2 - ly1)
        height_similarity = min(sh, lh) / max(sh, lh)
        vertical_overlap = max(0, min(sy2, ly2) - max(sy1, ly1)) / max(1, min(sh, lh))
        if (
            height_similarity >= 0.86
            and vertical_overlap >= 0.88
            and sw / lw <= 0.62
            and (abs(sx1 - lx1) <= 0.11 * lw or abs(sx2 - lx2) <= 0.11 * lw)
        ):
            return "vertical_subset"
        width_similarity = min(sw, lw) / max(sw, lw)
        horizontal_overlap = max(0, min(sx2, lx2) - max(sx1, lx1)) / max(1, min(sw, lw))
        if (
            width_similarity >= 0.70
            and horizontal_overlap >= 0.88
            and sh / lh <= 0.74
            and (abs(sy1 - ly1) <= 0.12 * lh or abs(sy2 - ly2) <= 0.12 * lh)
        ):
            return "horizontal_subset"
        return None

    def separator_score(image, smaller, larger, kind):
        gray = image.convert("L")
        sx1, sy1, sx2, sy2 = smaller
        lx1, ly1, lx2, ly2 = larger
        radius = max(8, round(min(gray.size) * 0.007))
        best = 0.0
        if kind == "vertical_subset":
            boundary = sx2 if abs(sx1 - lx1) < abs(sx2 - lx2) else sx1
            start = max(sy1, ly1) + max(2, round(min(sy2, ly2) - max(sy1, ly1)) // 25)
            end = min(sy2, ly2) - max(2, round(min(sy2, ly2) - max(sy1, ly1)) // 25)
            for position in range(max(0, boundary - radius), min(gray.width, boundary + radius + 1)):
                values = list(gray.crop((position, start, position + 1, end)).getdata())
                if values:
                    best = max(
                        best,
                        sum(value <= 38 for value in values) / len(values),
                        sum(value >= 220 for value in values) / len(values),
                    )
        else:
            boundary = sy2 if abs(sy1 - ly1) < abs(sy2 - ly2) else sy1
            start = max(sx1, lx1) + max(2, round(min(sx2, lx2) - max(sx1, lx1)) // 25)
            end = min(sx2, lx2) - max(2, round(min(sx2, lx2) - max(sx1, lx1)) // 25)
            for position in range(max(0, boundary - radius), min(gray.height, boundary + radius + 1)):
                values = list(gray.crop((start, position, end, position + 1)).getdata())
                if values:
                    best = max(
                        best,
                        sum(value <= 38 for value in values) / len(values),
                        sum(value >= 220 for value in values) / len(values),
                    )
        return best

    def clipped_union_area(container, boxes):
        x1, y1, x2, y2 = container
        clipped = []
        for other in boxes:
            ox1, oy1, ox2, oy2 = other
            rect = (max(x1, ox1), max(y1, oy1), min(x2, ox2), min(y2, oy2))
            if rect[2] > rect[0] and rect[3] > rect[1]:
                clipped.append(rect)
        if not clipped:
            return 0
        xs = sorted({value for rect in clipped for value in (rect[0], rect[2])})
        total = 0
        for left, right in zip(xs, xs[1:]):
            intervals = sorted(
                (rect[1], rect[3]) for rect in clipped if rect[0] < right and rect[2] > left
            )
            covered = 0
            if intervals:
                start, end = intervals[0]
                for next_start, next_end in intervals[1:]:
                    if next_start <= end:
                        end = max(end, next_end)
                    else:
                        covered += end - start
                        start, end = next_start, next_end
                covered += end - start
            total += (right - left) * covered
        return total

    def high_text_risks(box, analysis, image_w, image_h):
        return [
            risk
            for risk in module.diagnose_text_crop_edge(box, box, analysis, image_w, image_h)
            if risk.get("severity") == "high"
        ]

    def filter_bad_fragments(boxes, image, captured_raw):
        image_w, image_h = image.size
        page_area = max(1, image_w * image_h)
        analysis = module.analyze_page_geometry(image)
        removed = set()
        details = []

        # Complete-parent dominance: if one crop is merely an aligned, short
        # or narrow version of a complete panel, keep the complete panel even
        # when several old detectors agreed on the truncated crop.
        for smaller in boxes:
            if tuple(smaller) in removed:
                continue
            for larger in boxes:
                if smaller == larger or module.box_area(smaller) >= module.box_area(larger):
                    continue
                kind = aligned_subset_kind(smaller, larger)
                if kind is None:
                    continue
                larger_ratio = module.box_area(larger) / page_area
                if larger_ratio >= 0.48:
                    continue
                boundary_score = separator_score(image, smaller, larger, kind)
                if boundary_score >= 0.78:
                    continue
                removed.add(tuple(smaller))
                details.append(
                    {
                        "box": list(smaller),
                        "reason": "TRUNCATED_DUPLICATE_WITH_COMPLETE_ALTERNATIVE",
                        "kind": kind,
                        "larger_box": list(larger),
                        "separator_score": round(boundary_score, 4),
                        "support": support_for(smaller, captured_raw),
                    }
                )
                break

        survivors = [box for box in boxes if tuple(box) not in removed]

        # A real inset panel may legitimately sit inside a larger scene. It is
        # removed only when its own boundary cuts text that is fully readable
        # in the larger alternative.
        for candidate in survivors:
            if tuple(candidate) in removed:
                continue
            candidate_area = max(1, module.box_area(candidate))
            risks = high_text_risks(candidate, analysis, image_w, image_h)
            if not risks:
                continue
            for larger in survivors:
                larger_area = module.box_area(larger)
                if candidate == larger or larger_area <= candidate_area:
                    continue
                if larger_area / page_area >= 0.48:
                    continue
                if relaxed_containment(candidate, larger) < 0.88:
                    continue
                if candidate_area / max(1, larger_area) > 0.76:
                    continue
                covered = [
                    risk
                    for risk in risks
                    if module.containment_ratio(tuple(risk["text_box"]), larger) >= 0.94
                ]
                if not covered:
                    continue
                removed.add(tuple(candidate))
                details.append(
                    {
                        "box": list(candidate),
                        "reason": "TEXT_CUT_SUBSET_WITH_COMPLETE_ALTERNATIVE",
                        "larger_box": list(larger),
                        "cut_text_regions": [risk["text_box"] for risk in covered],
                        "support": support_for(candidate, captured_raw),
                    }
                )
                break

        survivors = [box for box in survivors if tuple(box) not in removed]

        # Intersection crops such as Torn output 191 are mostly the shared
        # region of several larger candidates and cut text at their boundary.
        for candidate in survivors:
            if tuple(candidate) in removed:
                continue
            candidate_area = max(1, module.box_area(candidate))
            larger = [
                other
                for other in survivors
                if other != candidate
                and module.box_area(other) >= 1.22 * candidate_area
                and module.intersection_area(candidate, other) / candidate_area >= 0.16
            ]
            if len(larger) < 2:
                continue
            union_ratio = clipped_union_area(candidate, larger) / candidate_area
            risks = high_text_risks(candidate, analysis, image_w, image_h)
            covered_risks = [
                risk
                for risk in risks
                if any(
                    module.containment_ratio(tuple(risk["text_box"]), other) >= 0.94
                    for other in larger
                )
            ]
            if union_ratio < 0.68 or not covered_risks:
                continue
            removed.add(tuple(candidate))
            details.append(
                {
                    "box": list(candidate),
                    "reason": "OVERLAP_INTERSECTION_CROP_WITH_CUT_TEXT",
                    "larger_boxes": [list(box) for box in larger],
                    "represented_ratio": round(union_ratio, 4),
                    "cut_text_regions": [risk["text_box"] for risk in covered_risks],
                    "support": support_for(candidate, captured_raw),
                }
            )

        return [box for box in boxes if tuple(box) not in removed], details, analysis

    def cluster_values(values, tolerance):
        groups = []
        for index, value in sorted(values, key=lambda item: item[1]):
            if not groups or value - groups[-1][-1][1] > tolerance:
                groups.append([(index, value)])
            else:
                groups[-1].append((index, value))
        return groups

    def align_shared_edges(boxes, analysis, image_w, image_h):
        adjusted = [list(box) for box in boxes]
        details = []
        y_tolerance = max(8, round(image_h * 0.018))
        x_tolerance = max(8, round(image_w * 0.018))

        for edge_index, edge_name in ((1, "top"), (3, "bottom")):
            values = [(index, box[edge_index]) for index, box in enumerate(adjusted)]
            for group in cluster_values(values, y_tolerance):
                if len(group) < 2:
                    continue
                indices = [index for index, _ in group]
                original = round(median(value for _, value in group))
                if original <= 0.04 * image_h or original >= 0.96 * image_h:
                    continue
                span_start = min(adjusted[index][0] for index in indices)
                span_end = max(adjusted[index][2] for index in indices)
                if span_end - span_start < 0.52 * image_w:
                    continue
                target = module.best_horizontal_boundary(
                    analysis,
                    original,
                    span_start,
                    span_end,
                    max(y_tolerance, round(image_h * 0.035)),
                )
                if target == original:
                    continue
                for index in indices:
                    old = adjusted[index][edge_index]
                    candidate = list(adjusted[index])
                    candidate[edge_index] = target
                    if candidate[3] - candidate[1] >= 0.72 * (adjusted[index][3] - adjusted[index][1]):
                        adjusted[index] = candidate
                        details.append({"box_index": index, "edge": edge_name, "from": old, "to": target})

        for edge_index, edge_name in ((0, "left"), (2, "right")):
            values = [(index, box[edge_index]) for index, box in enumerate(adjusted)]
            for group in cluster_values(values, x_tolerance):
                if len(group) < 2:
                    continue
                indices = [index for index, _ in group]
                original = round(median(value for _, value in group))
                if original <= 0.04 * image_w or original >= 0.96 * image_w:
                    continue
                span_start = min(adjusted[index][1] for index in indices)
                span_end = max(adjusted[index][3] for index in indices)
                if span_end - span_start < 0.52 * image_h:
                    continue
                target = module.best_vertical_boundary(
                    analysis,
                    original,
                    span_start,
                    span_end,
                    max(x_tolerance, round(image_w * 0.035)),
                )
                if target == original:
                    continue
                for index in indices:
                    old = adjusted[index][edge_index]
                    candidate = list(adjusted[index])
                    candidate[edge_index] = target
                    if candidate[2] - candidate[0] >= 0.72 * (adjusted[index][2] - adjusted[index][0]):
                        adjusted[index] = candidate
                        details.append({"box_index": index, "edge": edge_name, "from": old, "to": target})
        return [tuple(box) for box in adjusted], details

    def complementary_slice_pair(candidates, image_w, image_h):
        """Recover two readable halves only when they jointly explain a page."""
        page_area = max(1, image_w * image_h)
        eligible = [
            item
            for item in candidates
            if float(item.get("conf", 0.0)) >= 0.40
            and module.box_area(item["box"]) / page_area >= 0.26
        ]
        best = None
        best_score = -1.0
        for first_index, first in enumerate(eligible):
            for second in eligible[first_index + 1 :]:
                a = first["box"]
                b = second["box"]
                aw, ah = max(1, a[2] - a[0]), max(1, a[3] - a[1])
                bw, bh = max(1, b[2] - b[0]), max(1, b[3] - b[1])
                inter = module.intersection_area(a, b)
                minimum_area = max(1, min(module.box_area(a), module.box_area(b)))
                overlap = inter / minimum_area
                horizontal_pair = (
                    aw >= 0.82 * image_w
                    and bw >= 0.82 * image_w
                    and 0.28 * image_h <= ah <= 0.68 * image_h
                    and 0.28 * image_h <= bh <= 0.68 * image_h
                    and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 >= 0.25 * image_h
                    and overlap <= 0.18
                )
                vertical_pair = (
                    ah >= 0.82 * image_h
                    and bh >= 0.82 * image_h
                    and 0.28 * image_w <= aw <= 0.72 * image_w
                    and 0.28 * image_w <= bw <= 0.72 * image_w
                    and abs((a[0] + a[2]) - (b[0] + b[2])) / 2 >= 0.22 * image_w
                    and overlap <= 0.55
                )
                if not (horizontal_pair or vertical_pair):
                    continue
                union_ratio = (
                    module.box_area(a) + module.box_area(b) - inter
                ) / page_area
                if union_ratio < 0.72:
                    continue
                score = union_ratio + 0.08 * (
                    float(first.get("conf", 0.0)) + float(second.get("conf", 0.0))
                )
                if score > best_score:
                    best_score = score
                    best = [first, second]
        if best is None:
            return []
        return sorted(best, key=lambda item: (item["box"][1], item["box"][0]))

    def horizontal_boundary_strength(image, y, x1, x2):
        gray = image.convert("L")
        radius = max(8, round(gray.height * 0.006))
        start = max(0, x1 + round((x2 - x1) * 0.04))
        end = min(gray.width, x2 - round((x2 - x1) * 0.04))
        best = 0.0
        for position in range(max(0, y - radius), min(gray.height, y + radius + 1)):
            values = list(gray.crop((start, position, end, position + 1)).getdata())
            if values:
                best = max(
                    best,
                    sum(value <= 38 for value in values) / len(values),
                    sum(value >= 220 for value in values) / len(values),
                )
        return best

    def vertical_boundary_strength(image, x, y1, y2):
        gray = image.convert("L")
        radius = max(8, round(gray.width * 0.006))
        start = max(0, y1 + round((y2 - y1) * 0.04))
        end = min(gray.height, y2 - round((y2 - y1) * 0.04))
        best = 0.0
        for position in range(max(0, x - radius), min(gray.width, x + radius + 1)):
            values = list(gray.crop((position, start, position + 1, end)).getdata())
            if values:
                best = max(
                    best,
                    sum(value <= 38 for value in values) / len(values),
                    sum(value >= 220 for value in values) / len(values),
                )
        return best

    def horizontal_dark_boundary_strength(image, y, x1, x2):
        """Measure ink-like borders without mistaking a pale panel for a gutter."""
        gray = image.convert("L")
        radius = max(8, round(gray.height * 0.006))
        start = max(0, x1 + round((x2 - x1) * 0.04))
        end = min(gray.width, x2 - round((x2 - x1) * 0.04))
        best = 0.0
        for position in range(max(0, y - radius), min(gray.height, y + radius + 1)):
            values = list(gray.crop((start, position, end, position + 1)).getdata())
            if values:
                best = max(best, sum(value <= 38 for value in values) / len(values))
        return best

    def merge_row_fragments(boxes, image):
        """Join two pieces of one wide row panel when no real gutter exists."""
        image_w, image_h = image.size
        merged = [tuple(box) for box in boxes]
        details = []
        changed = True
        while changed:
            changed = False
            for first_index in range(len(merged)):
                if changed:
                    break
                for second_index in range(first_index + 1, len(merged)):
                    first = merged[first_index]
                    second = merged[second_index]
                    left, right = sorted((first, second), key=lambda box: box[0])
                    left_w = max(1, left[2] - left[0])
                    right_w = max(1, right[2] - right[0])
                    left_h = max(1, left[3] - left[1])
                    right_h = max(1, right[3] - right[1])
                    vertical_overlap = max(
                        0, min(left[3], right[3]) - max(left[1], right[1])
                    ) / max(1, min(left_h, right_h))
                    if vertical_overlap < 0.88:
                        continue
                    if abs(left[1] - right[1]) > 0.04 * image_h:
                        continue
                    if abs(left[3] - right[3]) > 0.04 * image_h:
                        continue
                    gap = right[0] - left[2]
                    # A mere touch/gap can be a real straight or diagonal
                    # gutter. Require a material detector overlap so this
                    # rule only repairs duplicate fragments of one panel.
                    if not -0.06 * image_w <= gap <= -0.018 * image_w:
                        continue
                    if min(left_w, right_w) > 0.36 * image_w:
                        continue
                    if max(left_w, right_w) < 0.45 * image_w:
                        continue
                    union_width = max(left[2], right[2]) - min(left[0], right[0])
                    if union_width < 1.15 * max(left_w, right_w):
                        continue
                    boundary = round((left[2] + right[0]) / 2)
                    strength = vertical_boundary_strength(
                        image,
                        boundary,
                        max(left[1], right[1]),
                        min(left[3], right[3]),
                    )
                    if strength >= 0.62:
                        continue
                    combined = (
                        min(left[0], right[0]),
                        min(left[1], right[1]),
                        max(left[2], right[2]),
                        max(left[3], right[3]),
                    )
                    details.append(
                        {
                            "boxes": [list(left), list(right)],
                            "merged_box": list(combined),
                            "reason": "ROW_FRAGMENTS_WITHOUT_REAL_GUTTER",
                            "separator_score": round(strength, 4),
                        }
                    )
                    merged = [
                        box
                        for index, box in enumerate(merged)
                        if index not in {first_index, second_index}
                    ]
                    merged.append(combined)
                    changed = True
                    break
        return merged, details

    def complete_short_columns(boxes, image, analysis):
        """Extend an early-stopping column to its peers if no border ends it."""
        image_w, image_h = image.size
        adjusted = [list(box) for box in boxes]
        details = []
        top_tolerance = max(10, round(image_h * 0.025))
        values = [(index, box[1]) for index, box in enumerate(adjusted)]
        for group in cluster_values(values, top_tolerance):
            if len(group) < 3:
                continue
            indices = [index for index, _ in group]
            span_start = min(adjusted[index][0] for index in indices)
            span_end = max(adjusted[index][2] for index in indices)
            if span_end - span_start < 0.70 * image_w:
                continue
            for index in indices:
                box = adjusted[index]
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width > 0.32 * image_w or height < 0.35 * image_h:
                    continue
                longer = [
                    adjusted[other][3]
                    for other in indices
                    if other != index
                    and adjusted[other][3] - box[3] >= 0.14 * image_h
                ]
                if len(longer) < 2:
                    continue
                strength = horizontal_dark_boundary_strength(
                    image, box[3], box[0], box[2]
                )
                if strength >= 0.45:
                    continue
                expected = round(median(longer))
                target = module.best_horizontal_boundary(
                    analysis,
                    expected,
                    box[0],
                    box[2],
                    max(12, round(image_h * 0.04)),
                )
                target = max(expected - round(image_h * 0.04), target)
                target = min(image_h, target)
                if target - box[3] < 0.10 * image_h:
                    continue
                old_bottom = box[3]
                box[3] = target
                details.append(
                    {
                        "box_index": index,
                        "from": old_bottom,
                        "to": target,
                        "reason": "COLUMN_ENDED_WITHOUT_REAL_BORDER",
                        "separator_score": round(strength, 4),
                    }
                )
        return [tuple(box) for box in adjusted], details

    def suppress_low_value_page_crops(boxes, image):
        """Avoid emitting cover/splash crops that mostly repeat the original."""
        image_w, image_h = image.size
        page_area = max(1, image_w * image_h)
        if len(boxes) == 1:
            box = boxes[0]
            width = box[2] - box[0]
            height = box[3] - box[1]
            area_ratio = module.box_area(box) / page_area
            if (
                area_ratio >= 0.56
                and width >= 0.75 * image_w
                and height >= 0.70 * image_h
            ):
                return [], [
                    {
                        "box": list(box),
                        "reason": "SINGLE_GIANT_CROP_REPEATS_ORIGINAL",
                        "page_area_ratio": round(area_ratio, 4),
                    }
                ]

        if 2 <= len(boxes) <= 4 and all(
            box[2] - box[0] >= 0.92 * image_w for box in boxes
        ):
            heights = [box[3] - box[1] for box in boxes]
            coverage = clipped_union_area((0, 0, image_w, image_h), boxes) / page_area
            redundant_pair = any(
                module.intersection_area(first, second)
                / max(1, min(module.box_area(first), module.box_area(second)))
                >= 0.28
                for first_index, first in enumerate(boxes)
                for second in boxes[first_index + 1 :]
            )
            if max(heights) >= 0.70 * image_h and coverage >= 0.78 and redundant_pair:
                return [], [
                    {
                        "boxes": [list(box) for box in boxes],
                        "reason": "OVERLAPPING_BROAD_SLICES_REPEAT_ORIGINAL",
                        "page_coverage": round(coverage, 4),
                    }
                ]
        return boxes, []

    def resolve_overlapping_layouts(boxes, image):
        """Remove partial re-interpretations and trim two-column overlaps."""
        image_w, image_h = image.size
        page_area = max(1, image_w * image_h)
        removed = set()
        removal_details = []

        # Nested partial crops created by dirty/old labels: keep the complete
        # alternative unless the partial crop has a strong real panel border.
        for smaller in boxes:
            smaller_area = module.box_area(smaller)
            for larger in boxes:
                larger_area = module.box_area(larger)
                if smaller == larger or smaller_area >= 0.62 * larger_area:
                    continue
                if relaxed_containment(smaller, larger) < 0.92:
                    continue
                sx1, sy1, sx2, sy2 = smaller
                lx1, ly1, lx2, ly2 = larger
                lw, lh = max(1, lx2 - lx1), max(1, ly2 - ly1)
                shares_edge = (
                    abs(sx1 - lx1) <= 0.10 * lw
                    or abs(sx2 - lx2) <= 0.10 * lw
                    or abs(sy1 - ly1) <= 0.10 * lh
                    or abs(sy2 - ly2) <= 0.10 * lh
                )
                if not shares_edge or larger_area / page_area >= 0.48:
                    continue
                if (sy2 - sy1) / lh <= (sx2 - sx1) / lw:
                    boundary_y = sy1 if abs(sy1 - ly1) > abs(sy2 - ly2) else sy2
                    strength = horizontal_boundary_strength(image, boundary_y, sx1, sx2)
                else:
                    kind = "vertical_subset"
                    strength = separator_score(image, smaller, larger, kind)
                if strength >= 0.76:
                    continue
                removed.add(tuple(smaller))
                removal_details.append(
                    {
                        "box": list(smaller),
                        "reason": "NESTED_PARTIAL_WITH_COMPLETE_ALTERNATIVE",
                        "larger_box": list(larger),
                        "separator_score": round(strength, 4),
                    }
                )
                break

        survivors = [box for box in boxes if tuple(box) not in removed]

        # A broad lower strip formed from the shared region of two taller
        # panels is not a third panel (Torn's former 190/191 pattern).
        for candidate in survivors:
            if tuple(candidate) in removed:
                continue
            cx1, cy1, cx2, cy2 = candidate
            candidate_area = max(1, module.box_area(candidate))
            contributors = []
            for other in survivors:
                if other == candidate:
                    continue
                ox1, oy1, ox2, oy2 = other
                contribution = module.intersection_area(candidate, other) / candidate_area
                extends_above = cy1 - oy1 >= 0.18 * max(1, cy2 - cy1)
                if contribution >= 0.22 and extends_above:
                    contributors.append(other)
            if len(contributors) < 2:
                continue
            union_ratio = clipped_union_area(candidate, contributors) / candidate_area
            boundary_strength = horizontal_boundary_strength(image, cy1, cx1, cx2)
            if union_ratio < 0.80 or boundary_strength >= 0.72:
                continue
            removed.add(tuple(candidate))
            removal_details.append(
                {
                    "box": list(candidate),
                    "reason": "OVERLAPPING_LOWER_STRIP_REINTERPRETATION",
                    "contributors": [list(other) for other in contributors],
                    "represented_ratio": round(union_ratio, 4),
                    "separator_score": round(boundary_strength, 4),
                }
            )

        survivors = [box for box in survivors if tuple(box) not in removed]
        trimmed = [list(box) for box in survivors]
        trim_details = []

        # When a left-anchored and right-anchored tall panel overlap, use the
        # right panel's leading edge as the left crop's clean stopping point.
        for left_index, left in enumerate(trimmed):
            lx1, ly1, lx2, ly2 = left
            if lx1 > 0.04 * image_w:
                continue
            for right_index, right in enumerate(trimmed):
                if left_index == right_index:
                    continue
                rx1, ry1, rx2, ry2 = right
                if rx2 < 0.94 * image_w or not rx1 < lx2 < rx2:
                    continue
                overlap_width = lx2 - rx1
                if not 0.08 * image_w <= overlap_width <= 0.30 * image_w:
                    continue
                vertical_overlap = max(0, min(ly2, ry2) - max(ly1, ry1))
                vertical_overlap /= max(1, min(ly2 - ly1, ry2 - ry1))
                if vertical_overlap < 0.72 or rx1 - lx1 < 0.25 * image_w:
                    continue
                old = tuple(left)
                left[2] = rx1
                trim_details.append(
                    {
                        "box_index": left_index,
                        "from": list(old),
                        "to": list(left),
                        "reason": "TWO_COLUMN_OVERLAP_TRIM",
                        "right_neighbor": list(right),
                    }
                )
                break

        return [tuple(box) for box in trimmed], removal_details, trim_details

    def detect_hybrid_v6(*args, **kwargs):
        image = args[3] if len(args) > 3 else kwargs["image"]
        captured_raw = {}

        def capture(model, frame, confidence, source_name, *extra_args, **extra_kwargs):
            effective_confidence = confidence
            if frame.size == image.size and source_name == "base":
                effective_confidence = max(confidence, 0.45)
            elif frame.size == image.size and source_name == "legacy":
                effective_confidence = max(confidence, 0.40)
            detections = original_get_frame_detections(
                model,
                frame,
                effective_confidence,
                source_name,
                *extra_args,
                **extra_kwargs,
            )
            if frame.size == image.size and source_name in {"clean", "legacy", "base"}:
                captured_raw.setdefault(source_name, [dict(item) for item in detections])
            return detections

        module.get_frame_detections = capture
        try:
            boxes, clean_count, legacy_count, base_count, diagnostics = original_detect_hybrid(
                *args, **kwargs
            )
        finally:
            module.get_frame_detections = original_get_frame_detections

        slice_rescues = []
        if not boxes:
            legacy_candidates = module.geometry_filter(
                captured_raw.get("legacy", []), image.width, image.height
            )
            slice_rescues = complementary_slice_pair(
                legacy_candidates, image.width, image.height
            )
            if slice_rescues:
                boxes = [item["box"] for item in slice_rescues]

        filtered, removals, analysis = filter_bad_fragments(boxes, image, captured_raw)
        resolved, layout_removals, overlap_trims = resolve_overlapping_layouts(filtered, image)
        merged, row_merges = merge_row_fragments(resolved, image)
        completed, column_completions = complete_short_columns(merged, image, analysis)
        aligned, alignments = align_shared_edges(
            completed, analysis, image.width, image.height
        )
        aligned = module.sort_panels(aligned, image.width, image.height)
        aligned, repeat_suppressions = suppress_low_value_page_crops(aligned, image)
        if slice_rescues:
            stale_reason = "MODEL_ACTIVITY_BUT_NO_FINAL_PANEL"
            if stale_reason in diagnostics.get("reasons", []):
                diagnostics["reasons"].remove(stale_reason)
                diagnostics["score"] = max(0, int(diagnostics.get("score", 0)) - 2)

            first = slice_rescues[0]["box"]
            second = slice_rescues[1]["box"]
            first_area = module.box_area(first)
            second_area = module.box_area(second)
            overlap_x = max(0, min(first[2], second[2]) - max(first[0], second[0]))
            overlap_y = max(0, min(first[3], second[3]) - max(first[1], second[1]))
            union_area = first_area + second_area - (overlap_x * overlap_y)
            diagnostics["page_coverage"] = round(
                union_area / max(1, image.width * image.height), 4
            )
            diagnostics["suspicious"] = (
                diagnostics["score"] >= module.DIAGNOSTIC_SCORE_THRESHOLD
            )
        diagnostics["v6_clean_primary"] = str(primary_model)
        diagnostics["v6_conservative_rescue_model"] = str(rescue_model)
        diagnostics["v6_complementary_slice_rescue_count"] = len(slice_rescues)
        diagnostics["v6_complementary_slice_rescues"] = [
            {"box": list(item["box"]), "conf": round(float(item.get("conf", 0.0)), 4)}
            for item in slice_rescues
        ]
        diagnostics["v6_old_model_policy"] = "uncovered_regions_only_max_2"
        diagnostics["v6_full_page_thresholds"] = {
            "clean": module.CLEAN_CONF,
            "legacy": 0.40,
            "base": 0.45,
        }
        diagnostics["v6_fragment_removed_count"] = len(removals)
        diagnostics["v6_fragment_removals"] = removals
        diagnostics["v6_layout_removed_count"] = len(layout_removals)
        diagnostics["v6_layout_removals"] = layout_removals
        diagnostics["v6_overlap_trim_count"] = len(overlap_trims)
        diagnostics["v6_overlap_trims"] = overlap_trims
        diagnostics["v6_row_merge_count"] = len(row_merges)
        diagnostics["v6_row_merges"] = row_merges
        diagnostics["v6_column_completion_count"] = len(column_completions)
        diagnostics["v6_column_completions"] = column_completions
        diagnostics["v6_repeat_suppression_count"] = len(repeat_suppressions)
        diagnostics["v6_repeat_suppressions"] = repeat_suppressions
        diagnostics["v6_shared_edge_adjustment_count"] = len(alignments)
        diagnostics["v6_shared_edge_adjustments"] = alignments
        diagnostics["final_boxes"] = [list(box) for box in aligned]
        return aligned, clean_count, legacy_count, base_count, diagnostics

    module.detect_hybrid = detect_hybrid_v6
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--primary-model", type=Path, default=V6_MODEL)
    parser.add_argument("--rescue-model", type=Path, default=V3_RESCUE_MODEL)
    args = parser.parse_args()
    pipeline = load_pipeline(args.primary_model, args.rescue_model)
    pipeline.convert_comic(args.input, args.output)


if __name__ == "__main__":
    main()
