# SPDX-License-Identifier: AGPL-3.0-only
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
from ultralytics import YOLO
import torch
import cv2
import numpy as np

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
import statistics


# Prevent archive helpers (tar.exe / 7z.exe / WinRAR.exe) from opening a
# console window when the GUI application is running on Windows.
WINDOWLESS_SUBPROCESS = (
  {
    "creationflags": getattr(
      subprocess,
      "CREATE_NO_WINDOW",
      0
    )
  }
  if os.name == "nt"
  else {}
)


# ============================================================
# COMIC KOBO CONVERTER
# ============================================================
#
# PURPOSE:
#
# 1. Use the newly trained V4 clean panel model.
# 2. Keep stable 768px detection.
# 3. Keep the successful V2.1 group resolver.
# 4. Keep the successful V2.1 top-edge reading order.
# 5. Suppress near-full-page duplicate crops.
# 6. Rescue small/narrow panels only when multiple models agree.
# 7. Re-scan only suspicious removed groups at 1024px.
# 8. Remain completely automatic for the end user.
#
# Stable marvel_kobo.py is NOT touched.
#
# ============================================================


BASE_DIR = Path(__file__).resolve().parent


def get_app_directory():
  """Return the folder containing the EXE when frozen, otherwise this script."""
  if getattr(sys, "frozen", False):
    return Path(sys.executable).resolve().parent

  return BASE_DIR


def get_comics_output_directory():
  output_dir = get_app_directory() / "Comics"
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir


def get_diagnostics_directory(
  input_path
):
  diagnostics_dir = (
    get_app_directory()
    / "Diagnostics"
    / (
      Path(input_path).stem
      + "_V58_SMART_MERGE"
    )
  )
  diagnostics_dir.mkdir(
    parents=True,
    exist_ok=True
  )
  return diagnostics_dir

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Experimental speed switch used only by the regression harness. Production
# remains FP32 until the pixel/panel parity suite proves FP16 is equivalent.
USE_FP16_INFERENCE = (
  os.environ.get("CEC_FP16_INFERENCE", "0") == "1"
  and DEVICE != "cpu"
)

# Keep the models in memory while the app processes a batch.
_CLEAN_MODEL_CACHE = None
_LEGACY_MODEL_CACHE = None
_BASE_MODEL_CACHE = None


# ============================================================
# MODELS
# ============================================================

CLEAN_MODEL_PATH = (
  BASE_DIR
  / "comic_kobo_clean_v4_best.pt"
)

LEGACY_MODEL_PATH = (
  BASE_DIR
  / "runs"
  / "detect"
  / "runs_marvel_clean"
  / "marvel_clean_v3"
  / "weights"
  / "best.pt"
)

BASE_MODEL_PATH = (
  BASE_DIR
  / "comic_base.pt"
)


# ============================================================
# KOBO OUTPUT
# ============================================================

KOBO_W = 1072
KOBO_H = 1448

SCREEN_MARGIN = 8

TARGET_W = KOBO_W - SCREEN_MARGIN * 2
TARGET_H = KOBO_H - SCREEN_MARGIN * 2


# ============================================================
# DETECTION
# ============================================================

IMAGE_SIZE = 768

# The trained models and the main pipeline remain at 768px. A 1024px
# pass is used only inside a large group that the resolver has removed.
LOCAL_RESCUE_IMAGE_SIZE = 1024
LOCAL_RESCUE_MAX_GROUPS_PER_PAGE = 2
LOCAL_RESCUE_MIN_PARENT_PAGE_AREA = 0.22
LOCAL_RESCUE_MAX_PARENT_PAGE_AREA = 0.72
LOCAL_RESCUE_MAX_CHILD_COVERAGE = 0.84
LOCAL_RESCUE_MAX_CANDIDATE_PARENT_AREA = 0.68
LOCAL_RESCUE_EXISTING_OVERLAP = 0.72
LOCAL_RESCUE_PARENT_CONTAINMENT = 0.80
LOCAL_RESCUE_MAX_EXISTING_PANELS = 6
LOCAL_RESCUE_MIN_MODEL_SUPPORT = 2
LOCAL_RESCUE_SUPPORT_IOU = 0.55
LOCAL_RESCUE_MAX_ASPECT_RATIO = 2.00
LOCAL_RESCUE_MAX_ACCEPTED_PER_GROUP = 1

CLEAN_CONF = 0.25
LEGACY_CONF = 0.25
BASE_CONF = 0.30

MODEL_IOU = 0.45

# Containment alone must not make a real child panel a duplicate of a
# much larger group box. Genuine duplicate detections have similar area.
MERGE_MIN_AREA_SIMILARITY = 0.65
SMART_MERGE_MIN_BASE_CONF = 0.85
SMART_MERGE_MIN_CONF_GAIN = 0.12
SMART_MERGE_MIN_AREA_RATIO = 0.55
SMART_MERGE_MAX_AREA_RATIO = 0.80
SMART_MERGE_MAX_GROUP_CHILD_COVERAGE = 0.45


# ============================================================
# GEOMETRY FILTER
# ============================================================

MIN_PANEL_AREA_RATIO = 0.04

MIN_WIDTH_RATIO = 0.16
MIN_HEIGHT_RATIO = 0.10

MAX_ASPECT_RATIO = 5.5


# ============================================================
# CONSENSUS SMALL-PANEL RESCUE
# ============================================================

# The normal geometry limits remain authoritative. These relaxed
# limits are used ONLY when at least two different detector models agree
# that the same small / narrow region is a real comic panel.
CONSENSUS_RESCUE_MIN_AREA_RATIO = 0.022
CONSENSUS_RESCUE_MIN_WIDTH_RATIO = 0.09
CONSENSUS_RESCUE_MIN_HEIGHT_RATIO = 0.08
CONSENSUS_RESCUE_MAX_ASPECT_RATIO = 6.5
CONSENSUS_RESCUE_MAX_PAGE_AREA = 0.18

CONSENSUS_RESCUE_MIN_MODEL_SUPPORT = 2
CONSENSUS_RESCUE_SUPPORT_IOU = 0.50
CONSENSUS_RESCUE_SUPPORT_CONTAINMENT = 0.82
CONSENSUS_RESCUE_MAX_CENTER_DISTANCE = 0.24

CONSENSUS_RESCUE_MIN_BEST_CONF = 0.45
CONSENSUS_RESCUE_MIN_AVG_CONF = 0.35


# ============================================================
# NEAR-FULL-PAGE SUPPRESSION
# ============================================================

# The original page is always appended, so a model box that is
# effectively the whole page only creates a redundant duplicate.
NEAR_FULL_PAGE_MIN_WIDTH_RATIO = 0.90
NEAR_FULL_PAGE_MIN_HEIGHT_RATIO = 0.85


# ============================================================
# SAFE CROP
# ============================================================

PAD_LEFT = 0.09
PAD_RIGHT = 0.09

PAD_TOP = 0.13
PAD_BOTTOM = 0.11

NEIGHBOR_AXIS_OVERLAP = 0.30


# ============================================================
# BORDER / GUTTER SNAP
# ============================================================

BORDER_SNAP_PAGE_RANGE = 0.025
BORDER_SNAP_BOX_RANGE = 0.08
BORDER_SNAP_MIN_SCORE = 0.17
BORDER_SNAP_DISTANCE_PENALTY = 0.08


# ============================================================
# TEXT / SPEECH-BUBBLE EVIDENCE
# ============================================================

TEXT_MAX_HORIZONTAL_EXPANSION = 0.16
TEXT_MAX_VERTICAL_EXPANSION = 0.18
TEXT_REGION_MARGIN = 0.018


# ============================================================
# CONSERVATIVE OUTER-WHITESPACE TRIM
# ============================================================

def trim_outer_white_space(
  image,
  max_trim_ratio=0.08
):

  rgb = np.asarray(
    image.convert("RGB")
  )

  if rgb.size <= 0:
    return image

  gray = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2GRAY
  )

  h, w = gray.shape

  if w < 40 or h < 40:
    return image

  max_x = max(
    1,
    int(w * max_trim_ratio)
  )

  max_y = max(
    1,
    int(h * max_trim_ratio)
  )

  def row_is_blank(y):

    row = gray[y, :]

    return (
      float(np.mean(row >= 245)) >= 0.985
      and
      float(np.std(row)) <= 8.0
    )

  def col_is_blank(x):

    col = gray[:, x]

    return (
      float(np.mean(col >= 245)) >= 0.985
      and
      float(np.std(col)) <= 8.0
    )

  top = 0

  while (
    top < max_y
    and
    top < h - 1
    and
    row_is_blank(top)
  ):
    top += 1

  bottom = h

  while (
    h - bottom < max_y
    and
    bottom > top + 1
    and
    row_is_blank(bottom - 1)
  ):
    bottom -= 1

  left = 0

  while (
    left < max_x
    and
    left < w - 1
    and
    col_is_blank(left)
  ):
    left += 1

  right = w

  while (
    w - right < max_x
    and
    right > left + 1
    and
    col_is_blank(right - 1)
  ):
    right -= 1

  # Avoid pointless tiny trims.
  if (
    left < 2
    and
    top < 2
    and
    w - right < 2
    and
    h - bottom < 2
  ):
    return image

  return image.crop(
    (
      left,
      top,
      right,
      bottom
    )
  )


# GROUP RESOLVER SETTINGS
# ============================================================

GROUP_CHILD_OVERLAP = 0.45
GROUP_MAX_CHILD_SIZE = 0.72

GROUP_MIN_CHILDREN = 2

GROUP_MIN_CENTER_DISTANCE = 0.14

# Two apparent children are weak evidence: real comic panels often contain
# an inset panel plus one false fragment. Require the pair to explain at
# least half of the parent before discarding that parent as a group.
GROUP_MIN_COVERAGE_TWO = 0.18
GROUP_EDGE_SLICE_PRESERVE_MAX_COVERAGE = 0.50
GROUP_MIN_COVERAGE_THREE = 0.12

GROUP_MIN_CONF_RATIO = 0.50

LARGE_GROUP_PAGE_AREA = 0.24
VERY_LARGE_GROUP_PAGE_AREA = 0.34


# ============================================================
# READING ORDER SETTINGS
# ============================================================

READING_ROW_PAGE_TOLERANCE = 0.045
READING_ROW_PANEL_TOLERANCE = 0.16


# ============================================================
# DEBUG
# ============================================================

DEBUG_RESOLVER = False
DEBUG_READING_ORDER = False
DEBUG_IMAGES = False

# V5.8 keeps the narrow V5.7 group guard and adds a conservative smart-merge
# override for high-confidence, tighter base-model detections.
DIAGNOSTICS_ENABLED = True
# A single heuristic is too noisy on stylized comic pages. Promote a page
# only when several independent warning signals agree. Very low coverage and
# model activity with no surviving panel remain critical on their own.
DIAGNOSTIC_SCORE_THRESHOLD = 5
DIAGNOSTIC_LOW_COVERAGE = 0.30
DIAGNOSTIC_STRONG_CONFIDENCE = 0.42
DIAGNOSTIC_MAX_CANDIDATE_PAGE_AREA = 0.42
DIAGNOSTIC_TEXT_EDGE_MARGIN = 0.012


IMAGE_EXTENSIONS = {
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".bmp",
}


# ============================================================
# PROGRESS
# ============================================================

def report_progress(
  callback,
  stage,
  current=0,
  total=0,
  message=""
):

  if callback is not None:

    callback(
      stage,
      current,
      total,
      message
    )


# ============================================================
# NATURAL SORT
# ============================================================

def natural_sort_key(text):

  return [
    int(part)
    if part.isdigit()
    else part.lower()

    for part in re.split(
      r"(\d+)",
      str(text)
    )
  ]


# ============================================================
# BOX HELPERS
# ============================================================

def clamp(
  value,
  low,
  high
):

  return max(
    low,
    min(
      high,
      value
    )
  )


def box_width(box):

  return max(
    0,
    box[2] - box[0]
  )


def box_height(box):

  return max(
    0,
    box[3] - box[1]
  )


def box_area(box):

  return (
    box_width(box)
    * box_height(box)
  )


def box_center(box):

  return (
    (box[0] + box[2]) / 2,
    (box[1] + box[3]) / 2,
  )


def center_inside(
  inner_box,
  outer_box
):

  cx, cy = box_center(
    inner_box
  )

  x1, y1, x2, y2 = outer_box

  return (
    x1 <= cx <= x2
    and
    y1 <= cy <= y2
  )


# ============================================================
# INTERSECTION
# ============================================================

def intersection_area(
  a,
  b
):

  ax1, ay1, ax2, ay2 = a
  bx1, by1, bx2, by2 = b

  x1 = max(
    ax1,
    bx1
  )

  y1 = max(
    ay1,
    by1
  )

  x2 = min(
    ax2,
    bx2
  )

  y2 = min(
    ay2,
    by2
  )

  if (
    x2 <= x1
    or
    y2 <= y1
  ):

    return 0

  return (
    (x2 - x1)
    * (y2 - y1)
  )


# ============================================================
# IOU
# ============================================================

def box_iou(
  a,
  b
):

  inter = intersection_area(
    a,
    b
  )

  if inter <= 0:
    return 0.0

  union = (
    box_area(a)
    + box_area(b)
    - inter
  )

  if union <= 0:
    return 0.0

  return (
    inter / union
  )


# ============================================================
# CONTAINMENT
# ============================================================

def containment_ratio(
  inner,
  outer
):

  area = box_area(
    inner
  )

  if area <= 0:
    return 0.0

  return (
    intersection_area(
      inner,
      outer
    )
    / area
  )


# ============================================================
# V5.5 DIAGNOSTIC HELPERS
# ============================================================

def union_box_area(
  boxes
):
  """Return the exact union area of a small list of axis-aligned boxes."""

  valid = [
    tuple(box)
    for box in boxes
    if box_area(box) > 0
  ]

  if not valid:
    return 0

  x_values = sorted({
    coordinate
    for box in valid
    for coordinate in (box[0], box[2])
  })

  total = 0

  for left, right in zip(
    x_values,
    x_values[1:]
  ):

    if right <= left:
      continue

    y_intervals = sorted(
      (box[1], box[3])
      for box in valid
      if box[0] < right and box[2] > left
    )

    if not y_intervals:
      continue

    merged_height = 0
    current_start, current_end = y_intervals[0]

    for start, end in y_intervals[1:]:

      if start <= current_end:
        current_end = max(
          current_end,
          end
        )
      else:
        merged_height += max(
          0,
          current_end - current_start
        )
        current_start, current_end = start, end

    merged_height += max(
      0,
      current_end - current_start
    )

    total += (
      right - left
    ) * merged_height

  return total


def box_is_represented(
  candidate,
  final_boxes
):

  for final_box in final_boxes:

    if box_iou(
      candidate,
      final_box
    ) >= 0.34:
      return True

    if containment_ratio(
      candidate,
      final_box
    ) >= 0.82:
      return True

    if containment_ratio(
      final_box,
      candidate
    ) >= 0.82:
      return True

  return False


def reading_order_ambiguity_count(
  boxes,
  image_h
):

  ambiguous = 0
  top_tolerance = max(
    8,
    int(image_h * 0.045)
  )

  for index, first in enumerate(boxes):

    for second in boxes[index + 1:]:

      first_h = max(1, box_height(first))
      second_h = max(1, box_height(second))

      vertical_overlap = intersection_area(
        first,
        second
      ) / max(
        1,
        min(
          box_area(first),
          box_area(second)
        )
      )

      same_top_band = (
        abs(first[1] - second[1])
        <= top_tolerance
      )

      height_ratio = max(
        first_h / second_h,
        second_h / first_h
      )

      if (
        vertical_overlap >= 0.08
        or
        (
          same_top_band
          and
          height_ratio >= 2.15
        )
      ):
        ambiguous += 1

  return ambiguous


def analyze_detection_diagnostics(
  clean,
  legacy,
  base,
  final_boxes,
  removed_groups,
  consensus_rescued,
  local_rescued,
  image_w,
  image_h
):

  page_area = max(
    1,
    image_w * image_h
  )

  coverage = (
    union_box_area(final_boxes)
    / page_area
  )

  raw = (
    list(clean)
    + list(legacy)
    + list(base)
  )

  strong_unrepresented = []

  for item in raw:

    candidate = item["box"]
    candidate_area = (
      box_area(candidate)
      / page_area
    )

    if item.get("conf", 0.0) < DIAGNOSTIC_STRONG_CONFIDENCE:
      continue

    if not (
      0.018
      <= candidate_area
      <= DIAGNOSTIC_MAX_CANDIDATE_PAGE_AREA
    ):
      continue

    if box_is_represented(
      candidate,
      final_boxes
    ):
      continue

    if any(
      box_iou(
        candidate,
        existing["box"]
      ) >= 0.55
      for existing in strong_unrepresented
    ):
      continue

    strong_unrepresented.append(
      {
        "box": list(candidate),
        "conf": round(
          float(item.get("conf", 0.0)),
          4
        ),
        "source": item.get("source", "?"),
      }
    )

  ambiguity_count = reading_order_ambiguity_count(
    final_boxes,
    image_h
  )

  reasons = []
  score = 0

  if not final_boxes and len(raw) >= 2:
    reasons.append("MODEL_ACTIVITY_BUT_NO_FINAL_PANEL")
    score += 2

  if (
    final_boxes
    and
    coverage < DIAGNOSTIC_LOW_COVERAGE
    and
    len(raw) >= len(final_boxes) + 2
  ):
    reasons.append("LOW_FINAL_PAGE_COVERAGE")
    score += 2

  if strong_unrepresented:
    reasons.append("STRONG_RAW_BOX_NOT_REPRESENTED")
    score += 2

  if ambiguity_count:
    reasons.append("READING_ORDER_AMBIGUITY")
    score += min(
      2,
      ambiguity_count
    )

  if removed_groups and not local_rescued:
    reasons.append("REMOVED_GROUP_WITHOUT_LOCAL_RESCUE")
    score += 1

  return {
    "score": score,
    "suspicious": score >= DIAGNOSTIC_SCORE_THRESHOLD,
    "reasons": reasons,
    "page_coverage": round(coverage, 4),
    "reading_order_ambiguities": ambiguity_count,
    "strong_unrepresented": strong_unrepresented,
    "removed_group_count": len(removed_groups),
    "consensus_rescue_count": len(consensus_rescued),
    "local_rescue_count": len(local_rescued),
    "final_boxes": [
      list(box)
      for box in final_boxes
    ],
  }


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance_ratio(
  a,
  b
):

  ax, ay = box_center(a)
  bx, by = box_center(b)

  aw = max(
    1,
    box_width(a)
  )

  ah = max(
    1,
    box_height(a)
  )

  dx = (
    abs(ax - bx)
    / aw
  )

  dy = (
    abs(ay - by)
    / ah
  )

  return max(
    dx,
    dy
  )


def child_center_distance(
  a,
  b,
  parent
):

  ax, ay = box_center(a)
  bx, by = box_center(b)

  pw = max(
    1,
    box_width(parent)
  )

  ph = max(
    1,
    box_height(parent)
  )

  dx = (
    abs(ax - bx)
    / pw
  )

  dy = (
    abs(ay - by)
    / ph
  )

  return (
    (
      dx ** 2
      + dy ** 2
    )
    ** 0.5
  )


# ============================================================
# SAFE BOX EXPANSION
# ============================================================

def expand_box(
  box,
  image_w,
  image_h,
  neighbors=None
):

  x1, y1, x2, y2 = box

  width = (
    x2 - x1
  )

  height = (
    y2 - y1
  )

  expanded_left = clamp(
    x1 - int(width * PAD_LEFT),
    0,
    image_w
  )

  expanded_right = clamp(
    x2 + int(width * PAD_RIGHT),
    0,
    image_w
  )

  expanded_top = clamp(
    y1 - int(height * PAD_TOP),
    0,
    image_h
  )

  expanded_bottom = clamp(
    y2 + int(height * PAD_BOTTOM),
    0,
    image_h
  )

  if neighbors:

    for other in neighbors:

      if other == box:
        continue

      ox1, oy1, ox2, oy2 = other

      vertical_overlap = max(
        0,
        min(y2, oy2) - max(y1, oy1)
      )

      vertical_overlap /= max(
        1,
        min(height, oy2 - oy1)
      )

      horizontal_overlap = max(
        0,
        min(x2, ox2) - max(x1, ox1)
      )

      horizontal_overlap /= max(
        1,
        min(width, ox2 - ox1)
      )

      if vertical_overlap >= NEIGHBOR_AXIS_OVERLAP:

        if ox2 <= x1:
          expanded_left = max(
            expanded_left,
            (ox2 + x1) // 2
          )

        elif ox1 >= x2:
          expanded_right = min(
            expanded_right,
            (x2 + ox1) // 2
          )

      if horizontal_overlap >= NEIGHBOR_AXIS_OVERLAP:

        if oy2 <= y1:
          expanded_top = max(
            expanded_top,
            (oy2 + y1) // 2
          )

        elif oy1 >= y2:
          expanded_bottom = min(
            expanded_bottom,
            (y2 + oy1) // 2
          )

  return (
    expanded_left,
    expanded_top,
    expanded_right,
    expanded_bottom,
  )


# ============================================================
# PAGE GEOMETRY ANALYSIS
# ============================================================

def detect_text_regions(
  gray
):

  page_h, page_w = gray.shape

  binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    11,
  )

  horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (max(20, page_w // 18), 1)
  )

  vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, max(20, page_h // 18))
  )

  long_lines = cv2.bitwise_or(
    cv2.morphologyEx(
      binary,
      cv2.MORPH_OPEN,
      horizontal_kernel
    ),
    cv2.morphologyEx(
      binary,
      cv2.MORPH_OPEN,
      vertical_kernel
    )
  )

  text_mask = cv2.bitwise_and(
    binary,
    cv2.bitwise_not(
      long_lines
    )
  )

  feature_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
      max(9, page_w // 160),
      max(9, page_h // 240)
    )
  )

  light_features = cv2.morphologyEx(
    gray,
    cv2.MORPH_TOPHAT,
    feature_kernel
  )

  _, light_text = cv2.threshold(
    light_features,
    24,
    255,
    cv2.THRESH_BINARY
  )

  light_text = cv2.bitwise_and(
    light_text,
    cv2.bitwise_not(
      long_lines
    )
  )

  join_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (
      max(3, page_w // 280),
      max(2, page_h // 700)
    )
  )

  contours = []

  for candidate_mask, polarity in (
    (text_mask, "dark"),
    (light_text, "light")
  ):

    candidate_mask = cv2.morphologyEx(
      candidate_mask,
      cv2.MORPH_CLOSE,
      join_kernel
    )

    candidate_mask = cv2.dilate(
      candidate_mask,
      join_kernel,
      iterations=1
    )

    found, _ = cv2.findContours(
      candidate_mask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE
    )

    contours.extend(
      (
        contour,
        polarity
      )
      for contour in found
    )

  regions = []

  min_height = max(
    5,
    int(page_h * 0.004)
  )

  max_height = max(
    min_height + 1,
    int(page_h * 0.070)
  )

  min_width = max(
    8,
    int(page_w * 0.008)
  )

  max_width = max(
    min_width + 1,
    int(page_w * 0.38)
  )

  for contour, polarity in contours:

    x, y, width, height = cv2.boundingRect(
      contour
    )

    if not (
      min_width <= width <= max_width
      and
      min_height <= height <= max_height
    ):
      continue

    aspect = width / max(
      1,
      height
    )

    if not 0.35 <= aspect <= 20.0:
      continue

    patch = gray[
      y:y + height,
      x:x + width
    ]

    if patch.size <= 0:
      continue

    dark_ratio = float(
      np.mean(
        patch < 150
      )
    )

    bright_ratio = float(
      np.mean(
        patch > 195
      )
    )

    if polarity == "dark":

      if not (
        0.035 <= dark_ratio <= 0.68
        and
        bright_ratio >= 0.22
      ):
        continue

    else:

      if not (
        0.035 <= bright_ratio <= 0.68
        and
        dark_ratio >= 0.22
      ):
        continue

    regions.append(
      (
        x,
        y,
        x + width,
        y + height
      )
    )

  return regions


def analyze_page_geometry(
  image
):

  rgb = np.asarray(
    image.convert(
      "RGB"
    )
  )

  gray = cv2.cvtColor(
    rgb,
    cv2.COLOR_RGB2GRAY
  )

  vertical_edges = cv2.convertScaleAbs(
    cv2.Sobel(
      gray,
      cv2.CV_32F,
      1,
      0,
      ksize=3
    )
  )

  horizontal_edges = cv2.convertScaleAbs(
    cv2.Sobel(
      gray,
      cv2.CV_32F,
      0,
      1,
      ksize=3
    )
  )

  return {
    "gray": gray,
    "vertical_edges": vertical_edges,
    "horizontal_edges": horizontal_edges,
    "text_regions": detect_text_regions(
      gray
    ),
  }


def best_vertical_boundary(
  analysis,
  original_x,
  y1,
  y2,
  max_shift
):

  gray = analysis["gray"]
  edges = analysis["vertical_edges"]
  page_h, page_w = gray.shape

  span_margin = int(
    max(
      1,
      (y2 - y1) * 0.08
    )
  )

  sy1 = clamp(
    y1 + span_margin,
    0,
    page_h - 1
  )

  sy2 = clamp(
    y2 - span_margin,
    sy1 + 1,
    page_h
  )

  low = clamp(
    original_x - max_shift,
    1,
    page_w - 2
  )

  high = clamp(
    original_x + max_shift,
    low,
    page_w - 2
  )

  best_x = original_x
  best_adjusted = -1.0
  best_raw = 0.0

  for x in range(
    low,
    high + 1
  ):

    edge_band = edges[
      sy1:sy2,
      x - 1:x + 2
    ]

    gray_band = gray[
      sy1:sy2,
      x - 1:x + 2
    ]

    raw_score = (
      0.74
      * float(np.mean(edge_band))
      / 255.0
      +
      0.26
      * float(np.mean(gray_band < 72))
    )

    distance_ratio = (
      abs(x - original_x)
      / max(1, max_shift)
    )

    adjusted_score = (
      raw_score
      - BORDER_SNAP_DISTANCE_PENALTY
      * distance_ratio
    )

    if adjusted_score > best_adjusted:
      best_adjusted = adjusted_score
      best_raw = raw_score
      best_x = x

  if best_raw >= BORDER_SNAP_MIN_SCORE:
    return best_x

  return original_x


def best_horizontal_boundary(
  analysis,
  original_y,
  x1,
  x2,
  max_shift
):

  gray = analysis["gray"]
  edges = analysis["horizontal_edges"]
  page_h, page_w = gray.shape

  span_margin = int(
    max(
      1,
      (x2 - x1) * 0.08
    )
  )

  sx1 = clamp(
    x1 + span_margin,
    0,
    page_w - 1
  )

  sx2 = clamp(
    x2 - span_margin,
    sx1 + 1,
    page_w
  )

  low = clamp(
    original_y - max_shift,
    1,
    page_h - 2
  )

  high = clamp(
    original_y + max_shift,
    low,
    page_h - 2
  )

  best_y = original_y
  best_adjusted = -1.0
  best_raw = 0.0

  for y in range(
    low,
    high + 1
  ):

    edge_band = edges[
      y - 1:y + 2,
      sx1:sx2
    ]

    gray_band = gray[
      y - 1:y + 2,
      sx1:sx2
    ]

    raw_score = (
      0.74
      * float(np.mean(edge_band))
      / 255.0
      +
      0.26
      * float(np.mean(gray_band < 72))
    )

    distance_ratio = (
      abs(y - original_y)
      / max(1, max_shift)
    )

    adjusted_score = (
      raw_score
      - BORDER_SNAP_DISTANCE_PENALTY
      * distance_ratio
    )

    if adjusted_score > best_adjusted:
      best_adjusted = adjusted_score
      best_raw = raw_score
      best_y = y

  if best_raw >= BORDER_SNAP_MIN_SCORE:
    return best_y

  return original_y


def snap_box_to_border(
  box,
  analysis,
  image_w,
  image_h
):

  x1, y1, x2, y2 = box
  width = max(1, x2 - x1)
  height = max(1, y2 - y1)

  horizontal_shift = max(
    2,
    min(
      int(image_w * BORDER_SNAP_PAGE_RANGE),
      int(width * BORDER_SNAP_BOX_RANGE)
    )
  )

  vertical_shift = max(
    2,
    min(
      int(image_h * BORDER_SNAP_PAGE_RANGE),
      int(height * BORDER_SNAP_BOX_RANGE)
    )
  )

  snapped_x1 = best_vertical_boundary(
    analysis,
    x1,
    y1,
    y2,
    horizontal_shift
  )

  snapped_x2 = best_vertical_boundary(
    analysis,
    x2,
    y1,
    y2,
    horizontal_shift
  )

  snapped_y1 = best_horizontal_boundary(
    analysis,
    y1,
    x1,
    x2,
    vertical_shift
  )

  snapped_y2 = best_horizontal_boundary(
    analysis,
    y2,
    x1,
    x2,
    vertical_shift
  )

  minimum_width = max(
    8,
    int(width * 0.72)
  )

  minimum_height = max(
    8,
    int(height * 0.72)
  )

  if snapped_x2 - snapped_x1 < minimum_width:
    snapped_x1, snapped_x2 = x1, x2

  if snapped_y2 - snapped_y1 < minimum_height:
    snapped_y1, snapped_y2 = y1, y2

  return (
    clamp(snapped_x1, 0, image_w),
    clamp(snapped_y1, 0, image_h),
    clamp(snapped_x2, 0, image_w),
    clamp(snapped_y2, 0, image_h),
  )


def expand_box_for_text(
  safe_box,
  panel_box,
  analysis,
  image_w,
  image_h,
  neighbors
):

  sx1, sy1, sx2, sy2 = safe_box
  px1, py1, px2, py2 = panel_box

  panel_w = max(1, px2 - px1)
  panel_h = max(1, py2 - py1)

  max_horizontal = int(
    panel_w
    * TEXT_MAX_HORIZONTAL_EXPANSION
  )

  max_vertical = int(
    panel_h
    * TEXT_MAX_VERTICAL_EXPANSION
  )

  margin = max(
    2,
    int(
      min(image_w, image_h)
      * TEXT_REGION_MARGIN
    )
  )

  left_limit = clamp(
    px1 - max_horizontal,
    0,
    image_w
  )

  right_limit = clamp(
    px2 + max_horizontal,
    0,
    image_w
  )

  top_limit = clamp(
    py1 - max_vertical,
    0,
    image_h
  )

  bottom_limit = clamp(
    py2 + max_vertical,
    0,
    image_h
  )

  for other in neighbors:

    if other == panel_box:
      continue

    ox1, oy1, ox2, oy2 = other

    vertical_overlap = max(
      0,
      min(py2, oy2) - max(py1, oy1)
    ) / max(1, min(panel_h, oy2 - oy1))

    horizontal_overlap = max(
      0,
      min(px2, ox2) - max(px1, ox1)
    ) / max(1, min(panel_w, ox2 - ox1))

    if vertical_overlap >= NEIGHBOR_AXIS_OVERLAP:

      if ox2 <= px1:
        left_limit = max(
          left_limit,
          (ox2 + px1) // 2
        )

      elif ox1 >= px2:
        right_limit = min(
          right_limit,
          (px2 + ox1) // 2
        )

    if horizontal_overlap >= NEIGHBOR_AXIS_OVERLAP:

      if oy2 <= py1:
        top_limit = max(
          top_limit,
          (oy2 + py1) // 2
        )

      elif oy1 >= py2:
        bottom_limit = min(
          bottom_limit,
          (py2 + oy1) // 2
        )

  for tx1, ty1, tx2, ty2 in analysis["text_regions"]:

    vertical_overlap = max(
      0,
      min(py2, ty2) - max(py1, ty1)
    ) / max(1, min(panel_h, ty2 - ty1))

    horizontal_overlap = max(
      0,
      min(px2, tx2) - max(px1, tx1)
    ) / max(1, min(panel_w, tx2 - tx1))

    if vertical_overlap >= 0.18:

      if (
        tx1 < sx1
        and
        tx2 >= left_limit
      ):
        sx1 = max(
          left_limit,
          tx1 - margin
        )

      if (
        tx2 > sx2
        and
        tx1 <= right_limit
      ):
        sx2 = min(
          right_limit,
          tx2 + margin
        )

    if horizontal_overlap >= 0.18:

      if (
        ty1 < sy1
        and
        ty2 >= top_limit
      ):
        sy1 = max(
          top_limit,
          ty1 - margin
        )

      if (
        ty2 > sy2
        and
        ty1 <= bottom_limit
      ):
        sy2 = min(
          bottom_limit,
          ty2 + margin
        )

  return (
    clamp(sx1, 0, image_w),
    clamp(sy1, 0, image_h),
    clamp(sx2, 0, image_w),
    clamp(sy2, 0, image_h),
  )


def diagnose_text_crop_edge(
  panel_box,
  safe_box,
  analysis,
  image_w,
  image_h
):
  """Report text-like regions that may be clipped by a crop boundary."""

  sx1, sy1, sx2, sy2 = safe_box
  px1, py1, px2, py2 = panel_box

  panel_w = max(1, px2 - px1)
  panel_h = max(1, py2 - py1)

  edge_margin_x = max(
    3,
    int(image_w * DIAGNOSTIC_TEXT_EDGE_MARGIN)
  )
  edge_margin_y = max(
    3,
    int(image_h * DIAGNOSTIC_TEXT_EDGE_MARGIN)
  )

  risks = []

  for text_box in analysis["text_regions"]:

    tx1, ty1, tx2, ty2 = text_box

    vertical_overlap = max(
      0,
      min(py2, ty2) - max(py1, ty1)
    ) / max(1, min(panel_h, ty2 - ty1))

    horizontal_overlap = max(
      0,
      min(px2, tx2) - max(px1, tx1)
    ) / max(1, min(panel_w, tx2 - tx1))

    included = containment_ratio(
      text_box,
      safe_box
    )

    crossed_edges = []

    if vertical_overlap >= 0.18:

      if (
        tx1 < sx1 < tx2
      ):
        crossed_edges.append("left")

      if (
        tx1 < sx2 < tx2
      ):
        crossed_edges.append("right")

    if horizontal_overlap >= 0.18:

      if (
        ty1 < sy1 < ty2
      ):
        crossed_edges.append("top")

      if (
        ty1 < sy2 < ty2
      ):
        crossed_edges.append("bottom")

    if (
      crossed_edges
      and
      0.08 < included < 0.94
    ):
      risks.append(
        {
          "severity": "high",
          "text_box": list(text_box),
          "included_ratio": round(included, 4),
          "edges": crossed_edges,
        }
      )

      continue

    if included < 0.94:
      continue

    near_edges = []

    if vertical_overlap >= 0.18:
      if abs(tx1 - sx1) <= edge_margin_x:
        near_edges.append("left")
      if abs(sx2 - tx2) <= edge_margin_x:
        near_edges.append("right")

    if horizontal_overlap >= 0.18:
      if abs(ty1 - sy1) <= edge_margin_y:
        near_edges.append("top")
      if abs(sy2 - ty2) <= edge_margin_y:
        near_edges.append("bottom")

    if near_edges:
      risks.append(
        {
          "severity": "edge",
          "text_box": list(text_box),
          "included_ratio": round(included, 4),
          "edges": near_edges,
        }
      )

  return risks


# ============================================================
# DEBUG IMAGE DRAWING
# ============================================================

def get_debug_font():

  try:

    return ImageFont.truetype(
      "arial.ttf",
      28
    )

  except Exception:

    return ImageFont.load_default()


def draw_debug_detections(
  image,
  detections,
  output_path,
  title="",
  show_order=False
):

  debug_image = ImageOps.exif_transpose(
    image.convert("RGB")
  ).copy()

  draw = ImageDraw.Draw(
    debug_image
  )

  font = get_debug_font()

  # --------------------------------------------------------
  # TITLE
  # --------------------------------------------------------

  if title:

    title_box = (
      0,
      0,
      debug_image.width,
      55
    )

    draw.rectangle(
      title_box,
      fill="black"
    )

    draw.text(
      (12, 10),
      title,
      fill="white",
      font=font
    )

  # --------------------------------------------------------
  # BOXES
  # --------------------------------------------------------

  for index, item in enumerate(
    detections,
    start=1
  ):

    if isinstance(
      item,
      dict
    ):

      box = item["box"]

      confidence = item.get(
        "conf",
        0.0
      )

      source = item.get(
        "source",
        "?"
      )

      if source == "clean":
        outline = "lime"

      elif source == "base":
        outline = "cyan"

      else:
        outline = "yellow"

      label = (
        f"{index} "
        f"{source[0].upper()} "
        f"{confidence:.2f}"
      )

    else:

      box = item
      outline = "yellow"

      if show_order:

        label = (
          f"ORDER {index}"
        )

      else:

        label = str(
          index
        )

    x1, y1, x2, y2 = box

    # ----------------------------------------------------
    # BOX BORDER
    # ----------------------------------------------------

    for offset in range(
      5
    ):

      draw.rectangle(
        (
          x1 - offset,
          y1 - offset,
          x2 + offset,
          y2 + offset
        ),
        outline=outline
      )

    # ----------------------------------------------------
    # LABEL
    # ----------------------------------------------------

    label_y = max(
      58,
      y1
    )

    try:

      bbox = draw.textbbox(
        (
          x1,
          label_y
        ),
        label,
        font=font
      )

      tx1, ty1, tx2, ty2 = bbox

    except Exception:

      tx1 = x1
      ty1 = label_y
      tx2 = x1 + 180
      ty2 = label_y + 40

    draw.rectangle(
      (
        tx1 - 4,
        ty1 - 3,
        tx2 + 4,
        ty2 + 3
      ),
      fill="black"
    )

    draw.text(
      (
        x1,
        label_y
      ),
      label,
      fill=outline,
      font=font
    )

  debug_image.save(
    output_path,
    quality=95
  )


# ============================================================
# YOLO DETECTION
# ============================================================

def get_frame_detections(
  model,
  image,
  confidence,
  source_name,
  image_size=None
):

  if image_size is None:
    image_size = IMAGE_SIZE

  results = model.predict(
    source=image,
    imgsz=image_size,
    conf=confidence,
    iou=MODEL_IOU,
    max_det=40,
    device=DEVICE,
    half=USE_FP16_INFERENCE,
    verbose=False,
  )

  detections = []

  if not results:
    return detections

  result = results[0]

  if result.boxes is None:
    return detections

  for prediction in result.boxes:

    cls_id = int(
      prediction.cls[0].item()
    )

    class_name = str(
      model.names[
        cls_id
      ]
    ).lower()

    if (
      "frame" not in class_name
      and
      "panel" not in class_name
    ):

      continue

    confidence_value = float(
      prediction.conf[0].item()
    )

    x1, y1, x2, y2 = (
      prediction
      .xyxy[0]
      .tolist()
    )

    detections.append(
      {
        "box": (
          int(round(x1)),
          int(round(y1)),
          int(round(x2)),
          int(round(y2)),
        ),

        "conf":
          confidence_value,

        "source":
          source_name,
      }
    )

  return detections


# ============================================================
# GEOMETRY FILTER
# ============================================================

def geometry_filter(
  detections,
  image_w,
  image_h
):

  page_area = (
    image_w
    * image_h
  )

  filtered = []

  for item in detections:

    x1, y1, x2, y2 = (
      item["box"]
    )

    width = (
      x2 - x1
    )

    height = (
      y2 - y1
    )

    if (
      width <= 0
      or
      height <= 0
    ):

      continue

    area_ratio = (
      width
      * height
    ) / page_area

    width_ratio = (
      width
      / image_w
    )

    height_ratio = (
      height
      / image_h
    )

    aspect = max(
      width
      / max(
        height,
        1
      ),

      height
      / max(
        width,
        1
      ),
    )

    if (
      area_ratio
      < MIN_PANEL_AREA_RATIO
    ):

      continue

    if (
      width_ratio
      < MIN_WIDTH_RATIO
    ):

      continue

    if (
      height_ratio
      < MIN_HEIGHT_RATIO
    ):

      continue

    if (
      aspect
      > MAX_ASPECT_RATIO
    ):

      continue

    filtered.append(
      item
    )

  return filtered


# ============================================================
# CONSENSUS SMALL-PANEL RESCUE
# ============================================================

def relaxed_consensus_geometry_candidate(
  item,
  image_w,
  image_h
):

  x1, y1, x2, y2 = item["box"]

  width = x2 - x1
  height = y2 - y1

  if width <= 0 or height <= 0:
    return False

  page_area = max(
    1,
    image_w * image_h
  )

  area_ratio = (
    width * height
  ) / page_area

  width_ratio = (
    width / max(1, image_w)
  )

  height_ratio = (
    height / max(1, image_h)
  )

  aspect = max(
    width / max(1, height),
    height / max(1, width),
  )

  if area_ratio < CONSENSUS_RESCUE_MIN_AREA_RATIO:
    return False

  if area_ratio > CONSENSUS_RESCUE_MAX_PAGE_AREA:
    return False

  if width_ratio < CONSENSUS_RESCUE_MIN_WIDTH_RATIO:
    return False

  if height_ratio < CONSENSUS_RESCUE_MIN_HEIGHT_RATIO:
    return False

  if aspect > CONSENSUS_RESCUE_MAX_ASPECT_RATIO:
    return False

  return True


def consensus_boxes_match(
  a,
  b
):

  iou = box_iou(
    a,
    b
  )

  if iou >= CONSENSUS_RESCUE_SUPPORT_IOU:
    return True

  a_inside_b = containment_ratio(
    a,
    b
  )

  b_inside_a = containment_ratio(
    b,
    a
  )

  if max(
    a_inside_b,
    b_inside_a
  ) < CONSENSUS_RESCUE_SUPPORT_CONTAINMENT:
    return False

  ax, ay = box_center(a)
  bx, by = box_center(b)

  norm_w = max(
    1,
    min(
      box_width(a),
      box_width(b)
    )
  )

  norm_h = max(
    1,
    min(
      box_height(a),
      box_height(b)
    )
  )

  distance = max(
    abs(ax - bx) / norm_w,
    abs(ay - by) / norm_h,
  )

  return (
    distance
    <= CONSENSUS_RESCUE_MAX_CENTER_DISTANCE
  )


def rescue_consensus_small_panels(
  clean,
  legacy,
  base,
  normal_geometry,
  image_w,
  image_h
):

  raw_all = (
    list(clean)
    + list(legacy)
    + list(base)
  )

  # Only look at boxes that failed the normal geometry filter but
  # still satisfy a conservative relaxed geometry envelope.
  normal_ids = {
    id(item)
    for item in normal_geometry
  }

  relaxed = [
    item
    for item in raw_all
    if (
      id(item) not in normal_ids
      and
      relaxed_consensus_geometry_candidate(
        item,
        image_w,
        image_h
      )
    )
  ]

  if not relaxed:
    return [], []

  accepted = []
  diagnostics = []
  consumed = set()

  ordered = sorted(
    relaxed,
    key=lambda item: item.get("conf", 0.0),
    reverse=True,
  )

  for candidate in ordered:

    candidate_id = id(candidate)

    if candidate_id in consumed:
      continue

    supporters = []

    for other in relaxed:

      if (
        other["source"]
        == candidate["source"]
        and
        other is not candidate
      ):
        continue

      if consensus_boxes_match(
        candidate["box"],
        other["box"]
      ):
        supporters.append(
          other
        )

    # Count each detector at most once.
    best_by_source = {}

    for item in supporters:

      source = item["source"]

      if (
        source not in best_by_source
        or
        item.get("conf", 0.0)
        > best_by_source[source].get("conf", 0.0)
      ):
        best_by_source[source] = item

    support_items = list(
      best_by_source.values()
    )

    support_sources = {
      item["source"]
      for item in support_items
    }

    if (
      len(support_sources)
      < CONSENSUS_RESCUE_MIN_MODEL_SUPPORT
    ):
      continue

    confidences = [
      float(
        item.get("conf", 0.0)
      )
      for item in support_items
    ]

    best_conf = max(
      confidences,
      default=0.0
    )

    avg_conf = (
      sum(confidences)
      / max(1, len(confidences))
    )

    if (
      best_conf
      < CONSENSUS_RESCUE_MIN_BEST_CONF
      and
      avg_conf
      < CONSENSUS_RESCUE_MIN_AVG_CONF
    ):
      continue

    # Pick the strongest actual model box instead of inventing a new
    # averaged rectangle. This keeps crop geometry predictable.
    representative = max(
      support_items,
      key=lambda item: item.get("conf", 0.0)
    )

    # Do not rescue something that is already represented by a normal
    # accepted panel.
    if any(
      detections_are_duplicates(
        representative["box"],
        existing["box"]
      )
      or
      box_iou(
        representative["box"],
        existing["box"]
      ) >= 0.50
      for existing in normal_geometry
    ):
      continue

    # Prevent the same consensus cluster from being added twice.
    if any(
      detections_are_duplicates(
        representative["box"],
        existing["box"]
      )
      or
      box_iou(
        representative["box"],
        existing["box"]
      ) >= 0.50
      for existing in accepted
    ):
      continue

    rescued_item = {
      "box": representative["box"],
      "conf": representative["conf"],
      "source": "consensus",
    }

    accepted.append(
      rescued_item
    )

    diagnostics.append(
      {
        "item": rescued_item,
        "sources": sorted(
          support_sources
        ),
        "best_conf": best_conf,
        "avg_conf": avg_conf,
      }
    )

    for item in support_items:
      consumed.add(
        id(item)
      )

  return accepted, diagnostics


# ============================================================
# NEAR-FULL-PAGE FILTER
# ============================================================

def filter_near_full_page_detections(
  detections,
  image_w,
  image_h
):

  kept = []
  removed = []

  for item in detections:

    x1, y1, x2, y2 = (
      item["box"]
    )

    width_ratio = (
      (x2 - x1)
      / max(image_w, 1)
    )

    height_ratio = (
      (y2 - y1)
      / max(image_h, 1)
    )

    if (
      width_ratio
      >= NEAR_FULL_PAGE_MIN_WIDTH_RATIO
      and
      height_ratio
      >= NEAR_FULL_PAGE_MIN_HEIGHT_RATIO
    ):

      removed.append(
        item
      )

    else:

      kept.append(
        item
      )

  if removed:

    print(
      "\n[NEAR-FULL-PAGE FILTER]"
    )

    print(
      f"Removed: {len(removed)}"
    )

    for index, item in enumerate(
      removed,
      1
    ):

      print(
        f" REMOVED {index}: "
        f"box={item['box']} "
        f"conf={item.get('conf', 0):.3f} "
        f"source={item.get('source', 'unknown')}"
      )

  return kept


# ============================================================
# DUPLICATE MERGE
# ============================================================

def should_prefer_tighter_base_detection(
  candidate,
  existing
):

  if (
    candidate.get("source") != "base"
    or
    existing.get("source") != "clean"
  ):
    return False

  candidate_conf = float(
    candidate.get("conf", 0.0)
  )
  existing_conf = float(
    existing.get("conf", 0.0)
  )

  if (
    candidate_conf
    < SMART_MERGE_MIN_BASE_CONF
    or
    candidate_conf - existing_conf
    < SMART_MERGE_MIN_CONF_GAIN
  ):
    return False

  candidate_area = box_area(
    candidate["box"]
  )
  existing_area = box_area(
    existing["box"]
  )

  if existing_area <= 0:
    return False

  area_ratio = (
    candidate_area
    / existing_area
  )

  return (
    SMART_MERGE_MIN_AREA_RATIO
    <= area_ratio
    <= SMART_MERGE_MAX_AREA_RATIO
  )


def should_preserve_tighter_base_for_group(
  candidate,
  existing,
  detections,
  image_w,
  image_h
):

  if not should_prefer_tighter_base_detection(
    candidate,
    existing
  ):
    return False

  comparison_detections = [
    item
    for item in detections
    if item is not candidate
  ]

  children = find_group_children(
    existing,
    comparison_detections
  )

  if len(children) != 2:
    return False

  coverage = approximate_children_coverage(
    children,
    existing["box"]
  )

  if (
    coverage
    >= SMART_MERGE_MAX_GROUP_CHILD_COVERAGE
  ):
    return False

  return is_group_detection(
    existing,
    comparison_detections,
    image_w,
    image_h
  )


def merge_detections(
  detections,
  image_w=None,
  image_h=None,
  preserve_tighter_base_for_groups=False
):

  if not detections:
    return []

  detections = sorted(
    detections,

    key=lambda d: (
      d["source"]
      == "clean",

      d["conf"],
    ),

    reverse=True,
  )

  accepted = []

  for candidate in detections:

    candidate_box = (
      candidate["box"]
    )

    duplicate = False

    for existing_index, existing in enumerate(
      accepted
    ):

      existing_box = (
        existing["box"]
      )

      # ------------------------------------------------
      # STRONG IOU
      # ------------------------------------------------

      if (
        box_iou(
          candidate_box,
          existing_box
        )
        >= 0.60
      ):

        if (
          preserve_tighter_base_for_groups
          and
          image_w is not None
          and
          image_h is not None
          and
          should_preserve_tighter_base_for_group(
            candidate,
            existing,
            detections,
            image_w,
            image_h
          )
        ):

          continue

        duplicate = True
        break

      # ------------------------------------------------
      # NEAR IDENTICAL CONTAINMENT
      # ------------------------------------------------

      candidate_inside = (
        containment_ratio(
          candidate_box,
          existing_box
        )
      )

      existing_inside = (
        containment_ratio(
          existing_box,
          candidate_box
        )
      )

      candidate_area = box_area(
        candidate_box
      )

      existing_area = box_area(
        existing_box
      )

      area_similarity = (
        min(
          candidate_area,
          existing_area
        )
        /
        max(
          1,
          candidate_area,
          existing_area
        )
      )

      if (
        candidate_inside >= 0.94
        or
        existing_inside >= 0.94
      ):

        if (
          area_similarity
          >= MERGE_MIN_AREA_SIMILARITY
          and
          center_distance_ratio(
            candidate_box,
            existing_box
          )
          < 0.20
        ):

          if (
            preserve_tighter_base_for_groups
            and
            image_w is not None
            and
            image_h is not None
            and
            should_preserve_tighter_base_for_group(
              candidate,
              existing,
              detections,
              image_w,
              image_h
            )
          ):

            continue

          duplicate = True
          break

    if not duplicate:

      accepted.append(
        candidate
      )

  return accepted


# ============================================================
# LEGACY RESCUE MERGE
# ============================================================

def detections_are_duplicates(
  candidate_box,
  existing_box
):

  if box_iou(
    candidate_box,
    existing_box
  ) >= 0.60:
    return True

  candidate_inside = containment_ratio(
    candidate_box,
    existing_box
  )

  existing_inside = containment_ratio(
    existing_box,
    candidate_box
  )

  candidate_area = box_area(
    candidate_box
  )

  existing_area = box_area(
    existing_box
  )

  area_similarity = (
    min(
      candidate_area,
      existing_area
    )
    /
    max(
      1,
      candidate_area,
      existing_area
    )
  )

  if (
    candidate_inside >= 0.94
    or
    existing_inside >= 0.94
  ):

    if (
      area_similarity
      >= MERGE_MIN_AREA_SIMILARITY
      and
      center_distance_ratio(
        candidate_box,
        existing_box
      )
      < 0.20
    ):
      return True

  return False


def add_legacy_rescues(
  primary,
  legacy_candidates,
  image_w,
  image_h
):

  if not legacy_candidates:
    return list(primary), []

  comparison_pool = (
    list(primary)
    + list(legacy_candidates)
  )

  accepted = list(primary)
  rescued = []

  ordered = sorted(
    legacy_candidates,
    key=lambda item: item["conf"],
    reverse=True,
  )

  for candidate in ordered:

    # A broad legacy box containing several real panels is not a
    # rescue. The current resolver remains authoritative.
    if is_group_detection(
      candidate,
      comparison_pool,
      image_w,
      image_h
    ):
      continue

    if any(
      detections_are_duplicates(
        candidate["box"],
        existing["box"]
      )
      for existing in accepted
    ):
      continue

    accepted.append(
      candidate
    )

    rescued.append(
      candidate
    )

  return accepted, rescued


# ============================================================
# GROUP CHILD FINDING
# ============================================================

def find_group_children(
  parent,
  detections
):

  parent_box = (
    parent["box"]
  )

  parent_area = box_area(
    parent_box
  )

  if parent_area <= 0:
    return []

  children = []

  for child in detections:

    if child is parent:
      continue

    child_box = (
      child["box"]
    )

    child_area = box_area(
      child_box
    )

    if child_area <= 0:
      continue

    size_ratio = (
      child_area
      / parent_area
    )

    if (
      size_ratio
      > GROUP_MAX_CHILD_SIZE
    ):

      continue

    intersection = intersection_area(
      child_box,
      parent_box
    )

    child_overlap = (
      intersection
      / child_area
    )

    child_center_is_inside = (
      center_inside(
        child_box,
        parent_box
      )
    )

    if (
      child_overlap
      >= GROUP_CHILD_OVERLAP

      or

      child_center_is_inside
    ):

      children.append(
        child
      )

  return children


# ============================================================
# CHILD SPATIAL DIVERSITY
# ============================================================

def children_are_spatially_distinct(
  children,
  parent_box
):

  if len(children) < 2:
    return False

  best_distance = 0.0

  for i in range(
    len(children)
  ):

    for j in range(
      i + 1,
      len(children)
    ):

      distance = (
        child_center_distance(
          children[i]["box"],
          children[j]["box"],
          parent_box
        )
      )

      best_distance = max(
        best_distance,
        distance
      )

  return (
    best_distance
    >= GROUP_MIN_CENTER_DISTANCE
  )


# ============================================================
# CHILD COVERAGE
# ============================================================

def approximate_children_coverage(
  children,
  parent_box
):

  parent_area = box_area(
    parent_box
  )

  if parent_area <= 0:
    return 0.0

  total = 0

  for child in children:

    total += intersection_area(
      child["box"],
      parent_box
    )

  return min(
    1.0,
    total / parent_area
  )


# ============================================================
# GROUP DECISION
# ============================================================

def is_group_detection(
  candidate,
  detections,
  image_w,
  image_h
):

  candidate_box = (
    candidate["box"]
  )

  candidate_area = box_area(
    candidate_box
  )

  if candidate_area <= 0:
    return False

  page_area = (
    image_w
    * image_h
  )

  page_ratio = (
    candidate_area
    / page_area
  )

  children = find_group_children(
    candidate,
    detections
  )

  if (
    len(children)
    < GROUP_MIN_CHILDREN
  ):

    return False

  if not children_are_spatially_distinct(
    children,
    candidate_box
  ):

    return False

  coverage = (
    approximate_children_coverage(
      children,
      candidate_box
    )
  )

  parent_conf = max(
    0.01,
    float(
      candidate["conf"]
    )
  )

  child_conf_average = (
    sum(
      float(
        child["conf"]
      )
      for child in children
    )
    / len(children)
  )

  confidence_ratio = (
    child_conf_average
    / parent_conf
  )

  # --------------------------------------------------------
  # THREE OR MORE CHILDREN
  # --------------------------------------------------------

  if len(children) >= 3:

    if (
      coverage
      >= GROUP_MIN_COVERAGE_THREE

      and

      confidence_ratio
      >= GROUP_MIN_CONF_RATIO
    ):

      return True

  # --------------------------------------------------------
  # TWO CHILDREN
  # --------------------------------------------------------

  if len(children) == 2:

    # A low-coverage parent can be a real panel when one apparent child is
    # merely a thin edge slice (for example, the bookshelf strip on the
    # Witcher room panel). Preserve only this narrow, geometric case; ordinary
    # two-child groups continue to use the stable V5.5 threshold.
    if (
      coverage
      < GROUP_EDGE_SLICE_PRESERVE_MAX_COVERAGE

      and

      any(
        is_edge_slice_fragment(
          child["box"],
          candidate_box
        )
        for child in children
      )
    ):

      return False

    if (
      coverage
      >= GROUP_MIN_COVERAGE_TWO

      and

      confidence_ratio
      >= GROUP_MIN_CONF_RATIO
    ):

      return True

    # Two-child candidates are fully decided here so the broad large/very-
    # large fallbacks cannot override the dedicated safeguard above.
    return False

  # --------------------------------------------------------
  # LARGE GROUP
  # --------------------------------------------------------

  if (
    page_ratio
    >= LARGE_GROUP_PAGE_AREA

    and

    len(children)
    >= 2

    and

    coverage
    >= 0.13

    and

    confidence_ratio
    >= 0.45
  ):

    return True

  # --------------------------------------------------------
  # VERY LARGE GROUP
  # --------------------------------------------------------

  if (
    page_ratio
    >= VERY_LARGE_GROUP_PAGE_AREA

    and

    len(children)
    >= 2

    and

    coverage
    >= 0.10
  ):

    return True

  return False


# ============================================================
# GROUP RESOLVER
# ============================================================

def is_edge_slice_fragment(
  child_box,
  parent_box
):

  parent_w = max(1, box_width(parent_box))
  parent_h = max(1, box_height(parent_box))
  child_w = max(1, box_width(child_box))
  child_h = max(1, box_height(child_box))

  if containment_ratio(
    child_box,
    parent_box
  ) < 0.92:
    return False

  px1, py1, px2, py2 = parent_box
  cx1, cy1, cx2, cy2 = child_box

  near_vertical_edge = (
    abs(cx1 - px1) <= parent_w * 0.08
    or
    abs(cx2 - px2) <= parent_w * 0.08
  )

  near_horizontal_edge = (
    abs(cy1 - py1) <= parent_h * 0.08
    or
    abs(cy2 - py2) <= parent_h * 0.08
  )

  vertical_slice = (
    child_h / parent_h >= 0.92
    and
    child_w / parent_w <= 0.42
    and
    near_vertical_edge
  )

  horizontal_slice = (
    child_w / parent_w >= 0.92
    and
    child_h / parent_h <= 0.42
    and
    near_horizontal_edge
  )

  return (
    vertical_slice
    or
    horizontal_slice
  )


def resolve_group_detections(
  detections,
  image_w,
  image_h,
  return_removed=False
):

  if (
    not detections
    or
    len(detections) < 3
  ):

    if return_removed:
      return detections, []

    return detections

  ordered = sorted(
    detections,

    key=lambda item:
    box_area(
      item["box"]
    ),

    reverse=True,
  )

  kept = []
  removed = []
  preserved_group_slices = set()

  for candidate in ordered:

    if is_group_detection(
      candidate,
      detections,
      image_w,
      image_h
    ):

      children = find_group_children(
        candidate,
        detections
      )

      coverage = (
        approximate_children_coverage(
          children,
          candidate["box"]
        )
      )

      removed.append(
        (
          candidate,
          len(children),
          coverage
        )
      )

    else:

      children = find_group_children(
        candidate,
        detections
      )

      if len(children) == 2:

        coverage = approximate_children_coverage(
          children,
          candidate["box"]
        )

        if (
          coverage
          < GROUP_EDGE_SLICE_PRESERVE_MAX_COVERAGE
        ):

          for child in children:

            if is_edge_slice_fragment(
              child["box"],
              candidate["box"]
            ):
              preserved_group_slices.add(
                id(child)
              )

      kept.append(
        candidate
      )

  if preserved_group_slices:

    kept = [
      item
      for item in kept
      if id(item) not in preserved_group_slices
    ]

  kept = merge_detections(
    kept
  )

  if DEBUG_RESOLVER:

    print(
      "\n[V4 GROUP RESOLVER]"
    )

    print(
      f"Before : {len(detections)}"
    )

    print(
      f"Removed: {len(removed)}"
    )

    print(
      f"After : {len(kept)}"
    )

    for (
      index,
      data
    ) in enumerate(
      removed,
      start=1
    ):

      item, child_count, coverage = data

      print(
        " "
        f"REMOVED {index}: "
        f"box={item['box']} "
        f"conf={item['conf']:.3f} "
        f"source={item['source']} "
        f"children={child_count} "
        f"coverage={coverage:.3f}"
      )

  if return_removed:
    return kept, removed

  return kept


# ============================================================
# LOCAL GROUP RESCUE
# ============================================================

def rescue_panels_from_removed_groups(
  clean_model,
  legacy_model,
  base_model,
  image,
  removed_groups,
  existing,
  image_w,
  image_h
):

  if not removed_groups:
    return list(existing), []

  # Dense pages are already well covered by the normal pipeline.
  # A local high-resolution pass on those pages tended to reinterpret
  # pieces of a large panel as additional panels.
  if len(existing) > LOCAL_RESCUE_MAX_EXISTING_PANELS:
    return list(existing), []

  page_area = max(
    1,
    image_w * image_h
  )

  accepted = list(existing)
  rescued = []
  eligible_groups = []

  for data in removed_groups:

    parent, child_count, coverage = data
    parent_box = parent["box"]
    parent_ratio = (
      box_area(parent_box)
      / page_area
    )

    if child_count < 2:
      continue

    if not (
      LOCAL_RESCUE_MIN_PARENT_PAGE_AREA
      <= parent_ratio
      <= LOCAL_RESCUE_MAX_PARENT_PAGE_AREA
    ):
      continue

    if coverage > LOCAL_RESCUE_MAX_CHILD_COVERAGE:
      continue

    eligible_groups.append(data)

  eligible_groups = sorted(
    eligible_groups,
    key=lambda data: box_area(
      data[0]["box"]
    ),
    reverse=True,
  )[:LOCAL_RESCUE_MAX_GROUPS_PER_PAGE]

  for parent, child_count, coverage in eligible_groups:

    px1, py1, px2, py2 = parent["box"]
    px1 = clamp(px1, 0, image_w)
    py1 = clamp(py1, 0, image_h)
    px2 = clamp(px2, 0, image_w)
    py2 = clamp(py2, 0, image_h)

    if px2 <= px1 or py2 <= py1:
      continue

    parent_box = (
      px1,
      py1,
      px2,
      py2,
    )

    parent_area = max(
      1,
      box_area(parent_box)
    )

    crop = image.crop(
      parent_box
    )

    local = []

    for (
      model,
      confidence,
      source_name
    ) in (
      (
        clean_model,
        CLEAN_CONF,
        "clean_local",
      ),
      (
        legacy_model,
        LEGACY_CONF,
        "legacy_local",
      ),
      (
        base_model,
        BASE_CONF,
        "base_local",
      ),
    ):

      crop_detections = (
        get_frame_detections(
          model,
          crop,
          confidence,
          source_name,
          image_size=(
            LOCAL_RESCUE_IMAGE_SIZE
          )
        )
      )

      for item in crop_detections:

        x1, y1, x2, y2 = (
          item["box"]
        )

        item = dict(item)
        item["box"] = (
          x1 + px1,
          y1 + py1,
          x2 + px1,
          y2 + py1,
        )

        local.append(item)

    local = geometry_filter(
      local,
      image_w,
      image_h
    )

    # Keep the unmerged detections so a rescue must be independently
    # supported by more than one model. This is intentionally stricter
    # than the normal panel pass: a missed panel is safer than adding
    # decorative fragments to an otherwise readable page.
    local_support_pool = list(local)

    local = merge_detections(
      local
    )

    comparison_pool = (
      list(accepted)
      + list(local)
    )

    ordered = sorted(
      local,
      key=lambda item: item["conf"],
      reverse=True,
    )

    accepted_this_group = 0

    for candidate in ordered:

      if (
        accepted_this_group
        >= LOCAL_RESCUE_MAX_ACCEPTED_PER_GROUP
      ):
        break

      candidate_box = (
        candidate["box"]
      )

      x1, y1, x2, y2 = candidate_box
      candidate_w = max(1, x2 - x1)
      candidate_h = max(1, y2 - y1)
      aspect_ratio = max(
        candidate_w / candidate_h,
        candidate_h / candidate_w,
      )

      if aspect_ratio > LOCAL_RESCUE_MAX_ASPECT_RATIO:
        continue

      supporting_models = {
        item.get("source", "unknown").split("_", 1)[0]
        for item in local_support_pool
        if box_iou(
          candidate_box,
          item["box"]
        ) >= LOCAL_RESCUE_SUPPORT_IOU
      }

      if (
        len(supporting_models)
        < LOCAL_RESCUE_MIN_MODEL_SUPPORT
      ):
        continue

      if (
        containment_ratio(
          candidate_box,
          parent_box
        )
        < LOCAL_RESCUE_PARENT_CONTAINMENT
      ):
        continue

      if (
        box_area(candidate_box)
        / parent_area
        > LOCAL_RESCUE_MAX_CANDIDATE_PARENT_AREA
      ):
        continue

      if any(
        containment_ratio(
          candidate_box,
          item["box"]
        )
        >= LOCAL_RESCUE_EXISTING_OVERLAP
        for item in accepted
      ):
        continue

      if any(
        detections_are_duplicates(
          candidate_box,
          item["box"]
        )
        for item in accepted
      ):
        continue

      if is_group_detection(
        candidate,
        comparison_pool,
        image_w,
        image_h
      ):
        continue

      accepted.append(candidate)
      rescued.append(candidate)
      accepted_this_group += 1

  return accepted, rescued


# ============================================================
# READING ORDER
# ============================================================

def sort_panels(
  boxes,
  image_w,
  image_h
):

  if len(boxes) <= 1:
    return boxes

  heights = [
    box_height(box)
    for box in boxes
    if box_height(box) > 0
  ]

  if heights:

    median_height = statistics.median(
      heights
    )

  else:

    median_height = (
      image_h * 0.25
    )

  row_tolerance = max(
    image_h
    * READING_ROW_PAGE_TOLERANCE,

    median_height
    * READING_ROW_PANEL_TOLERANCE
  )

  remaining = sorted(
    boxes,

    key=lambda box: (
      box[1],
      box[0]
    )
  )

  rows = []

  while remaining:

    anchor = remaining[0]

    anchor_top = (
      anchor[1]
    )

    row = []
    later = []

    for box in remaining:

      top_difference = abs(
        box[1]
        - anchor_top
      )

      if (
        top_difference
        <= row_tolerance
      ):

        row.append(
          box
        )

      else:

        later.append(
          box
        )

    row.sort(
      key=lambda box: (
        box[0],
        box[1]
      )
    )

    rows.append(
      row
    )

    remaining = later

  ordered = []

  for row in rows:

    ordered.extend(
      row
    )

  if DEBUG_READING_ORDER:

    print(
      "\n[V4 READING ORDER]"
    )

    print(
      f"Row tolerance: "
      f"{row_tolerance:.1f}px"
    )

    for (
      row_index,
      row
    ) in enumerate(
      rows,
      start=1
    ):

      print(
        f" ROW {row_index}:"
      )

      for box in row:

        print(
          f"  {box}"
        )

    print(
      " FINAL ORDER:"
    )

    for (
      index,
      box
    ) in enumerate(
      ordered,
      start=1
    ):

      print(
        f"  {index}. {box}"
      )

  return ordered


# ============================================================
# KOBO IMAGE
# ============================================================

def make_kobo_image(
  image
):

  image = ImageOps.exif_transpose(
    image.convert(
      "RGB"
    )
  )

  width, height = (
    image.size
  )

  scale = min(
    TARGET_W / width,
    TARGET_H / height,
  )

  new_width = max(
    1,
    int(
      round(
        width * scale
      )
    )
  )

  new_height = max(
    1,
    int(
      round(
        height * scale
      )
    )
  )

  resized = image.resize(
    (
      new_width,
      new_height
    ),

    Image.Resampling.LANCZOS
  )

  canvas = Image.new(
    "RGB",
    (
      KOBO_W,
      KOBO_H
    ),
    "black"
  )

  x = (
    KOBO_W
    - new_width
  ) // 2

  y = (
    KOBO_H
    - new_height
  ) // 2

  canvas.paste(
    resized,
    (
      x,
      y
    )
  )

  return canvas


# ============================================================
# HYBRID DETECTION
# ============================================================

def detect_hybrid(
  clean_model,
  legacy_model,
  base_model,
  image,
  debug_dir=None,
  page_number=1
):

  image_w, image_h = (
    image.size
  )

  # --------------------------------------------------------
  # RAW CLEAN
  # --------------------------------------------------------

  clean = get_frame_detections(
    clean_model,
    image,
    CLEAN_CONF,
    "clean"
  )

  # --------------------------------------------------------
  # RAW LEGACY CLEAN - HIGH-RECALL RESCUE CANDIDATES
  # --------------------------------------------------------

  legacy = get_frame_detections(
    legacy_model,
    image,
    LEGACY_CONF,
    "legacy"
  )

  # --------------------------------------------------------
  # RAW BASE
  # --------------------------------------------------------

  base = get_frame_detections(
    base_model,
    image,
    BASE_CONF,
    "base"
  )

  print(
    f"\nRaw Clean detections: "
    f"{len(clean)}"
  )

  print(
    f"Raw Legacy detections: "
    f"{len(legacy)}"
  )

  print(
    f"Raw Base detections : "
    f"{len(base)}"
  )

  # --------------------------------------------------------
  # DEBUG RAW MODELS
  # --------------------------------------------------------

  if (
    DEBUG_IMAGES
    and
    debug_dir is not None
  ):

    draw_debug_detections(
      image,
      clean,
      debug_dir
      / f"{page_number:03d}_01_raw_clean.jpg",
      "01 - RAW CLEAN MODEL"
    )

    draw_debug_detections(
      image,
      base,
      debug_dir
      / f"{page_number:03d}_02_raw_base.jpg",
      "02 - RAW BASE MODEL"
    )

    draw_debug_detections(
      image,
      legacy,
      debug_dir
      / f"{page_number:03d}_02b_raw_legacy.jpg",
      "02B - RAW LEGACY RESCUE MODEL"
    )

  detections = (
    clean
    + base
  )

  primary_has_near_full_page = any(
    (
      (item["box"][2] - item["box"][0])
      / max(1, image_w)
      >= NEAR_FULL_PAGE_MIN_WIDTH_RATIO
    )
    and
    (
      (item["box"][3] - item["box"][1])
      / max(1, image_h)
      >= NEAR_FULL_PAGE_MIN_HEIGHT_RATIO
    )
    for item in detections
  )

  # --------------------------------------------------------
  # GEOMETRY
  # --------------------------------------------------------

  geometry = geometry_filter(
    detections,
    image_w,
    image_h
  )

  legacy_geometry = geometry_filter(
    legacy,
    image_w,
    image_h
  )

  print(
    f"After geometry   : "
    f"{len(geometry)}"
  )

  print(
    f"Legacy after geometry: "
    f"{len(legacy_geometry)}"
  )

  if (
    DEBUG_IMAGES
    and
    debug_dir is not None
  ):

    draw_debug_detections(
      image,
      geometry,
      debug_dir
      / f"{page_number:03d}_03_after_geometry.jpg",
      "03 - AFTER NORMAL GEOMETRY"
    )

  # --------------------------------------------------------
  # CONSENSUS SMALL-PANEL RESCUE
  # --------------------------------------------------------

  consensus_rescued, consensus_diagnostics = (
    rescue_consensus_small_panels(
      clean,
      legacy,
      base,
      geometry + legacy_geometry,
      image_w,
      image_h
    )
  )

  print(
    f"Consensus rescued  : "
    f"{len(consensus_rescued)}"
  )

  for index, data in enumerate(
    consensus_diagnostics,
    start=1
  ):

    print(
      " "
      f"CONSENSUS {index}: "
      f"box={data['item']['box']} "
      f"sources={','.join(data['sources'])} "
      f"best={data['best_conf']:.3f} "
      f"avg={data['avg_conf']:.3f}"
    )

  geometry_with_consensus = (
    list(geometry)
    + list(consensus_rescued)
  )

  if (
    DEBUG_IMAGES
    and
    debug_dir is not None
  ):

    draw_debug_detections(
      image,
      geometry_with_consensus,
      debug_dir
      / f"{page_number:03d}_03b_after_consensus_rescue.jpg",
      "03B - AFTER CONSENSUS RESCUE"
    )

  # --------------------------------------------------------
  # DUPLICATE MERGE
  # --------------------------------------------------------

  merged = merge_detections(
    geometry_with_consensus,
    image_w=image_w,
    image_h=image_h,
    preserve_tighter_base_for_groups=True
  )

  print(
    f"After duplicate merge: "
    f"{len(merged)}"
  )

  if (
    DEBUG_IMAGES
    and
    debug_dir is not None
  ):

    draw_debug_detections(
      image,
      merged,
      debug_dir
      / f"{page_number:03d}_04_after_merge.jpg",
      "04 - AFTER DUPLICATE MERGE"
    )

  # --------------------------------------------------------
  # GROUP RESOLVER
  # --------------------------------------------------------

  resolved, removed_groups = resolve_group_detections(
    merged,
    image_w,
    image_h,
    return_removed=True
  )

  # --------------------------------------------------------
  # NEAR-FULL-PAGE SUPPRESSION
  # --------------------------------------------------------

  resolved = filter_near_full_page_detections(
    resolved,
    image_w,
    image_h
  )

  legacy_geometry = filter_near_full_page_detections(
    legacy_geometry,
    image_w,
    image_h
  )

  # Covers, advertisements and true splash pages often produce one
  # near-full-page primary candidate. If intentionally suppresses
  # that candidate and leaves no panels, do not let the legacy model
  # fragment the page into decorative crops.
  if (
    not resolved
    and
    primary_has_near_full_page
  ):
    legacy_geometry = []

  resolved, rescued = add_legacy_rescues(
    resolved,
    legacy_geometry,
    image_w,
    image_h
  )

  print(
    f"Legacy panels rescued: "
    f"{len(rescued)}"
  )

  resolved, local_rescued = (
    rescue_panels_from_removed_groups(
      clean_model,
      legacy_model,
      base_model,
      image,
      removed_groups,
      resolved,
      image_w,
      image_h
    )
  )

  print(
    f"Local panels rescued : "
    f"{len(local_rescued)}"
  )

  if (
    DEBUG_IMAGES
    and
    debug_dir is not None
  ):

    draw_debug_detections(
      image,
      resolved,
      debug_dir
      / f"{page_number:03d}_05_after_group_resolver.jpg",
      "05 - PRIMARY + LEGACY + LOCAL RESCUES"
    )

  boxes = [
    item["box"]
    for item in resolved
  ]

  # --------------------------------------------------------
  # READING ORDER
  # --------------------------------------------------------

  boxes = sort_panels(
    boxes,
    image_w,
    image_h
  )

  detection_diagnostics = analyze_detection_diagnostics(
    clean,
    legacy,
    base,
    boxes,
    removed_groups,
    consensus_rescued,
    local_rescued,
    image_w,
    image_h
  )

  if (
    DEBUG_IMAGES
    and
    debug_dir is not None
  ):

    draw_debug_detections(
      image,
      boxes,
      debug_dir
      / f"{page_number:03d}_06_final_reading_order.jpg",
      "06 - FINAL READING ORDER",
      show_order=True
    )

  return (
    boxes,
    len(clean),
    len(legacy),
    len(base),
    detection_diagnostics,
  )


# ============================================================
# PROCESS PAGE
# ============================================================

def process_page(
  clean_model,
  legacy_model,
  base_model,
  image,
  debug_dir=None,
  page_number=1
):

  image = ImageOps.exif_transpose(
    image.convert(
      "RGB"
    )
  )

  image_w, image_h = (
    image.size
  )

  (
    boxes,
    clean_count,
    legacy_count,
    base_count,
    diagnostics,
  ) = detect_hybrid(
    clean_model,
    legacy_model,
    base_model,
    image,
    debug_dir,
    page_number
  )

  outputs = []

  # --------------------------------------------------------
  # NO PANELS
  # --------------------------------------------------------

  if not boxes:

    outputs.append(
      make_kobo_image(
        image
      )
    )

    return (
      outputs,
      {
        "clean": clean_count,
        "legacy": legacy_count,
        "base": base_count,
        "final": 0,
        "mode": "ORIGINAL ONLY",
        "diagnostics": diagnostics,
      }
    )

  # --------------------------------------------------------
  # SPLASH PAGE
  # --------------------------------------------------------

  if len(boxes) == 1:

    ratio = (
      box_area(
        boxes[0]
      )
      /
      (
        image_w
        * image_h
      )
    )

    if ratio >= 0.62:

      outputs.append(
        make_kobo_image(
          image
        )
      )

      return (
        outputs,
        {
          "clean": clean_count,
          "legacy": legacy_count,
          "base": base_count,
          "final": 1,
          "mode": "SPLASH ORIGINAL",
          "diagnostics": diagnostics,
        }
      )

  # --------------------------------------------------------
  # PANELS
  # --------------------------------------------------------

  analysis = analyze_page_geometry(
    image
  )

  refined_boxes = [
    snap_box_to_border(
      box,
      analysis,
      image_w,
      image_h
    )
    for box in boxes
  ]

  prepared_panels = []
  text_crop_risks = []
  final_safe_boxes = []

  for box in refined_boxes:

    safe_box = expand_box(
      box,
      image_w,
      image_h,
      neighbors=refined_boxes
    )

    safe_box = expand_box_for_text(
      safe_box,
      box,
      analysis,
      image_w,
      image_h,
      refined_boxes
    )

    panel_text_risks = diagnose_text_crop_edge(
      box,
      safe_box,
      analysis,
      image_w,
      image_h
    )

    if panel_text_risks:
      text_crop_risks.append(
        {
          "panel_box": list(box),
          "safe_box": list(safe_box),
          "risks": panel_text_risks,
        }
      )

    final_safe_boxes.append(
      safe_box
    )

    crop = image.crop(
      safe_box
    )

    crop = trim_outer_white_space(
      crop
    )

    prepared_panels.append(
      (
        box,
        crop
      )
    )

  # A text region cut by one crop is harmless when it is completely
  # readable in a neighboring panel crop. This mirrors the reader's stated
  # preference and avoids inflating panels merely to repeat the same bubble.
  for panel_data in text_crop_risks:

    own_safe_box = tuple(
      panel_data["safe_box"]
    )

    filtered_risks = []

    for risk in panel_data["risks"]:

      text_box = tuple(
        risk["text_box"]
      )

      covered_elsewhere = (
        risk["severity"] == "high"
        and
        any(
          other_safe_box != own_safe_box
          and
          containment_ratio(
            text_box,
            other_safe_box
          ) >= 0.94
          for other_safe_box in final_safe_boxes
        )
      )

      if not covered_elsewhere:
        filtered_risks.append(
          risk
        )

    panel_data["risks"] = filtered_risks

  text_crop_risks = [
    panel_data
    for panel_data in text_crop_risks
    if panel_data["risks"]
  ]

  # --------------------------------------------------------
  # PANEL OUTPUT
  # --------------------------------------------------------
  # Keep every detected panel intact. Wide/horizontal panels are never
  # split. make_kobo_image preserves aspect ratio and keeps the existing
  # black Kobo canvas/padding behavior.

  for box, crop in prepared_panels:

    outputs.append(
      make_kobo_image(
        crop
      )
    )

  # --------------------------------------------------------
  # ORIGINAL PAGE
  # --------------------------------------------------------

  outputs.append(
    make_kobo_image(
      image
    )
  )

  high_text_risks = sum(
    1
    for panel in text_crop_risks
    for risk in panel["risks"]
    if risk["severity"] == "high"
  )

  edge_text_risks = sum(
    1
    for panel in text_crop_risks
    for risk in panel["risks"]
    if risk["severity"] == "edge"
  )

  if high_text_risks:
    diagnostics["reasons"].append(
      "TEXT_REGION_PARTLY_OUTSIDE_CROP"
    )
    diagnostics["score"] += 2
  elif edge_text_risks:
    diagnostics["reasons"].append(
      "TEXT_REGION_NEAR_CROP_EDGE"
    )
    diagnostics["score"] += 1

  diagnostics["text_crop_risks"] = text_crop_risks
  diagnostics["high_text_risk_count"] = high_text_risks
  diagnostics["edge_text_risk_count"] = edge_text_risks
  critical_reasons = {
    "MODEL_ACTIVITY_BUT_NO_FINAL_PANEL",
    "LOW_FINAL_PAGE_COVERAGE",
  }

  diagnostics["suspicious"] = (
    diagnostics["score"]
    >= DIAGNOSTIC_SCORE_THRESHOLD
    or
    any(
      reason in critical_reasons
      for reason in diagnostics["reasons"]
    )
  )

  return (
    outputs,
    {
      "clean": clean_count,
      "legacy": legacy_count,
      "base": base_count,
      "final": len(boxes),
      "mode": "SMART PANELS THEN ORIGINAL",
      "diagnostics": diagnostics,
    }
  )


# ============================================================
# 7-ZIP
# ============================================================

def find_7zip():

  candidates = [
    shutil.which("7z"),
    shutil.which("7z.exe"),

    Path(
      r"C:\Program Files\7-Zip\7z.exe"
    ),

    Path(
      r"C:\Program Files (x86)\7-Zip\7z.exe"
    ),
  ]

  for candidate in candidates:

    if not candidate:
      continue

    candidate = Path(
      candidate
    )

    if candidate.exists():
      return candidate

  return None


# ============================================================
# WINRAR
# ============================================================

def find_winrar():

  candidates = [
    shutil.which("WinRAR"),
    shutil.which("WinRAR.exe"),

    Path(
      r"C:\Program Files\WinRAR\WinRAR.exe"
    ),

    Path(
      r"C:\Program Files (x86)\WinRAR\WinRAR.exe"
    ),
  ]

  for candidate in candidates:

    if not candidate:
      continue

    candidate = Path(
      candidate
    )

    if candidate.exists():
      return candidate

  return None


# ============================================================
# EXTRACT CBZ
# ============================================================

def extract_cbz(
  archive_path,
  output_dir
):

  try:

    with zipfile.ZipFile(
      archive_path,
      "r"
    ) as archive:

      archive.extractall(
        output_dir
      )

  except zipfile.BadZipFile as error:

    raise RuntimeError(
      "The CBZ archive is corrupted "
      "or is not a valid ZIP archive."
    ) from error


# ============================================================
# EXTRACT CBR
# ============================================================

def extract_cbr(
  archive_path,
  output_dir
):

  tar_path = shutil.which(
    "tar"
  )

  if tar_path:

    result = subprocess.run(
      [
        tar_path,
        "-xf",
        str(archive_path),
        "-C",
        str(output_dir),
      ],
      capture_output=True,
      text=True,
      **WINDOWLESS_SUBPROCESS
    )

    if result.returncode == 0:
      return

  seven_zip = find_7zip()

  if seven_zip:

    result = subprocess.run(
      [
        str(seven_zip),
        "x",
        "-y",
        f"-o{output_dir}",
        str(archive_path),
      ],
      capture_output=True,
      text=True,
      **WINDOWLESS_SUBPROCESS
    )

    if result.returncode == 0:
      return

  winrar = find_winrar()

  if winrar:

    result = subprocess.run(
      [
        str(winrar),
        "x",
        "-ibck",
        "-y",
        str(archive_path),
        str(output_dir) + "\\",
      ],
      capture_output=True,
      text=True,
      **WINDOWLESS_SUBPROCESS
    )

    if result.returncode == 0:
      return

  raise RuntimeError(
    "The CBR archive could not be extracted.\n\n"
    "Please make sure 7-Zip or WinRAR is installed."
  )


# ============================================================
# EXTRACT COMIC
# ============================================================

def extract_comic(
  archive_path,
  output_dir
):

  extension = (
    archive_path
    .suffix
    .lower()
  )

  if extension == ".cbz":

    extract_cbz(
      archive_path,
      output_dir
    )

  elif extension == ".cbr":

    extract_cbr(
      archive_path,
      output_dir
    )

  else:

    raise ValueError(
      "Only .cbz and .cbr files are supported."
    )


# ============================================================
# FIND IMAGES
# ============================================================

def find_comic_images(
  folder
):

  images = []

  for path in folder.rglob(
    "*"
  ):

    if (
      path.is_file()
      and
      path.suffix.lower()
      in IMAGE_EXTENSIONS
    ):

      images.append(
        path
      )

  images.sort(
    key=lambda p:
    natural_sort_key(
      str(
        p.relative_to(
          folder
        )
      )
    )
  )

  return images


# ============================================================
# CONVERT
# ============================================================

class ConversionCancelled(Exception):
  """Raised when the user requests a safe conversion stop."""


def check_conversion_cancelled(cancel_check):

  if cancel_check is not None and cancel_check():
    raise ConversionCancelled()

def _convert_comic_impl(
  input_path,
  output_path=None,
  progress_callback=None,
  cancel_check=None
):

  check_conversion_cancelled(
    cancel_check
  )

  input_path = Path(
    input_path
  ).resolve()

  if not input_path.exists():

    raise FileNotFoundError(
      f"File not found:\n"
      f"{input_path}"
    )

  if (
    input_path.suffix.lower()
    not in {
      ".cbz",
      ".cbr",
    }
  ):

    raise ValueError(
      "Input must be a CBZ or CBR file."
    )

  # --------------------------------------------------------
  # OUTPUT
  # --------------------------------------------------------

  if output_path is None:

    output_path = (
      get_comics_output_directory()
      / (
        input_path.stem
        + "_V58_SMART_MERGE_TEST.cbz"
      )
    )

  else:

    output_path = Path(
      output_path
    ).resolve()

  partial_output_path = output_path.with_name(
    output_path.name + ".partial"
  )

  partial_output_path.unlink(
    missing_ok=True
  )

  # --------------------------------------------------------
  # DEBUG DIRECTORY
  # --------------------------------------------------------

  debug_dir = None

  diagnostics_dir = get_diagnostics_directory(
    input_path
  )

  diagnostic_pages_dir = (
    diagnostics_dir
    / "suspicious_pages"
  )

  diagnostic_pages_dir.mkdir(
    parents=True,
    exist_ok=True
  )

  diagnostic_report = {
    "format": "Comic Kobo Converter V5.8 smart merge diagnostics",
    "input": str(input_path),
    "output": str(output_path),
    "behavior_changed": False,
    "pages": [],
  }

  # --------------------------------------------------------
  # MODELS
  # --------------------------------------------------------

  if progress_callback:
    progress_callback(
      "loading_models",
      0,
      1,
      "Loading AI models"
    )

  check_conversion_cancelled(
    cancel_check
  )

  if not CLEAN_MODEL_PATH.exists():

    raise FileNotFoundError(
      "Trained panel detection model not found:\n"
      + str(
        CLEAN_MODEL_PATH
      )
    )

  if not BASE_MODEL_PATH.exists():

    raise FileNotFoundError(
      "Base panel detection model not found:\n"
      + str(
        BASE_MODEL_PATH
      )
    )

  if not LEGACY_MODEL_PATH.exists():

    raise FileNotFoundError(
      "Legacy rescue panel model not found:\n"
      + str(
        LEGACY_MODEL_PATH
      )
    )

  print(
    "\n"
    "============================================"
  )

  print(
    "COMIC KOBO CONVERTER"
  )

  print(
    "768px Main + Consensus Small-Panel Rescue + 1024px Local Rescue"
  )

  print(
    "V4 Clean + Legacy V3 Rescue + Base Model"
  )

  print(
    "============================================"
    "\n"
  )

  global _CLEAN_MODEL_CACHE
  global _LEGACY_MODEL_CACHE
  global _BASE_MODEL_CACHE

  if _CLEAN_MODEL_CACHE is None:
    _CLEAN_MODEL_CACHE = YOLO(
      str(
        CLEAN_MODEL_PATH
      )
    )

  if _BASE_MODEL_CACHE is None:
    _BASE_MODEL_CACHE = YOLO(
      str(
        BASE_MODEL_PATH
      )
    )

  if _LEGACY_MODEL_CACHE is None:
    _LEGACY_MODEL_CACHE = YOLO(
      str(
        LEGACY_MODEL_PATH
      )
    )

  clean_model = _CLEAN_MODEL_CACHE
  legacy_model = _LEGACY_MODEL_CACHE
  base_model = _BASE_MODEL_CACHE

  check_conversion_cancelled(
    cancel_check
  )

  # --------------------------------------------------------
  # TEMP DIRECTORY
  # --------------------------------------------------------

  with tempfile.TemporaryDirectory(
    prefix="ComicKoboV5_"
  ) as temp_folder:

    temp_folder = Path(
      temp_folder
    )

    if progress_callback:
      progress_callback(
        "extracting",
        0,
        1,
        "Extracting comic pages"
      )

    check_conversion_cancelled(
      cancel_check
    )

    extract_comic(
      input_path,
      temp_folder
    )

    check_conversion_cancelled(
      cancel_check
    )

    image_paths = find_comic_images(
      temp_folder
    )

    if not image_paths:

      raise RuntimeError(
        "No readable comic pages were found."
      )

    total_pages = len(
      image_paths
    )

    print(
      f"Pages found: "
      f"{total_pages}"
    )

    # ----------------------------------------------------
    # OUTPUT CBZ
    # ----------------------------------------------------

    with zipfile.ZipFile(
      partial_output_path,
      "w",
      # JPEG pages are already compressed. Storing them directly avoids a
      # second, lossless ZIP compression pass without changing image pixels.
      compression=zipfile.ZIP_STORED,
    ) as zout:

      output_counter = 1

      for (
        page_number,
        image_path
      ) in enumerate(
        image_paths,
        start=1
      ):

        check_conversion_cancelled(
          cancel_check
        )

        print(
          "\n"
          "--------------------------------------------"
        )

        print(
          f"Processing page "
          f"{page_number}/{total_pages}: "
          f"{image_path.name}"
        )

        with Image.open(
          image_path
        ) as image:

          image.load()

          pages, info = process_page(
            clean_model,
            legacy_model,
            base_model,
            image,
            debug_dir,
            page_number
          )

          page_diagnostics = dict(
            info.get(
              "diagnostics",
              {}
            )
          )

          if page_diagnostics.get(
            "suspicious",
            False
          ):

            reason_text = ", ".join(
              page_diagnostics.get(
                "reasons",
                []
              )
            )

            draw_debug_detections(
              image,
              page_diagnostics.get(
                "final_boxes",
                []
              ),
              diagnostic_pages_dir
              / f"{page_number:04d}_suspect.jpg",
              (
                f"V5.8 SCORE {page_diagnostics.get('score', 0)}: "
                f"{reason_text}"
              ),
              show_order=True
            )

            ImageOps.exif_transpose(
              image.convert("RGB")
            ).save(
              diagnostic_pages_dir
              / f"{page_number:04d}_original.jpg",
              quality=95
            )

        check_conversion_cancelled(
          cancel_check
        )

        print(
          f"\nClean detections: "
          f"{info['clean']}"
        )

        print(
          f"Legacy detections: "
          f"{info['legacy']}"
        )

        print(
          f"Base detections : "
          f"{info['base']}"
        )

        print(
          f"Final panels  : "
          f"{info['final']}"
        )

        print(
          f"Mode      : "
          f"{info['mode']}"
        )

        page_output_start = output_counter

        for output_image in pages:

          check_conversion_cancelled(
            cancel_check
          )

          buffer = io.BytesIO()

          output_image.save(
            buffer,
            format="JPEG",
            quality=95,
            subsampling=0,
            # Huffman optimization only reduces file size; it does not improve
            # decoded image quality. Skipping it makes export considerably
            # faster while preserving the same JPEG quality settings.
            optimize=False,
          )

          filename = (
            f"{output_counter:05d}.jpg"
          )

          zout.writestr(
            filename,
            buffer.getvalue()
          )

          output_counter += 1

        diagnostic_report["pages"].append(
          {
            "page_number": page_number,
            "source_file": image_path.name,
            "output_first": page_output_start,
            "output_last": output_counter - 1,
            "mode": info["mode"],
            "clean_detections": info["clean"],
            "legacy_detections": info["legacy"],
            "base_detections": info["base"],
            "final_panels": info["final"],
            "diagnostics": page_diagnostics,
          }
        )

        if progress_callback:
          progress_callback(
            "processing",
            page_number,
            total_pages,
            (
              f"Page {page_number} "
              f"of {total_pages}"
            )
          )

        check_conversion_cancelled(
          cancel_check
        )

  check_conversion_cancelled(
    cancel_check
  )

  partial_output_path.replace(
    output_path
  )

  diagnostic_report["summary"] = {
    "total_pages": len(
      diagnostic_report["pages"]
    ),
    "suspicious_pages": sum(
      1
      for page in diagnostic_report["pages"]
      if page["diagnostics"].get(
        "suspicious",
        False
      )
    ),
  }

  diagnostic_report_path = (
    diagnostics_dir
    / "diagnostic_report.json"
  )

  with diagnostic_report_path.open(
    "w",
    encoding="utf-8"
  ) as report_file:
    json.dump(
      diagnostic_report,
      report_file,
      ensure_ascii=False,
      indent=2
    )

  diagnostic_summary_path = (
    diagnostics_dir
    / "SUPHELI_SAYFALAR.txt"
  )

  suspicious_pages = [
    page
    for page in diagnostic_report["pages"]
    if page["diagnostics"].get(
      "suspicious",
      False
    )
  ]

  summary_lines = [
    "COMIC KOBO CONVERTER V5.8 SMART MERGE - SUPHELI SAYFALAR",
    "================================================",
    "",
    f"Kaynak: {input_path}",
    f"Toplam sayfa: {len(diagnostic_report['pages'])}",
    f"Supheli sayfa: {len(suspicious_pages)}",
    "",
  ]

  if suspicious_pages:

    for page in suspicious_pages:

      reasons = ", ".join(
        page["diagnostics"].get(
          "reasons",
          []
        )
      )

      summary_lines.extend(
        [
          (
            f"Sayfa {page['page_number']} "
            f"({page['source_file']})"
          ),
          (
            f"  Cikti araligi: "
            f"{page['output_first']:05d}.jpg - "
            f"{page['output_last']:05d}.jpg"
          ),
          (
            f"  Risk puani: "
            f"{page['diagnostics'].get('score', 0)}"
          ),
          f"  Nedenler: {reasons}",
          "",
        ]
      )

  else:
    summary_lines.append(
      "Otomatik sistem supheli sayfa bulmadi."
    )

  diagnostic_summary_path.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
  )

  print(
    "\n"
    "============================================"
  )

  print(
    "COMIC KOBO CONVERTER COMPLETE"
  )

  print(
    f"Output CBZ:"
  )

  print(
    output_path
  )

  print(
    "Diagnostic report:"
  )

  print(
    diagnostic_report_path
  )

  print(
    "Suspicious-page summary:"
  )

  print(
    diagnostic_summary_path
  )

  print(
    "============================================"
    "\n"
  )

  if progress_callback:
    progress_callback(
      "finished",
      1,
      1,
      "Conversion complete"
    )

  return output_path


def convert_comic(
  input_path,
  output_path=None,
  progress_callback=None,
  cancel_check=None
):

  resolved_input = Path(
    input_path
  ).resolve()

  if output_path is None:
    resolved_output = (
      get_comics_output_directory()
      / (
        resolved_input.stem
        + "_V58_SMART_MERGE_TEST.cbz"
      )
    )
  else:
    resolved_output = Path(
      output_path
    ).resolve()

  partial_output_path = resolved_output.with_name(
    resolved_output.name + ".partial"
  )

  try:
    return _convert_comic_impl(
      input_path,
      output_path,
      progress_callback=progress_callback,
      cancel_check=cancel_check
    )
  except ConversionCancelled:
    try:
      partial_output_path.unlink(
        missing_ok=True
      )
    except OSError:
      pass
    raise


# ============================================================
# COMMAND LINE
# ============================================================

def main():

  parser = argparse.ArgumentParser(
    description=(
      "Comic Kobo Converter"
    )
  )

  parser.add_argument(
    "input",
    help="Input CBZ or CBR file"
  )

  parser.add_argument(
    "-o",
    "--output",
    default=None,
    help="Output CBZ file"
  )

  args = parser.parse_args()

  convert_comic(
    args.input,
    args.output
  )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

  main()
