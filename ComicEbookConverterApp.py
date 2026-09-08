# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image
from tkinterdnd2 import DND_FILES, TkinterDnD


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSET_DIR / "ComicEbookConverter_Logo.png"
ICON_PATH = ASSET_DIR / "ComicEbookConverter.ico"


def resource_path(relative: str) -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / relative
    return BASE_DIR / relative


class CTkDnD(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class ComicEbookConverterApp(CTkDnD):
    BG = "#090C12"
    SURFACE = "#0F141D"
    SURFACE_ALT = "#131A24"
    SURFACE_HOVER = "#18212E"
    BORDER = "#2A3443"
    BORDER_SOFT = "#202A37"
    TEXT = "#F5F0E4"
    TEXT_MUTED = "#8D99AA"
    ACCENT = "#702963"
    ACCENT_HOVER = "#8E3A7C"
    ACCENT_TEXT = "#FFF8FC"
    RED = "#E75A52"
    RED_HOVER = "#B9433E"
    GREEN = "#45C486"
    BLUE_GRAY = "#607087"

    def __init__(self):
        super().__init__()

        self.title("Comic Ebook Converter")
        self.geometry("1180x800")
        self.minsize(1040, 720)
        self.configure(fg_color=self.BG)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.selected_files: list[Path] = []
        self.output_files: list[Path] = []
        self.queue_states: list[str] = []
        self.queue_rows: list[ctk.CTkFrame] = []
        self.active_file_index = 0
        self.is_converting = False
        self.cancel_event = threading.Event()
        self.pipeline = None

        self.logo_image = None
        self._load_brand_assets()
        self._build_layout()
        self._register_drop_targets()

        if os.environ.get("CEC_ENGINE_SMOKE_TEST") == "1":
            self.after(150, self._run_engine_smoke_test)
        elif os.environ.get("CEC_UI_SMOKE_TEST") == "1":
            self.after(900, self.destroy)

    def _load_brand_assets(self) -> None:
        logo_path = resource_path("assets/ComicEbookConverter_Logo.png")
        icon_path = resource_path("assets/ComicEbookConverter.ico")

        if logo_path.exists():
            logo = Image.open(logo_path).convert("RGBA")
            self.logo_image = ctk.CTkImage(
                light_image=logo,
                dark_image=logo,
                size=(76, 76),
            )

        if icon_path.exists():
            self.after(120, lambda: self._set_window_icon(icon_path))

    def _set_window_icon(self, icon_path: Path) -> None:
        try:
            self.iconbitmap(str(icon_path))
        except Exception:
            pass

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_main_area()
        self._build_progress_area()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=34, pady=(24, 18))
        header.grid_columnconfigure(1, weight=1)

        if self.logo_image:
            logo_label = ctk.CTkLabel(header, text="", image=self.logo_image)
        else:
            logo_label = ctk.CTkLabel(
                header,
                text="CE",
                width=76,
                height=76,
                corner_radius=18,
                fg_color=self.SURFACE_ALT,
                text_color=self.ACCENT,
                font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            )
        logo_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 18))

        title = ctk.CTkLabel(
            header,
            text="COMIC EBOOK CONVERTER",
            anchor="w",
            text_color=self.TEXT,
            font=ctk.CTkFont(family="Bahnschrift Condensed", size=31, weight="bold"),
        )
        title.grid(row=0, column=1, sticky="sw")

        subtitle = ctk.CTkLabel(
            header,
            text="Prepare comics for a better e-reader experience",
            anchor="w",
            text_color=self.TEXT_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        subtitle.grid(row=1, column=1, sticky="nw", pady=(2, 0))

    def _build_main_area(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=34)
        main.grid_columnconfigure(0, weight=5, uniform="main")
        main.grid_columnconfigure(1, weight=6, uniform="main")
        main.grid_rowconfigure(0, weight=1)

        self._build_drop_card(main)
        self._build_queue_card(main)

    def _build_drop_card(self, parent: ctk.CTkFrame) -> None:
        self.drop_card = ctk.CTkFrame(
            parent,
            fg_color=self.SURFACE,
            border_width=1,
            border_color=self.BORDER,
            corner_radius=18,
        )
        self.drop_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.drop_card.grid_columnconfigure(0, weight=1)
        self.drop_card.grid_rowconfigure(0, weight=1)

        inner = ctk.CTkFrame(
            self.drop_card,
            fg_color="transparent",
            border_width=2,
            border_color=self.ACCENT,
            corner_radius=16,
        )
        inner.grid(row=0, column=0, sticky="nsew", padx=46, pady=42)
        inner.grid_columnconfigure(0, weight=1)
        inner.grid_rowconfigure(0, weight=1)
        self.drop_inner = inner

        content = ctk.CTkFrame(inner, fg_color="transparent")
        content.grid(row=0, column=0)

        glyph = ctk.CTkLabel(
            content,
            text="＋",
            width=82,
            height=82,
            corner_radius=22,
            fg_color=self.SURFACE_ALT,
            text_color=self.ACCENT,
            font=ctk.CTkFont(family="Segoe UI", size=42, weight="bold"),
        )
        glyph.pack(pady=(0, 22))

        self.drop_title = ctk.CTkLabel(
            content,
            text="Drop your comics here",
            text_color=self.TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=23, weight="bold"),
        )
        self.drop_title.pack()

        self.drop_hint = ctk.CTkLabel(
            content,
            text="CBZ and CBR files",
            text_color=self.TEXT_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=14),
        )
        self.drop_hint.pack(pady=(8, 22))

        self.browse_button = ctk.CTkButton(
            content,
            text="Browse Files",
            command=self.select_files,
            width=180,
            height=46,
            corner_radius=11,
            fg_color=self.ACCENT,
            hover_color=self.ACCENT_HOVER,
            text_color=self.ACCENT_TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
        )
        self.browse_button.pack()

        note = ctk.CTkLabel(
            self.drop_card,
            text="Add several comics at once. They will be converted in queue order.",
            text_color="#687588",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            wraplength=420,
        )
        note.grid(row=1, column=0, pady=(0, 18))

    def _build_queue_card(self, parent: ctk.CTkFrame) -> None:
        queue_card = ctk.CTkFrame(
            parent,
            fg_color=self.SURFACE,
            border_width=1,
            border_color=self.BORDER,
            corner_radius=18,
        )
        queue_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        queue_card.grid_columnconfigure(0, weight=1)
        queue_card.grid_rowconfigure(1, weight=1)

        queue_header = ctk.CTkFrame(queue_card, fg_color="transparent")
        queue_header.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 12))
        queue_header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            queue_header,
            text="Conversion Queue",
            anchor="w",
            text_color=self.TEXT,
            font=ctk.CTkFont(family="Bahnschrift", size=20, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        self.queue_count_label = ctk.CTkLabel(
            queue_header,
            text="No comics selected",
            text_color=self.TEXT_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.queue_count_label.grid(row=0, column=1, sticky="e")

        self.queue_scroll = ctk.CTkScrollableFrame(
            queue_card,
            fg_color="transparent",
            scrollbar_button_color=self.BORDER,
            scrollbar_button_hover_color=self.BLUE_GRAY,
        )
        self.queue_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.queue_scroll.grid_columnconfigure(0, weight=1)

        self.empty_queue_label = ctk.CTkLabel(
            self.queue_scroll,
            text="Your conversion queue will appear here",
            text_color="#667387",
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        self.empty_queue_label.grid(row=0, column=0, pady=90)

        footer = ctk.CTkFrame(queue_card, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=20, pady=(4, 16))
        footer.grid_columnconfigure(0, weight=1)

        self.queue_size_label = ctk.CTkLabel(
            footer,
            text="0 MB",
            text_color=self.TEXT_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.queue_size_label.grid(row=0, column=0, sticky="w")

        self.clear_button = ctk.CTkButton(
            footer,
            text="Clear Queue",
            command=self.clear_queue,
            width=116,
            height=34,
            corner_radius=9,
            fg_color="transparent",
            hover_color=self.SURFACE_HOVER,
            border_width=1,
            border_color=self.BORDER,
            text_color=self.TEXT_MUTED,
            state="disabled",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        )
        self.clear_button.grid(row=0, column=1, sticky="e")

    def _build_progress_area(self) -> None:
        panel = ctk.CTkFrame(
            self,
            fg_color=self.SURFACE,
            border_width=1,
            border_color=self.BORDER,
            corner_radius=18,
        )
        panel.grid(row=2, column=0, sticky="ew", padx=34, pady=(18, 24))
        panel.grid_columnconfigure(0, weight=1)

        progress_content = ctk.CTkFrame(panel, fg_color="transparent")
        progress_content.grid(row=0, column=0, sticky="ew", padx=22, pady=18)
        progress_content.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            progress_content,
            text="Select comics to begin",
            anchor="w",
            text_color=self.TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.progress_count_label = ctk.CTkLabel(
            progress_content,
            text="0 / 0  •  0%",
            text_color=self.TEXT_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        )
        self.progress_count_label.grid(row=0, column=1, sticky="e")

        self.progress_bar = ctk.CTkProgressBar(
            progress_content,
            height=12,
            corner_radius=8,
            fg_color="#202733",
            progress_color=self.ACCENT,
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 5))
        self.progress_bar.set(0)

        self.detail_label = ctk.CTkLabel(
            progress_content,
            text="Files are saved in the Comics folder next to the application.",
            anchor="w",
            text_color="#687588",
            font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        self.detail_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        actions = ctk.CTkFrame(panel, fg_color="transparent")
        actions.grid(row=0, column=1, padx=(8, 22), pady=18)

        self.start_button = ctk.CTkButton(
            actions,
            text="Start Conversion",
            command=self.start_conversion,
            width=184,
            height=46,
            corner_radius=10,
            state="disabled",
            fg_color=self.ACCENT,
            hover_color=self.ACCENT_HOVER,
            text_color=self.ACCENT_TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
        )
        self.start_button.grid(row=0, column=0, padx=(0, 10))

        self.cancel_button = ctk.CTkButton(
            actions,
            text="Cancel",
            command=self.cancel_conversion,
            width=108,
            height=46,
            corner_radius=10,
            state="disabled",
            fg_color="transparent",
            hover_color="#321A1B",
            border_width=1,
            border_color=self.RED,
            text_color=self.RED,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        )
        self.cancel_button.grid(row=0, column=1, padx=(0, 10))

        self.output_button = ctk.CTkButton(
            actions,
            text="Open Output Folder",
            command=self.open_output_folder,
            width=160,
            height=46,
            corner_radius=10,
            state="disabled",
            fg_color=self.SURFACE_ALT,
            hover_color=self.SURFACE_HOVER,
            border_width=1,
            border_color=self.BORDER,
            text_color=self.TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        )
        self.output_button.grid(row=0, column=2)

    def _register_drop_targets(self) -> None:
        for widget in (self, self.drop_card, self.drop_inner):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
            widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop_enter(self, _event):
        if not self.is_converting:
            self.drop_inner.configure(border_color=self.GREEN)
            self.drop_hint.configure(text="Release to add files", text_color=self.GREEN)
        return "copy"

    def _on_drop_leave(self, _event):
        self._reset_drop_style()
        return "copy"

    def _on_drop(self, event):
        self._reset_drop_style()
        if self.is_converting:
            return "copy"
        raw_paths = self.tk.splitlist(event.data)
        self.add_files(Path(path) for path in raw_paths)
        return "copy"

    def _reset_drop_style(self) -> None:
        self.drop_inner.configure(border_color=self.ACCENT)
        self.drop_hint.configure(text="CBZ and CBR files", text_color=self.TEXT_MUTED)

    def get_app_directory(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return BASE_DIR

    def get_output_directory(self) -> Path:
        output_dir = self.get_app_directory() / "Comics"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def select_files(self) -> None:
        if self.is_converting:
            return
        paths = filedialog.askopenfilenames(
            title="Select one or more comics",
            filetypes=[
                ("Comic Book Archives", "*.cbz *.cbr"),
                ("CBZ Comics", "*.cbz"),
                ("CBR Comics", "*.cbr"),
            ],
        )
        if paths:
            self.add_files(Path(path) for path in paths)

    def add_files(self, paths) -> None:
        if self.is_converting:
            return

        valid: list[Path] = []
        invalid_found = False
        existing = {str(path.resolve()).lower() for path in self.selected_files}

        for candidate in paths:
            candidate = Path(candidate)
            if candidate.is_dir():
                children = sorted(
                    path
                    for path in candidate.iterdir()
                    if path.is_file() and path.suffix.lower() in {".cbz", ".cbr"}
                )
                for child in children:
                    key = str(child.resolve()).lower()
                    if key not in existing:
                        valid.append(child)
                        existing.add(key)
                continue

            if not candidate.is_file() or candidate.suffix.lower() not in {".cbz", ".cbr"}:
                invalid_found = True
                continue

            key = str(candidate.resolve()).lower()
            if key not in existing:
                valid.append(candidate)
                existing.add(key)

        if not valid:
            if invalid_found:
                self._show_message("Only CBZ and CBR files can be added.", self.RED)
            return

        self.selected_files.extend(valid)
        self.queue_states.extend("Waiting" for _ in valid)
        self._sync_output_files()
        self._refresh_queue()
        self._set_selection_state()

    def _sync_output_files(self) -> None:
        output_dir = self.get_output_directory()
        self.output_files = [output_dir / f"{path.stem}.cbz" for path in self.selected_files]

    def _refresh_queue(self) -> None:
        for row in self.queue_rows:
            row.destroy()
        self.queue_rows.clear()

        if hasattr(self, "empty_queue_label"):
            self.empty_queue_label.grid_remove()

        if not self.selected_files:
            self.empty_queue_label.grid()
            self.queue_count_label.configure(text="No comics selected")
            self.queue_size_label.configure(text="0 MB")
            self.clear_button.configure(state="disabled")
            return

        for index, (path, state) in enumerate(zip(self.selected_files, self.queue_states)):
            row = self._create_queue_row(index, path, state)
            row.grid(row=index, column=0, sticky="ew", pady=(0, 8))
            self.queue_rows.append(row)

        count = len(self.selected_files)
        self.queue_count_label.configure(text=f"{count} comic{'s' if count != 1 else ''}")
        total_size = sum(path.stat().st_size for path in self.selected_files if path.exists())
        if total_size >= 1024**3:
            size_text = f"{total_size / (1024**3):.2f} GB selected"
        else:
            size_text = f"{total_size / (1024**2):.1f} MB selected"
        self.queue_size_label.configure(text=size_text)
        self.clear_button.configure(state="disabled" if self.is_converting else "normal")

    def _create_queue_row(self, index: int, path: Path, state: str) -> ctk.CTkFrame:
        row = ctk.CTkFrame(
            self.queue_scroll,
            fg_color=self.SURFACE_ALT,
            border_width=1,
            border_color=self.BORDER_SOFT,
            corner_radius=12,
            height=72,
        )
        row.grid_columnconfigure(2, weight=1)
        row.grid_propagate(False)

        reorder = ctk.CTkFrame(row, fg_color="transparent")
        reorder.grid(row=0, column=0, padx=(8, 4), pady=8)

        button_state = "disabled" if self.is_converting else "normal"
        up = ctk.CTkButton(
            reorder,
            text="↑",
            width=25,
            height=24,
            corner_radius=7,
            state=button_state if index > 0 else "disabled",
            command=lambda i=index: self.move_item(i, -1),
            fg_color="transparent",
            hover_color=self.SURFACE_HOVER,
            text_color=self.TEXT_MUTED,
        )
        up.grid(row=0, column=0)
        down = ctk.CTkButton(
            reorder,
            text="↓",
            width=25,
            height=24,
            corner_radius=7,
            state=button_state if index < len(self.selected_files) - 1 else "disabled",
            command=lambda i=index: self.move_item(i, 1),
            fg_color="transparent",
            hover_color=self.SURFACE_HOVER,
            text_color=self.TEXT_MUTED,
        )
        down.grid(row=1, column=0)

        extension = path.suffix.replace(".", "").upper()
        badge = ctk.CTkLabel(
            row,
            text=extension,
            width=46,
            height=46,
            corner_radius=10,
            fg_color="#202836",
            text_color=self.ACCENT,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
        )
        badge.grid(row=0, column=1, padx=(4, 12), pady=12)

        display_name = path.name if len(path.name) <= 43 else path.name[:40] + "..."
        name = ctk.CTkLabel(
            row,
            text=display_name,
            anchor="w",
            text_color=self.TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        )
        name.grid(row=0, column=2, sticky="w")

        colors = {
            "Waiting": ("#202836", self.TEXT_MUTED),
            "Queued": ("#202836", self.TEXT_MUTED),
            "Converting": ("#27101F", "#C87BB8"),
            "Complete": ("#153327", self.GREEN),
            "Cancelled": ("#27101F", "#C87BB8"),
            "Failed": ("#3A1C1E", self.RED),
        }
        chip_bg, chip_text = colors.get(state, ("#202836", self.TEXT_MUTED))
        chip = ctk.CTkLabel(
            row,
            text=state,
            width=82,
            height=28,
            corner_radius=9,
            fg_color=chip_bg,
            text_color=chip_text,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
        )
        chip.grid(row=0, column=3, padx=10)

        remove = ctk.CTkButton(
            row,
            text="×",
            width=32,
            height=32,
            corner_radius=8,
            state=button_state,
            command=lambda i=index: self.remove_item(i),
            fg_color="transparent",
            hover_color="#321A1B",
            text_color=self.RED,
            font=ctk.CTkFont(family="Segoe UI", size=20),
        )
        remove.grid(row=0, column=4, padx=(0, 10))
        return row

    def move_item(self, index: int, offset: int) -> None:
        if self.is_converting:
            return
        destination = index + offset
        if not 0 <= destination < len(self.selected_files):
            return
        self.selected_files[index], self.selected_files[destination] = (
            self.selected_files[destination],
            self.selected_files[index],
        )
        self.queue_states[index], self.queue_states[destination] = (
            self.queue_states[destination],
            self.queue_states[index],
        )
        self._sync_output_files()
        self._refresh_queue()

    def remove_item(self, index: int) -> None:
        if self.is_converting:
            return
        if 0 <= index < len(self.selected_files):
            self.selected_files.pop(index)
            self.queue_states.pop(index)
            self._sync_output_files()
            self._refresh_queue()
            self._set_selection_state()

    def clear_queue(self) -> None:
        if self.is_converting:
            return
        self.selected_files.clear()
        self.output_files.clear()
        self.queue_states.clear()
        self._refresh_queue()
        self._set_selection_state()

    def _set_selection_state(self) -> None:
        has_files = bool(self.selected_files)
        self.start_button.configure(state="normal" if has_files else "disabled")
        self.output_button.configure(
            state="normal" if any(path.exists() for path in self.output_files) else "disabled"
        )
        self.progress_bar.set(0)
        self.progress_count_label.configure(text="0 / 0  •  0%")
        if has_files:
            count = len(self.selected_files)
            self.status_label.configure(
                text=f"{count} comic{'s' if count != 1 else ''} selected",
                text_color=self.TEXT,
            )
            self.detail_label.configure(text="The queue will run from top to bottom.")
        else:
            self.status_label.configure(text="Select comics to begin", text_color=self.TEXT)
            self.detail_label.configure(
                text="Files are saved in the Comics folder next to the application."
            )

    def start_conversion(self) -> None:
        if not self.selected_files or self.is_converting:
            return

        self.cancel_event.clear()
        self.is_converting = True
        self.active_file_index = 0
        self.queue_states = ["Waiting" for _ in self.selected_files]
        self.queue_states[0] = "Converting"
        self._refresh_queue()

        self.start_button.configure(state="disabled", text="Converting...")
        self.cancel_button.configure(state="normal", text="Cancel")
        self.output_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.clear_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_count_label.configure(text="0 / 0  •  0%")
        self.status_label.configure(text=f"Preparing {self.selected_files[0].name}")
        self.detail_label.configure(text="Loading the panel detection models...")

        threading.Thread(target=self._convert_queue, daemon=True).start()

    def cancel_conversion(self) -> None:
        if not self.is_converting or self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled", text="Stopping...")
        self.status_label.configure(text="Stopping safely...", text_color="#C87BB8")
        self.detail_label.configure(
            text="The current AI step will finish before the conversion stops."
        )

    def _load_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        if getattr(sys, "frozen", False):
            candidates = [resource_path("engine/run_v6_clean_pipeline.py")]
        else:
            candidates = [
                BASE_DIR.parent / "run_v6_clean_pipeline.py",
                BASE_DIR / "engine" / "run_v6_clean_pipeline.py",
            ]

        runner_path = next((path for path in candidates if path.exists()), None)
        if runner_path is None:
            raise FileNotFoundError("The conversion engine could not be found.")

        spec = importlib.util.spec_from_file_location("cec_v6_runner", runner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.pipeline = module.load_pipeline()
        return self.pipeline

    def _run_engine_smoke_test(self) -> None:
        import json
        import traceback

        report_path = Path(os.environ["CEC_SMOKE_REPORT"])
        source_path = Path(os.environ["CEC_SMOKE_SOURCE"])
        output_path = Path(os.environ["CEC_SMOKE_OUTPUT"])
        report = {
            "source": str(source_path),
            "output": str(output_path),
            "success": False,
        }

        try:
            pipeline = self._load_pipeline()
            report.update(
                {
                    "cuda_available": bool(pipeline.torch.cuda.is_available()),
                    "device": str(pipeline.DEVICE),
                    "clean_model": str(pipeline.CLEAN_MODEL_PATH),
                    "legacy_model": str(pipeline.LEGACY_MODEL_PATH),
                    "base_model": str(pipeline.BASE_MODEL_PATH),
                }
            )
            pipeline.convert_comic(
                source_path,
                output_path,
                progress_callback=lambda *_args: None,
                cancel_check=lambda: False,
            )
            report["success"] = output_path.exists() and output_path.stat().st_size > 0
            report["output_bytes"] = (
                output_path.stat().st_size if output_path.exists() else 0
            )
        except Exception as error:
            report["error"] = str(error)
            report["traceback"] = traceback.format_exc()
        finally:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.destroy()

    def _convert_queue(self) -> None:
        pipeline = None
        try:
            pipeline = self._load_pipeline()
            for file_index, (source, output) in enumerate(
                zip(self.selected_files, self.output_files)
            ):
                if self.cancel_event.is_set():
                    raise pipeline.ConversionCancelled()

                self.active_file_index = file_index
                self.after(0, lambda i=file_index: self._mark_active_file(i))

                def callback(stage, current, total, message, index=file_index):
                    self.after(
                        0,
                        lambda: self._update_progress(
                            index, stage, current, total, message
                        ),
                    )

                pipeline.convert_comic(
                    source,
                    output,
                    progress_callback=callback,
                    cancel_check=self.cancel_event.is_set,
                )
                self.after(0, lambda i=file_index: self._mark_complete_file(i))

            self.after(0, self._conversion_finished)
        except Exception as error:
            cancelled = self.cancel_event.is_set()
            if pipeline is not None:
                cancel_type = getattr(pipeline, "ConversionCancelled", None)
                cancelled = cancelled or (
                    cancel_type is not None and isinstance(error, cancel_type)
                )
            if cancelled:
                self._remove_partial_outputs()
                self.after(0, self._conversion_cancelled)
            else:
                error_text = str(error)
                self.after(0, lambda: self._conversion_failed(error_text))

    def _mark_active_file(self, index: int) -> None:
        for item in range(len(self.queue_states)):
            if item < index:
                self.queue_states[item] = "Complete"
            elif item == index:
                self.queue_states[item] = "Converting"
            elif self.queue_states[item] != "Complete":
                self.queue_states[item] = "Queued"
        self._refresh_queue()

    def _mark_complete_file(self, index: int) -> None:
        if 0 <= index < len(self.queue_states):
            self.queue_states[index] = "Complete"
        self._refresh_queue()

    def _update_progress(
        self,
        file_index: int,
        stage: str,
        current: int,
        total: int,
        _message: str,
    ) -> None:
        file_count = len(self.selected_files)
        file_name = self.selected_files[file_index].name

        if stage == "loading_models":
            progress = file_index / max(1, file_count)
            self.progress_bar.set(progress)
            self.status_label.configure(text=f"Loading models for {file_name}")
            self.detail_label.configure(
                text=f"Comic {file_index + 1} of {file_count}"
            )
            return

        if stage == "extracting":
            self.status_label.configure(text=f"Preparing {file_name}")
            self.detail_label.configure(text="Extracting comic pages...")
            return

        if stage == "processing":
            file_progress = current / total if total else 0
            file_progress = max(0.0, min(1.0, file_progress))
            overall = (file_index + file_progress) / max(1, file_count)
            percent = int(round(overall * 100))
            self.progress_bar.set(overall)
            self.progress_count_label.configure(
                text=f"{current} / {total}  •  {percent}%"
            )
            self.status_label.configure(text=f"Converting {file_name}")
            self.detail_label.configure(
                text=f"Comic {file_index + 1} of {file_count}"
            )
            return

        if stage == "finished":
            overall = (file_index + 1) / max(1, file_count)
            self.progress_bar.set(overall)
            self.progress_count_label.configure(text=f"{int(round(overall * 100))}%")

    def _conversion_finished(self) -> None:
        self.is_converting = False
        self.cancel_event.clear()
        self.queue_states = ["Complete" for _ in self.selected_files]
        self._refresh_queue()
        self.progress_bar.set(1)
        self.progress_count_label.configure(text="100%")

        count = len(self.output_files)
        self.status_label.configure(
            text=(
                "Conversion complete"
                if count == 1
                else f"{count} comics converted"
            ),
            text_color=self.GREEN,
        )
        self.detail_label.configure(text="Conversion completed successfully.")
        self.start_button.configure(state="normal", text="Convert Again")
        self.cancel_button.configure(state="disabled", text="Cancel")
        self.browse_button.configure(state="normal")
        self.clear_button.configure(state="normal")
        self.output_button.configure(state="normal")

    def _conversion_cancelled(self) -> None:
        self.is_converting = False
        self.cancel_event.clear()
        if 0 <= self.active_file_index < len(self.queue_states):
            self.queue_states[self.active_file_index] = "Cancelled"
        self._refresh_queue()
        self.status_label.configure(text="Conversion cancelled", text_color="#C87BB8")
        self.detail_label.configure(text="The unfinished output was removed safely.")
        self.start_button.configure(state="normal", text="Start Conversion")
        self.cancel_button.configure(state="disabled", text="Cancel")
        self.browse_button.configure(state="normal")
        self.clear_button.configure(state="normal")
        self.output_button.configure(
            state="normal" if any(path.exists() for path in self.output_files) else "disabled"
        )

    def _conversion_failed(self, error: str) -> None:
        self.is_converting = False
        self.cancel_event.clear()
        if 0 <= self.active_file_index < len(self.queue_states):
            self.queue_states[self.active_file_index] = "Failed"
        self._refresh_queue()
        self.status_label.configure(text="Conversion failed", text_color=self.RED)
        self.detail_label.configure(text=error)
        self.start_button.configure(state="normal", text="Try Again")
        self.cancel_button.configure(state="disabled", text="Cancel")
        self.browse_button.configure(state="normal")
        self.clear_button.configure(state="normal")

    def _remove_partial_outputs(self) -> None:
        for output in self.output_files:
            partial = output.with_name(output.name + ".partial")
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass

    def _show_message(self, text: str, color: str) -> None:
        self.status_label.configure(text=text, text_color=color)

    def open_output_folder(self) -> None:
        output_dir = self.get_output_directory()
        existing = [path for path in self.output_files if path.exists()]
        if len(existing) == 1:
            subprocess.Popen(["explorer.exe", "/select,", str(existing[0])])
        else:
            os.startfile(output_dir)

    def on_close(self) -> None:
        if self.is_converting:
            should_close = messagebox.askyesno(
                "Conversion in progress",
                "Stop the conversion and close the application?",
            )
            if not should_close:
                return
            self.cancel_event.set()
        self.destroy()


if __name__ == "__main__":
    ComicEbookConverterApp().mainloop()
