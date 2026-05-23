"""
gui.py
RoboRacer Map Cleaner GUI using customtkinter.
"""

import threading
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from config import MAPS_DIR, OUTPUT_DIR, SHOW_COMPARISON_PREVIEW
from map_io import build_comparison_image, load_map, pgm_to_rgb, save_map
from processing import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    derive_occupancy_grid_and_mask,
    grid_stats,
)
from sam_model import init_sam, resize_mask_to, segment_points

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
FONT = ("Inter", 13)
FONT_SMALL = ("Inter", 11)
FONT_TITLE = ("Inter", 15, "bold")

POINT_COLOURS = [
    "#e53935",
    "#8e24aa",
    "#1e88e5",
    "#00897b",
    "#f4511e",
    "#3949ab",
    "#00acc1",
    "#7cb342",
]


class MapCleanerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("RoboRacer Map Cleaner")
        self.geometry("1380x820")
        self.resizable(True, True)

        self._yaml_path = None
        self._original = None
        self._meta = None
        self._track_mask = None
        self._cleaned = None
        self._sam_model = None
        self._tk_image = None
        self._points: list[tuple[int, int]] = []  # map-space points
        self._render_info = None
        self._current_rgb = None
        self._manual_walls = None
        self._manual_unknown = None
        self._manual_free = None
        self._output_white = None
        self._output_black = None
        self._output_black_base = None
        self._output_black_white_base = None
        self._last_paint_xy = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_start = None
        self._brush_cursor_ids = []
        self._last_cursor_xy = None
        self._preview_mode_var = None
        self._show_comparison = SHOW_COMPARISON_PREVIEW

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- left panel -----------------------------------------------------
        left = ctk.CTkScrollableFrame(self, width=270, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(left, text="Map Cleaner", font=FONT_TITLE).pack(
            padx=16, pady=(20, 14), anchor="w"
        )

        # file
        ctk.CTkLabel(left, text="Map file", font=FONT).pack(anchor="w", padx=16)
        self._file_label = ctk.CTkLabel(
            left,
            text="No file selected",
            font=FONT_SMALL,
            text_color="gray",
            wraplength=250,
            justify="left",
        )
        self._file_label.pack(anchor="w", padx=16, pady=(2, 6))
        ctk.CTkButton(
            left, text="Browse .yaml…", font=FONT, command=self._browse_file
        ).pack(fill="x", padx=16, pady=(0, 10))

        _sep(left)

        _sep(left)

        # points list
        ctk.CTkLabel(left, text="Prompt points", font=FONT).pack(
            anchor="w", padx=16, pady=(10, 4)
        )

        self._points_frame = ctk.CTkScrollableFrame(
            left, height=140, fg_color="#f5f5f5", corner_radius=6
        )
        self._points_frame.pack(fill="x", padx=16, pady=(0, 4))

        self._no_pts_label = ctk.CTkLabel(
            self._points_frame,
            text="No points yet — click the preview",
            font=FONT_SMALL,
            text_color="gray",
        )
        self._no_pts_label.pack(pady=8)

        ctk.CTkButton(
            left,
            text="Clear all points",
            font=FONT_SMALL,
            fg_color="#ef5350",
            hover_color="#c62828",
            command=self._clear_points,
        ).pack(fill="x", padx=16, pady=(0, 10))

        _sep(left)

        # canvas mode / manual walls
        ctk.CTkLabel(left, text="Canvas mode", font=FONT).pack(
            anchor="w", padx=16, pady=(10, 4)
        )
        self._canvas_mode_var = ctk.StringVar(value="Points")
        self._canvas_mode_top = ctk.CTkSegmentedButton(
            left,
            values=["Points", "Black", "Erase line"],
            variable=self._canvas_mode_var,
            command=self._on_canvas_mode_change,
            font=FONT_SMALL,
        )
        self._canvas_mode_top.pack(fill="x", padx=16, pady=(0, 4))
        self._canvas_mode_bottom = ctk.CTkSegmentedButton(
            left,
            values=["Gray", "Green", "White"],
            variable=self._canvas_mode_var,
            command=self._on_canvas_mode_change,
            font=FONT_SMALL,
        )
        self._canvas_mode_bottom.pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkLabel(left, text="Brush size (px)", font=FONT).pack(
            anchor="w", padx=16, pady=(0, 2)
        )
        self._brush_var = ctk.IntVar(value=2)
        brush_row = ctk.CTkFrame(left, fg_color="transparent")
        brush_row.pack(fill="x", padx=16, pady=(0, 8))
        self._brush_slider = ctk.CTkSlider(
            brush_row,
            from_=1,
            to=10,
            number_of_steps=9,
            variable=self._brush_var,
            command=self._on_brush_change,
        )
        self._brush_slider.pack(side="left", fill="x", expand=True)
        self._brush_label = ctk.CTkLabel(
            brush_row, text="2", font=FONT_SMALL, width=32
        )
        self._brush_label.pack(side="left", padx=(8, 0))

        ctk.CTkButton(
            left,
            text="Clear painted walls",
            font=FONT_SMALL,
            fg_color="#616161",
            hover_color="#424242",
            command=self._clear_manual_walls,
        ).pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkButton(
            left,
            text="Clear gray edits",
            font=FONT_SMALL,
            fg_color="#616161",
            hover_color="#424242",
            command=self._clear_manual_unknown,
        ).pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkButton(
            left,
            text="Clear green/white edits",
            font=FONT_SMALL,
            fg_color="#616161",
            hover_color="#424242",
            command=self._clear_manual_free,
        ).pack(fill="x", padx=16, pady=(0, 10))

        _sep(left)

        # wall thickness
        ctk.CTkLabel(left, text="Wall thickness (px)", font=FONT).pack(
            anchor="w", padx=16, pady=(10, 2)
        )
        self._wall_var = ctk.IntVar(value=2)
        wall_row = ctk.CTkFrame(left, fg_color="transparent")
        wall_row.pack(fill="x", padx=16, pady=(0, 10))
        self._wall_slider = ctk.CTkSlider(
            wall_row,
            from_=1,
            to=8,
            number_of_steps=7,
            variable=self._wall_var,
            command=self._on_wall_change,
        )
        self._wall_slider.pack(side="left", fill="x", expand=True)
        self._wall_label = ctk.CTkLabel(wall_row, text="2", font=FONT_SMALL, width=24)
        self._wall_label.pack(side="left", padx=(8, 0))

        # smoothing
        ctk.CTkLabel(left, text="Wall smoothing (sigma)", font=FONT).pack(
            anchor="w", padx=16, pady=(10, 2)
        )
        self._sigma_var = ctk.DoubleVar(value=3.0)
        sigma_row = ctk.CTkFrame(left, fg_color="transparent")
        sigma_row.pack(fill="x", padx=16, pady=(0, 6))
        self._sigma_slider = ctk.CTkSlider(
            sigma_row,
            from_=0.5,
            to=8.0,
            number_of_steps=15,
            variable=self._sigma_var,
            command=self._on_sigma_change,
        )
        self._sigma_slider.pack(side="left", fill="x", expand=True)
        self._sigma_label = ctk.CTkLabel(
            sigma_row, text="3.0", font=FONT_SMALL, width=30
        )
        self._sigma_label.pack(side="left", padx=(8, 0))

        # min hole area
        ctk.CTkLabel(left, text="Min noise dot area (px)", font=FONT).pack(
            anchor="w", padx=16, pady=(4, 2)
        )
        self._hole_var = ctk.IntVar(value=500)
        hole_row = ctk.CTkFrame(left, fg_color="transparent")
        hole_row.pack(fill="x", padx=16, pady=(0, 10))
        self._hole_slider = ctk.CTkSlider(
            hole_row,
            from_=50,
            to=2000,
            number_of_steps=19,
            variable=self._hole_var,
            command=self._on_hole_change,
        )
        self._hole_slider.pack(side="left", fill="x", expand=True)
        self._hole_label = ctk.CTkLabel(hole_row, text="500", font=FONT_SMALL, width=40)
        self._hole_label.pack(side="left", padx=(8, 0))

        _sep(left)

        # run / save
        self._run_btn = ctk.CTkButton(
            left, text="▶  Run", font=FONT, state="disabled", command=self._run_sam
        )
        self._run_btn.pack(fill="x", padx=16, pady=(12, 6))

        self._save_btn = ctk.CTkButton(
            left,
            text="💾  Save map",
            font=FONT,
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            state="disabled",
            command=self._save_map,
        )
        self._save_btn.pack(fill="x", padx=16, pady=(0, 10))

        self._stats_label = ctk.CTkLabel(
            left,
            text="",
            font=FONT_SMALL,
            text_color="gray",
            wraplength=255,
            justify="left",
        )
        self._stats_label.pack(anchor="w", padx=16)

        self._status = ctk.CTkLabel(
            left,
            text="Ready",
            font=FONT_SMALL,
            text_color="gray",
            wraplength=255,
            justify="left",
        )
        self._status.pack(anchor="w", padx=16, pady=(0, 16))

        # ---- right panel (preview) ------------------------------------------
        right = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        top_row = ctk.CTkFrame(right, fg_color="transparent")
        top_row.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(top_row, text="Preview", font=FONT).pack(side="left")
        self._preview_mode_var = ctk.StringVar(
            value="3 images" if SHOW_COMPARISON_PREVIEW else "Result only"
        )
        self._preview_mode_selector = ctk.CTkSegmentedButton(
            top_row,
            values=["Result only", "3 images"],
            variable=self._preview_mode_var,
            command=self._on_preview_mode_change,
            font=FONT_SMALL,
            width=160,
        )
        self._preview_mode_selector.pack(side="left", padx=(12, 2))
        ctk.CTkButton(
            top_row,
            text="-",
            width=32,
            height=24,
            font=FONT_SMALL,
            command=self._zoom_out,
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            top_row,
            text="+",
            width=32,
            height=24,
            font=FONT_SMALL,
            command=self._zoom_in,
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            top_row,
            text="Fit",
            width=44,
            height=24,
            font=FONT_SMALL,
            command=self._reset_zoom,
        ).pack(side="left", padx=2)
        self._zoom_label = ctk.CTkLabel(
            top_row, text="100%", font=FONT_SMALL, text_color="gray"
        )
        self._zoom_label.pack(side="left", padx=(6, 0))
        self._coord_label = ctk.CTkLabel(
            top_row, text="", font=FONT_SMALL, text_color="gray"
        )
        self._coord_label.pack(side="right")

        self._canvas = ctk.CTkCanvas(right, bg="#e8e8e8", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<Leave>", self._hide_brush_cursor)
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self._canvas.bind("<Button-4>", self._on_mouse_wheel)
        self._canvas.bind("<Button-5>", self._on_mouse_wheel)
        self._canvas.bind("<Button-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan_drag)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self._canvas.bind("<Button-3>", self._on_pan_start)
        self._canvas.bind("<B3-Motion>", self._on_pan_drag)
        self._canvas.bind("<ButtonRelease-3>", self._on_pan_end)

    # ── points management ─────────────────────────────────────────────────────

    def _add_point(self, mx: int, my: int):
        self._points.append((mx, my))
        self._rebuild_points_ui()
        self._redraw_preview()

    def _delete_point(self, idx: int):
        self._points.pop(idx)
        self._rebuild_points_ui()
        self._redraw_preview()

    def _clear_points(self):
        self._points.clear()
        self._rebuild_points_ui()
        self._redraw_preview()

    def _clear_manual_walls(self):
        if self._manual_walls is None:
            return
        self._manual_walls.fill(0)
        self._last_paint_xy = None
        self._mark_manual_edits_changed()
        self._redraw_preview()

    def _clear_manual_unknown(self):
        if self._manual_unknown is None:
            return
        self._manual_unknown.fill(0)
        self._last_paint_xy = None
        self._mark_manual_edits_changed()
        self._redraw_preview()

    def _clear_manual_free(self):
        if self._manual_free is None:
            return
        self._manual_free.fill(0)
        self._last_paint_xy = None
        self._mark_manual_edits_changed()
        self._redraw_preview()

    def _paint_manual_wall(self, mx: int, my: int, erase: bool):
        if self._manual_walls is None:
            return

        start = self._last_paint_xy
        self._paint_mask(
            self._manual_walls, mx, my, value=0 if erase else 1, start_point=start
        )
        if not erase and self._manual_unknown is not None:
            self._paint_mask(
                self._manual_unknown,
                mx,
                my,
                value=0,
                update_last=False,
                start_point=start,
            )
        if not erase and self._manual_free is not None:
            self._paint_mask(
                self._manual_free,
                mx,
                my,
                value=0,
                update_last=False,
                start_point=start,
            )
        self._mark_manual_edits_changed()
        self._redraw_preview()

    def _paint_manual_unknown(self, mx: int, my: int):
        if self._manual_unknown is None:
            return

        start = self._last_paint_xy
        self._paint_mask(self._manual_unknown, mx, my, value=1, start_point=start)
        if self._manual_walls is not None:
            self._paint_mask(
                self._manual_walls,
                mx,
                my,
                value=0,
                update_last=False,
                start_point=start,
            )
        if self._manual_free is not None:
            self._paint_mask(
                self._manual_free,
                mx,
                my,
                value=0,
                update_last=False,
                start_point=start,
            )
        self._mark_manual_edits_changed()
        self._redraw_preview()

    def _paint_manual_free(self, mx: int, my: int):
        if self._manual_free is None:
            return

        start = self._last_paint_xy
        self._paint_mask(self._manual_free, mx, my, value=1, start_point=start)
        if self._manual_walls is not None:
            self._paint_mask(
                self._manual_walls,
                mx,
                my,
                value=0,
                update_last=False,
                start_point=start,
            )
        if self._manual_unknown is not None:
            self._paint_mask(
                self._manual_unknown,
                mx,
                my,
                value=0,
                update_last=False,
                start_point=start,
            )
        self._mark_manual_edits_changed()
        self._redraw_preview()

    def _paint_mask(
        self,
        mask: np.ndarray,
        mx: int,
        my: int,
        value: int,
        update_last=True,
        start_point=None,
    ):
        brush = max(1, int(self._brush_var.get()))
        point = (int(mx), int(my))
        start = point if start_point is None else start_point
        cv2.line(mask, start, point, value, brush, lineType=cv2.LINE_8)

        if update_last:
            self._last_paint_xy = point

    def _output_stroke_mask(self, mx: int, my: int, start_point=None):
        if self._cleaned is None:
            return None

        stroke = np.zeros_like(self._cleaned, dtype=np.uint8)
        self._paint_mask(
            stroke,
            mx,
            my,
            value=1,
            update_last=False,
            start_point=start_point,
        )
        return stroke == 1

    def _paint_output(self, mx: int, my: int, value: int, white: bool = False):
        if self._cleaned is None:
            return

        start = self._last_paint_xy
        stroke_mask = self._output_stroke_mask(mx, my, start)
        if stroke_mask is None:
            return

        if value == OCCUPIED and self._output_black is not None:
            first_black = stroke_mask & (self._output_black == 0)
            if self._output_black_base is not None:
                self._output_black_base[first_black] = self._cleaned[first_black]
            if (
                self._output_black_white_base is not None
                and self._output_white is not None
            ):
                self._output_black_white_base[first_black] = self._output_white[
                    first_black
                ]
            self._output_black[stroke_mask] = 1
        elif self._output_black is not None:
            self._output_black[stroke_mask] = 0

        self._cleaned[stroke_mask] = value
        if self._output_white is not None:
            self._output_white[stroke_mask] = 1 if white else 0
        self._last_paint_xy = (int(mx), int(my))
        self._refresh_cleaned_preview()

    def _erase_output_black(self, mx: int, my: int):
        if (
            self._cleaned is None
            or self._output_black is None
            or self._output_black_base is None
        ):
            return

        start = self._last_paint_xy
        stroke_mask = self._output_stroke_mask(mx, my, start)
        if stroke_mask is None:
            return

        erase_mask = stroke_mask & (self._output_black == 1)
        if np.any(erase_mask):
            self._cleaned[erase_mask] = self._output_black_base[erase_mask]
            self._output_black[erase_mask] = 0
            if (
                self._output_white is not None
                and self._output_black_white_base is not None
            ):
                self._output_white[erase_mask] = self._output_black_white_base[
                    erase_mask
                ]

        self._last_paint_xy = (int(mx), int(my))
        self._refresh_cleaned_preview()

    def _show_comparison_preview(self):
        return self._show_comparison

    def _build_current_preview_image(self):
        comparison = build_comparison_image(
            self._original,
            self._cleaned,
            self._track_mask,
            self._show_comparison_preview(),
        )
        if self._output_white is None or not np.any(self._output_white):
            return comparison

        h, w = self._original.shape
        white_mask = self._output_white == 1
        if self._show_comparison_preview():
            x0 = (w + 20) * 2
            cleaned_panel = comparison[:, x0 : x0 + w]
            cleaned_panel[white_mask] = (255, 255, 255)
        else:
            comparison[white_mask] = (255, 255, 255)
        return comparison

    def _refresh_cleaned_preview(self, status_text="Output edited - review then save."):
        if self._cleaned is None:
            return

        comparison = self._build_current_preview_image()
        self._current_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        self._save_btn.configure(state="normal")
        stats = grid_stats(self._cleaned)
        self._stats_label.configure(
            text=f"free {stats['free']}%   occ {stats['occupied']}%   unk {stats['unknown']}%"
        )
        if status_text:
            self._status.configure(text=status_text)
        self._redraw_preview()

    def _mark_manual_edits_changed(self):
        self._cleaned = None
        self._output_white = None
        self._output_black = None
        self._output_black_base = None
        self._output_black_white_base = None
        self._save_btn.configure(state="disabled")
        self._stats_label.configure(text="")
        self._status.configure(
            text="Manual edits changed. Run again to apply."
        )

    def _original_with_manual_edits(self):
        has_walls = self._manual_walls is not None and np.any(self._manual_walls)
        has_unknown = self._manual_unknown is not None and np.any(self._manual_unknown)
        has_free = self._manual_free is not None and np.any(self._manual_free)
        if not has_walls and not has_unknown and not has_free:
            return self._original

        original = self._original.copy()
        if has_unknown:
            original[self._manual_unknown == 1] = 205
        if has_free:
            original[self._manual_free == 1] = 255
        if has_walls:
            original[self._manual_walls == 1] = 0
        return original

    def _rebuild_points_ui(self):
        for w in self._points_frame.winfo_children():
            w.destroy()

        if not self._points:
            self._no_pts_label = ctk.CTkLabel(
                self._points_frame,
                text="No points yet — click the preview",
                font=FONT_SMALL,
                text_color="gray",
            )
            self._no_pts_label.pack(pady=8)
            return

        for i, (px, py) in enumerate(self._points):
            colour = POINT_COLOURS[i % len(POINT_COLOURS)]
            row = ctk.CTkFrame(self._points_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            # colour dot
            dot = ctk.CTkLabel(
                row, text="●", font=("Inter", 16), text_color=colour, width=24
            )
            dot.pack(side="left")

            # coordinates
            ctk.CTkLabel(row, text=f"x={px}  y={py}", font=FONT_SMALL).pack(
                side="left", padx=4
            )

            # delete button
            btn = ctk.CTkButton(
                row,
                text="✕",
                width=28,
                height=24,
                font=FONT_SMALL,
                fg_color="#ef5350",
                hover_color="#c62828",
                command=lambda i=i: self._delete_point(i),
            )
            btn.pack(side="right", padx=4)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_wall_change(self, val):
        self._wall_label.configure(text=str(int(float(val))))

    def _on_sigma_change(self, val):
        self._sigma_label.configure(text=f"{float(val):.1f}")

    def _on_hole_change(self, val):
        self._hole_label.configure(text=str(int(float(val))))

    def _on_brush_change(self, val):
        self._brush_label.configure(text=str(int(float(val))))
        self._redraw_brush_cursor()

    def _on_preview_mode_change(self, mode=None):
        if mode is None and self._preview_mode_var is not None:
            mode = self._preview_mode_var.get()
        self._show_comparison = mode == "3 images"
        if self._preview_mode_var is not None:
            selected_mode = "3 images" if self._show_comparison else "Result only"
            self._preview_mode_var.set(selected_mode)
        self._last_paint_xy = None
        self._last_cursor_xy = None
        self._hide_brush_cursor()
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        if self._cleaned is not None:
            self._refresh_cleaned_preview(status_text="Preview mode changed.")
        elif self._current_rgb is not None:
            self._redraw_preview()

    def _on_canvas_mode_change(self, mode):
        self._last_paint_xy = None
        self._redraw_brush_cursor()
        if self._original is None:
            return
        if mode == "Points":
            self._status.configure(text="Click the preview to add prompt points.")
        elif mode == "Black":
            self._status.configure(
                text="Drag original for helper walls, or cleaned output for black cells."
            )
        elif mode == "Erase line":
            self._status.configure(
                text="Drag original to erase helper walls, or cleaned output for gray."
            )
        elif mode == "Gray":
            self._status.configure(text="Drag on the preview to gray out map pixels.")
        elif mode == "Green":
            self._status.configure(
                text="Drag to paint the track interior/free-space color."
            )
        else:
            self._status.configure(text="Drag to paint true white free-space cells.")

    def _browse_file(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            initialdir=str(MAPS_DIR),
            title="Open map YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        self._yaml_path = Path(path)
        self._file_label.configure(text=self._yaml_path.name, text_color="white")
        try:
            self._original, self._meta = load_map(self._yaml_path)
        except Exception as e:
            self._status.configure(text=f"Error: {e}")
            return
        self._track_mask = None
        self._cleaned = None
        self._manual_walls = np.zeros_like(self._original, dtype=np.uint8)
        self._manual_unknown = np.zeros_like(self._original, dtype=np.uint8)
        self._manual_free = np.zeros_like(self._original, dtype=np.uint8)
        self._output_white = None
        self._output_black = None
        self._output_black_base = None
        self._output_black_white_base = None
        self._last_paint_xy = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_start = None
        self._points.clear()
        self._rebuild_points_ui()
        self._current_rgb = cv2.cvtColor(self._original, cv2.COLOR_GRAY2RGB)
        self._render_to_canvas()
        self._run_btn.configure(state="normal")
        self._save_btn.configure(state="disabled")
        self._stats_label.configure(text="")
        self._status.configure(
            text=f"{self._original.shape[1]}×{self._original.shape[0]} px"
        )

    def _run_sam(self):
        if self._original is None:
            return
        if not self._points:
            self._status.configure(text="Add at least one point first.")
            return

        self._run_btn.configure(state="disabled", text="Running…")
        self._status.configure(text="Loading MobileSAM…")

        def worker():
            try:
                if self._sam_model is None:
                    self._sam_model = init_sam()
                self._status.configure(text="Segmenting…")
                image_rgb = pgm_to_rgb(self._original_with_manual_edits())
                mask = segment_points(self._sam_model, image_rgb, self._points)
                mask = resize_mask_to(mask, self._original.shape)
                cleaned, fitted_mask = derive_occupancy_grid_and_mask(
                    mask,
                    self._original,
                    self._wall_var.get(),
                    self._sigma_var.get(),
                    self._hole_var.get(),
                    self._manual_walls,
                    self._manual_unknown,
                    self._manual_free,
                    self._points,
                )
                self.after(0, self._on_done, fitted_mask, cleaned)
            except Exception as e:
                import traceback

                traceback.print_exc()
                self.after(0, self._on_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, mask, cleaned):
        self._track_mask = mask
        self._cleaned = cleaned
        self._output_white = np.zeros_like(cleaned, dtype=np.uint8)
        self._output_black = np.zeros_like(cleaned, dtype=np.uint8)
        self._output_black_base = cleaned.copy()
        self._output_black_white_base = np.zeros_like(cleaned, dtype=np.uint8)
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_start = None
        comparison = self._build_current_preview_image()
        self._current_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        self._render_to_canvas()
        self._run_btn.configure(state="normal", text="▶  Run")
        self._save_btn.configure(state="normal")
        stats = grid_stats(cleaned)
        self._stats_label.configure(
            text=f"free {stats['free']}%   occ {stats['occupied']}%   unk {stats['unknown']}%"
        )
        self._status.configure(text="Done — review then save.")

    def _on_error(self, msg):
        self._run_btn.configure(state="normal", text="▶  Run")
        self._status.configure(text=f"Error: {msg[:120]}")
        from tkinter import messagebox

        messagebox.showerror("Segmentation Error", msg)

    def _save_map(self):
        if self._cleaned is None:
            return
        from tkinter import filedialog

        stem = self._yaml_path.stem
        path = filedialog.asksaveasfilename(
            initialdir=str(OUTPUT_DIR),
            initialfile=f"{stem}_sam_cleaned.png",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if not path:
            return
        out_image = Path(path)
        out_yaml = out_image.with_suffix(".yaml")
        out_preview = out_image.with_name(out_image.stem + "_preview.png")
        save_map(self._cleaned, self._meta, out_image, out_yaml)
        comparison = self._build_current_preview_image()
        cv2.imwrite(str(out_preview), comparison)
        self._status.configure(text=f"Saved to {out_image.parent}")
        from tkinter import messagebox

        messagebox.showinfo(
            "Saved", f"Saved:\n  {out_image.name}\n  {out_yaml.name}\n  {out_preview.name}"
        )

    # ── preview ───────────────────────────────────────────────────────────────

    def _redraw_preview(self):
        """Re-render current image with point markers overlaid."""
        if self._current_rgb is None:
            return
        display = self._current_rgb.copy()
        self._draw_manual_edits_overlay(display)
        show_points = self._points and (
            self._show_comparison_preview() or self._cleaned is None
        )
        if self._render_info and show_points:
            scale = self._render_info["scale"]
            orig_w = (
                self._original.shape[1]
                if self._original is not None
                else display.shape[1]
            )
            for i, (px, py) in enumerate(self._points):
                colour_hex = POINT_COLOURS[i % len(POINT_COLOURS)]
                colour_bgr = _hex_to_bgr(colour_hex)
                # draw on display image directly — convert map coords to display coords
                dx = int(px)
                dy = int(py)
                cv2.circle(display, (dx, dy), 10, colour_bgr, -1)
                cv2.circle(display, (dx, dy), 10, (255, 255, 255), 2)
                cv2.circle(display, (dx, dy), 3, (255, 255, 255), -1)
                cv2.putText(
                    display,
                    str(i + 1),
                    (dx + 13, dy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour_bgr,
                    2,
                )
        self._show_image(display)

    def _draw_manual_edits_overlay(self, display: np.ndarray):
        if self._cleaned is not None and not self._show_comparison_preview():
            return

        has_walls = self._manual_walls is not None and np.any(self._manual_walls)
        has_unknown = self._manual_unknown is not None and np.any(self._manual_unknown)
        has_free = self._manual_free is not None and np.any(self._manual_free)
        if not has_walls and not has_unknown and not has_free:
            return

        h, w = self._original.shape
        wall_mask = self._manual_walls == 1 if has_walls else None
        unknown_mask = self._manual_unknown == 1 if has_unknown else None
        free_mask = self._manual_free == 1 if has_free else None
        panel_span = w + 20
        panel_count = max(1, (display.shape[1] + 20) // panel_span)
        overlay_panels = 1

        for panel_idx in range(overlay_panels):
            x0 = panel_idx * panel_span
            if x0 + w > display.shape[1] or display.shape[0] != h:
                continue
            panel = display[:, x0 : x0 + w]
            if unknown_mask is not None:
                panel[unknown_mask] = (205, 205, 205)
            if free_mask is not None:
                panel[free_mask] = (255, 255, 255)
            if wall_mask is not None:
                panel[wall_mask] = (0, 0, 0)

    def _show_image(self, rgb: np.ndarray):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        ih, iw = rgb.shape[:2]
        base_scale = min(cw / iw, ch / ih)
        scale = base_scale * self._zoom
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        self._clamp_pan(nw, nh, cw, ch)
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
        resized = cv2.resize(rgb, (nw, nh), interpolation=interpolation)
        pil_img = Image.fromarray(resized)
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        cx = cw / 2 + self._pan_x
        cy = ch / 2 + self._pan_y
        self._canvas.create_image(
            int(cx), int(cy), anchor="center", image=self._tk_image
        )
        self._render_info = {
            "scale": scale,
            "nw": nw,
            "nh": nh,
            "ox": cx - nw / 2,
            "oy": cy - nh / 2,
            "orig_w": rgb.shape[1],
            "orig_h": rgb.shape[0],
        }
        self._zoom_label.configure(text=f"{int(round(self._zoom * 100))}%")
        self._redraw_brush_cursor()

    def _clamp_pan(self, image_w: int, image_h: int, canvas_w: int, canvas_h: int):
        max_x = max(0, (image_w - canvas_w) / 2)
        max_y = max(0, (image_h - canvas_h) / 2)
        self._pan_x = min(max(self._pan_x, -max_x), max_x)
        self._pan_y = min(max(self._pan_y, -max_y), max_y)

    def _render_to_canvas(self):
        if self._current_rgb is not None:
            self._redraw_preview()

    def _on_canvas_resize(self, event):
        self._render_to_canvas()

    def _hide_brush_cursor(self, event=None):
        for cursor_id in self._brush_cursor_ids:
            self._canvas.delete(cursor_id)
        self._brush_cursor_ids = []
        if event is not None:
            self._last_cursor_xy = None

    def _redraw_brush_cursor(self):
        if self._last_cursor_xy is None:
            self._hide_brush_cursor()
            return
        self._update_brush_cursor(*self._last_cursor_xy)

    def _update_brush_cursor(self, cx, cy):
        self._hide_brush_cursor()
        self._last_cursor_xy = (cx, cy)
        mode = self._canvas_mode_var.get()
        if mode == "Points" or not self._render_info:
            return

        _mx, _my, panel_idx = self._canvas_to_target(cx, cy)
        if panel_idx is None:
            return
        if panel_idx != 0 and not self._is_cleaned_panel(panel_idx):
            return
        if self._is_cleaned_panel(panel_idx) and not self._can_edit_cleaned_panel(mode):
            return

        brush_px = max(1, int(self._brush_var.get())) * self._render_info["scale"]
        radius = max(2.0, brush_px / 2)
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius, cy + radius
        self._brush_cursor_ids = [
            self._canvas.create_oval(
                x0,
                y0,
                x1,
                y1,
                outline="white",
                width=3,
            ),
            self._canvas.create_oval(
                x0,
                y0,
                x1,
                y1,
                outline="black",
                width=1,
            ),
        ]

    def _zoom_in(self):
        self._set_zoom(self._zoom * 1.2, self._canvas_center())

    def _zoom_out(self):
        self._set_zoom(self._zoom / 1.2, self._canvas_center())

    def _reset_zoom(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._render_to_canvas()

    def _canvas_center(self):
        return (self._canvas.winfo_width() / 2, self._canvas.winfo_height() / 2)

    def _on_mouse_wheel(self, event):
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        self._set_zoom(self._zoom * factor, (event.x, event.y))
        return "break"

    def _set_zoom(self, zoom: float, anchor=None):
        if self._current_rgb is None:
            return

        old_info = self._render_info
        old_zoom = self._zoom
        self._zoom = max(0.25, min(8.0, float(zoom)))
        if abs(self._zoom - old_zoom) < 0.001:
            return

        if anchor and old_info:
            ax, ay = anchor
            image_x = (ax - old_info["ox"]) / old_info["scale"]
            image_y = (ay - old_info["oy"]) / old_info["scale"]
            ih, iw = self._current_rgb.shape[:2]
            if 0 <= image_x < iw and 0 <= image_y < ih:
                cw = self._canvas.winfo_width()
                ch = self._canvas.winfo_height()
                base_scale = min(cw / iw, ch / ih)
                new_scale = base_scale * self._zoom
                self._pan_x = ax + (iw / 2 - image_x) * new_scale - cw / 2
                self._pan_y = ay + (ih / 2 - image_y) * new_scale - ch / 2

        self._render_to_canvas()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y, self._pan_x, self._pan_y)
        self._hide_brush_cursor(event)
        self._canvas.configure(cursor="fleur")

    def _on_pan_drag(self, event):
        if self._pan_start is None:
            return
        start_x, start_y, start_pan_x, start_pan_y = self._pan_start
        self._pan_x = start_pan_x + event.x - start_x
        self._pan_y = start_pan_y + event.y - start_y
        self._render_to_canvas()

    def _on_pan_end(self, event):
        self._pan_start = None
        self._canvas.configure(cursor="")

    def _canvas_to_target(self, cx, cy):
        if not self._render_info or self._original is None:
            return None, None, None
        r = self._render_info
        ix = cx - r["ox"]
        iy = cy - r["oy"]
        if ix < 0 or iy < 0 or ix >= r["nw"] or iy >= r["nh"]:
            return None, None, None
        mx = int(ix / r["scale"])
        my = int(iy / r["scale"])
        orig_w = self._original.shape[1]
        panel_idx = 0
        if r["orig_w"] > orig_w:
            panel_span = orig_w + 20
            panel_idx = int(mx // panel_span)
            panel_x = mx % panel_span
            if panel_x >= orig_w:
                return None, None, None
            mx = panel_x
        mx = max(0, min(mx, orig_w - 1))
        my = max(0, min(my, self._original.shape[0] - 1))
        return mx, my, panel_idx

    def _canvas_to_map(self, cx, cy):
        mx, my, _panel_idx = self._canvas_to_target(cx, cy)
        return mx, my

    def _is_cleaned_panel(self, panel_idx):
        if self._cleaned is None:
            return False
        if not self._show_comparison_preview():
            return True
        return panel_idx == 2

    def _can_edit_cleaned_panel(self, mode):
        return mode == "Erase line" or self._output_value_for_mode(mode) is not None

    def _output_value_for_mode(self, mode):
        if mode == "Black":
            return OCCUPIED
        if mode == "Gray":
            return UNKNOWN
        if mode in ("Green", "White"):
            return FREE
        return None

    def _on_mouse_move(self, event):
        self._update_brush_cursor(event.x, event.y)
        mx, my = self._canvas_to_map(event.x, event.y)
        if mx is not None:
            self._coord_label.configure(text=f"x={mx}  y={my}")

    def _on_canvas_click(self, event):
        self._update_brush_cursor(event.x, event.y)
        mx, my, panel_idx = self._canvas_to_target(event.x, event.y)
        if mx is None:
            return

        mode = self._canvas_mode_var.get()
        if self._is_cleaned_panel(panel_idx):
            if mode == "Erase line":
                self._erase_output_black(mx, my)
                return
            value = self._output_value_for_mode(mode)
            if value is not None:
                self._paint_output(mx, my, value, white=(mode == "White"))
            return

        if panel_idx != 0:
            return

        if mode == "Points":
            self._add_point(mx, my)
            return

        if mode == "Gray":
            self._paint_manual_unknown(mx, my)
        elif mode in ("Green", "White"):
            self._paint_manual_free(mx, my)
        else:
            self._paint_manual_wall(mx, my, erase=(mode == "Erase line"))

    def _on_canvas_drag(self, event):
        mode = self._canvas_mode_var.get()
        self._update_brush_cursor(event.x, event.y)

        mx, my, panel_idx = self._canvas_to_target(event.x, event.y)
        if mx is None:
            self._last_paint_xy = None
            return

        if self._is_cleaned_panel(panel_idx):
            if mode == "Erase line":
                self._erase_output_black(mx, my)
                return
            value = self._output_value_for_mode(mode)
            if value is not None:
                self._paint_output(mx, my, value, white=(mode == "White"))
            return

        if mode == "Points" or panel_idx != 0:
            return

        if mode == "Gray":
            self._paint_manual_unknown(mx, my)
        elif mode in ("Green", "White"):
            self._paint_manual_free(mx, my)
        else:
            self._paint_manual_wall(mx, my, erase=(mode == "Erase line"))

    def _on_canvas_release(self, event):
        self._last_paint_xy = None


# ── helpers ───────────────────────────────────────────────────────────────────


def _sep(parent):
    ctk.CTkFrame(parent, height=1, fg_color="#e0e0e0").pack(fill="x", padx=16, pady=4)


def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
