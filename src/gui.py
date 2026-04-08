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

from config import MAPS_DIR, OUTPUT_DIR
from map_io import build_comparison_image, load_map, pgm_to_rgb, save_map
from processing import derive_occupancy_grid, grid_stats
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
        ctk.CTkLabel(left, text="Boundary smoothing (sigma)", font=FONT).pack(
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
        self._coord_label = ctk.CTkLabel(
            top_row, text="", font=FONT_SMALL, text_color="gray"
        )
        self._coord_label.pack(side="right")

        self._canvas = ctk.CTkCanvas(right, bg="#e8e8e8", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

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
                image_rgb = pgm_to_rgb(self._original)
                mask = segment_points(self._sam_model, image_rgb, self._points)
                mask = resize_mask_to(mask, self._original.shape)
                cleaned = derive_occupancy_grid(
                    mask,
                    self._original,
                    self._wall_var.get(),
                    self._sigma_var.get(),
                    self._hole_var.get(),
                )
                self.after(0, self._on_done, mask, cleaned)
            except Exception as e:
                import traceback

                traceback.print_exc()
                self.after(0, self._on_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, mask, cleaned):
        self._track_mask = mask
        self._cleaned = cleaned
        comparison = build_comparison_image(self._original, cleaned, mask)
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
            initialfile=f"{stem}_sam_cleaned.pgm",
            filetypes=[("PGM files", "*.pgm")],
        )
        if not path:
            return
        out_pgm = Path(path)
        out_yaml = out_pgm.with_suffix(".yaml")
        out_png = out_pgm.with_name(out_pgm.stem + "_preview.png")
        save_map(self._cleaned, self._meta, out_pgm, out_yaml)
        comparison = build_comparison_image(
            self._original, self._cleaned, self._track_mask
        )
        cv2.imwrite(str(out_png), comparison)
        self._status.configure(text=f"Saved to {out_pgm.parent}")
        from tkinter import messagebox

        messagebox.showinfo(
            "Saved", f"Saved:\n  {out_pgm.name}\n  {out_yaml.name}\n  {out_png.name}"
        )

    # ── preview ───────────────────────────────────────────────────────────────

    def _redraw_preview(self):
        """Re-render current image with point markers overlaid."""
        if self._current_rgb is None:
            return
        display = self._current_rgb.copy()
        if self._render_info and self._points:
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

    def _show_image(self, rgb: np.ndarray):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        ih, iw = rgb.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self._tk_image = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(
            cw // 2, ch // 2, anchor="center", image=self._tk_image
        )
        self._render_info = {
            "scale": scale,
            "nw": nw,
            "nh": nh,
            "ox": (cw - nw) // 2,
            "oy": (ch - nh) // 2,
            "orig_w": rgb.shape[1],
            "orig_h": rgb.shape[0],
        }

    def _render_to_canvas(self):
        if self._current_rgb is not None:
            self._redraw_preview()

    def _on_canvas_resize(self, event):
        self._render_to_canvas()

    def _canvas_to_map(self, cx, cy):
        if not self._render_info or self._original is None:
            return None, None
        r = self._render_info
        ix = cx - r["ox"]
        iy = cy - r["oy"]
        if ix < 0 or iy < 0 or ix >= r["nw"] or iy >= r["nh"]:
            return None, None
        mx = int(ix / r["scale"])
        my = int(iy / r["scale"])
        orig_w = self._original.shape[1]
        if mx >= orig_w:
            mx = mx - orig_w - 20
        mx = max(0, min(mx, orig_w - 1))
        my = max(0, min(my, self._original.shape[0] - 1))
        return mx, my

    def _on_mouse_move(self, event):
        mx, my = self._canvas_to_map(event.x, event.y)
        if mx is not None:
            self._coord_label.configure(text=f"x={mx}  y={my}")

    def _on_canvas_click(self, event):
        mx, my = self._canvas_to_map(event.x, event.y)
        if mx is not None:
            self._add_point(mx, my)


# ── helpers ───────────────────────────────────────────────────────────────────


def _sep(parent):
    ctk.CTkFrame(parent, height=1, fg_color="#e0e0e0").pack(fill="x", padx=16, pady=4)


def _hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
