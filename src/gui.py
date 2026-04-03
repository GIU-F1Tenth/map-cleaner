"""
gui.py
PyQt5 GUI for the RoboRacer map cleaner.
Provides a file picker, parameter controls, live preview, and save button.
"""

from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QStatusBar, QVBoxLayout, QWidget, QRadioButton,
    QButtonGroup, QFrame, QCheckBox,
)

from map_io import build_comparison_image, load_map, pgm_to_rgb, save_map
from processing import derive_occupancy_grid, grid_stats
from sam_model import init_sam, resize_mask_to, segment_auto, segment_point


# ── worker thread so the GUI doesn't freeze during SAM inference ──────────────

class SegmentWorker(QThread):
    finished  = pyqtSignal(object, object)   # (track_mask, cleaned_grid)
    errored   = pyqtSignal(str)
    progress  = pyqtSignal(str)

    def __init__(self, model, image_rgb, original_shape,
                 use_point, point, wall_thickness):
        super().__init__()
        self.model          = model
        self.image_rgb      = image_rgb
        self.original_shape = original_shape
        self.use_point      = use_point
        self.point          = point
        self.wall_thickness = wall_thickness

    def run(self):
        try:
            self.progress.emit("Running MobileSAM segmentation…")
            if self.use_point:
                mask = segment_point(self.model, self.image_rgb, self.point)
            else:
                mask = segment_auto(self.model, self.image_rgb)

            mask    = resize_mask_to(mask, self.original_shape)
            cleaned = derive_occupancy_grid(mask, self.wall_thickness)
            self.finished.emit(mask, cleaned)
        except Exception as e:
            self.errored.emit(str(e))


# ── main window ───────────────────────────────────────────────────────────────

class MapCleanerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoboRacer Map Cleaner — MobileSAM")
        self.resize(1200, 700)

        self._yaml_path   = None
        self._original    = None
        self._meta        = None
        self._track_mask  = None
        self._cleaned     = None
        self._sam_model   = None
        self._worker      = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # ---- left panel (controls) ------------------------------------------
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(280)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setSpacing(10)

        # file picker
        file_group = QGroupBox("Map File")
        file_layout = QVBoxLayout(file_group)
        self._file_label = QLabel("No file selected")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: grey; font-size: 11px;")
        browse_btn = QPushButton("Browse .yaml…")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self._file_label)
        file_layout.addWidget(browse_btn)
        ctrl_layout.addWidget(file_group)

        # model weights
        model_group = QGroupBox("MobileSAM Weights")
        model_layout = QVBoxLayout(model_group)
        self._model_label = QLabel("mobile_sam.pt  (auto-download)")
        self._model_label.setStyleSheet("color: grey; font-size: 11px;")
        self._model_label.setWordWrap(True)
        model_browse_btn = QPushButton("Browse .pt…")
        model_browse_btn.clicked.connect(self._browse_model)
        self._weights_path = "mobile_sam.pt"
        model_layout.addWidget(self._model_label)
        model_layout.addWidget(model_browse_btn)
        ctrl_layout.addWidget(model_group)

        # prompt mode
        prompt_group = QGroupBox("Prompt Mode")
        prompt_layout = QVBoxLayout(prompt_group)
        self._mode_auto  = QRadioButton("Auto  (SAM finds track automatically)")
        self._mode_point = QRadioButton("Point  (click a pixel inside track)")
        self._mode_auto.setChecked(True)
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self._mode_auto)
        self._mode_group.addButton(self._mode_point)
        prompt_layout.addWidget(self._mode_auto)
        prompt_layout.addWidget(self._mode_point)

        point_row = QHBoxLayout()
        point_row.addWidget(QLabel("X:"))
        self._px_spin = QSpinBox(); self._px_spin.setRange(0, 9999); self._px_spin.setValue(320)
        point_row.addWidget(self._px_spin)
        point_row.addWidget(QLabel("Y:"))
        self._py_spin = QSpinBox(); self._py_spin.setRange(0, 9999); self._py_spin.setValue(240)
        point_row.addWidget(self._py_spin)
        prompt_layout.addLayout(point_row)

        hint = QLabel("Tip: hover over the preview to read pixel coordinates.")
        hint.setStyleSheet("color: grey; font-size: 10px;")
        hint.setWordWrap(True)
        prompt_layout.addWidget(hint)
        ctrl_layout.addWidget(prompt_group)

        # wall thickness
        wall_group = QGroupBox("Wall Thickness (px)")
        wall_layout = QHBoxLayout(wall_group)
        self._wall_spin = QSpinBox()
        self._wall_spin.setRange(1, 10)
        self._wall_spin.setValue(2)
        wall_layout.addWidget(self._wall_spin)
        wall_layout.addStretch()
        ctrl_layout.addWidget(wall_group)

        # run button
        self._run_btn = QPushButton("▶  Run MobileSAM")
        self._run_btn.setFixedHeight(42)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_sam)
        font = self._run_btn.font()
        font.setPointSize(11)
        self._run_btn.setFont(font)
        ctrl_layout.addWidget(self._run_btn)

        # save button
        self._save_btn = QPushButton("💾  Save Cleaned Map")
        self._save_btn.setFixedHeight(36)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_map)
        ctrl_layout.addWidget(self._save_btn)

        # stats label
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("font-size: 11px; color: #444;")
        self._stats_label.setWordWrap(True)
        ctrl_layout.addWidget(self._stats_label)

        ctrl_layout.addStretch()
        root_layout.addWidget(ctrl_panel)

        # ---- divider --------------------------------------------------------
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        root_layout.addWidget(line)

        # ---- right panel (preview) ------------------------------------------
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_label_row = QHBoxLayout()
        preview_label_row.addWidget(QLabel("Preview"))
        self._coord_label = QLabel("")
        self._coord_label.setStyleSheet("color: grey; font-size: 11px;")
        preview_label_row.addStretch()
        preview_label_row.addWidget(self._coord_label)
        preview_layout.addLayout(preview_label_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._preview_label = _ClickableImageLabel(self._on_image_click)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setText("Open a .yaml map file to get started.")
        self._preview_label.setStyleSheet("color: grey; font-size: 13px;")
        self._preview_label.mouseMoveEvent = self._on_mouse_move
        self._preview_label.setMouseTracking(True)
        scroll.setWidget(self._preview_label)
        preview_layout.addWidget(scroll)

        root_layout.addWidget(preview_panel, stretch=1)

        # status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    # ── slots ─────────────────────────────────────────────────────────────────

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Map YAML", "", "YAML files (*.yaml *.yml)"
        )
        if not path:
            return
        self._yaml_path = Path(path)
        self._file_label.setText(self._yaml_path.name)
        self._file_label.setStyleSheet("font-size: 11px;")
        try:
            self._original, self._meta = load_map(self._yaml_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return
        self._track_mask = None
        self._cleaned    = None
        self._show_original()
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._stats_label.setText("")
        self.statusBar().showMessage(f"Loaded: {self._yaml_path.name}  "
                                     f"({self._original.shape[1]}×{self._original.shape[0]})")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MobileSAM weights", "", "PyTorch weights (*.pt)"
        )
        if path:
            self._weights_path = path
            self._model_label.setText(Path(path).name)
            self._model_label.setStyleSheet("font-size: 11px;")
            self._sam_model = None  # force reload

    def _run_sam(self):
        if self._original is None:
            return

        # lazy-load model
        if self._sam_model is None:
            self.statusBar().showMessage("Loading MobileSAM model…")
            QApplication.processEvents()
            try:
                self._sam_model = init_sam(self._weights_path)
            except RuntimeError as e:
                QMessageBox.critical(self, "Model Error", str(e))
                return

        use_point = self._mode_point.isChecked()
        point     = (self._px_spin.value(), self._py_spin.value())

        image_rgb = pgm_to_rgb(self._original)

        self._run_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._worker = SegmentWorker(
            self._sam_model, image_rgb, self._original.shape,
            use_point, point, self._wall_spin.value()
        )
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.finished.connect(self._on_segment_done)
        self._worker.errored.connect(self._on_segment_error)
        self._worker.start()

    def _on_segment_done(self, track_mask, cleaned):
        self._track_mask = track_mask
        self._cleaned    = cleaned
        self._show_comparison()
        self._run_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        stats = grid_stats(cleaned)
        self._stats_label.setText(
            f"free {stats['free']}%   occupied {stats['occupied']}%   unknown {stats['unknown']}%"
        )
        self.statusBar().showMessage("Done. Review the preview then save.")

    def _on_segment_error(self, msg):
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "Segmentation Error", msg)
        self.statusBar().showMessage("Error during segmentation.")

    def _save_map(self):
        if self._cleaned is None or self._yaml_path is None:
            return
        stem = self._yaml_path.stem
        default_pgm = str(self._yaml_path.with_name(f"{stem}_sam_cleaned.pgm"))
        pgm_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaned PGM", default_pgm, "PGM files (*.pgm)"
        )
        if not pgm_path:
            return
        out_pgm  = Path(pgm_path)
        out_yaml = out_pgm.with_suffix(".yaml")
        out_png  = out_pgm.with_name(out_pgm.stem + "_preview.png")
        try:
            save_map(self._cleaned, self._meta, out_pgm, out_yaml)
            # also save comparison PNG
            comparison = build_comparison_image(self._original, self._cleaned, self._track_mask)
            cv2.imwrite(str(out_png), comparison)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return
        QMessageBox.information(
            self, "Saved",
            f"Saved:\n  {out_pgm.name}\n  {out_yaml.name}\n  {out_png.name}"
        )
        self.statusBar().showMessage(f"Saved to {out_pgm.parent}")

    def _on_image_click(self, x_label, y_label):
        """Translate click on the preview label to map pixel coords and fill spinboxes."""
        mx, my = self._label_to_map_coords(x_label, y_label)
        if mx is not None:
            self._px_spin.setValue(mx)
            self._py_spin.setValue(my)
            self._mode_point.setChecked(True)

    def _on_mouse_move(self, event):
        mx, my = self._label_to_map_coords(event.x(), event.y())
        if mx is not None:
            self._coord_label.setText(f"x={mx}  y={my}")

    def _label_to_map_coords(self, lx, ly):
        """Convert label-relative pixel coords to original map coords."""
        if self._original is None or self._pixmap is None:
            return None, None
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        lw = self._preview_label.width()
        lh = self._preview_label.height()
        # image is centred in label
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        ix = lx - ox
        iy = ly - oy
        if ix < 0 or iy < 0 or ix >= pw or iy >= ph:
            return None, None
        # the pixmap shows the comparison (2*w+20), left half = original
        # but we want map coords relative to original image width
        orig_w = self._original.shape[1]
        orig_h = self._original.shape[0]
        panel_w = pw  # full comparison width in display pixels
        # scale from display to full-res comparison
        full_w = orig_w * 2 + 20
        scale  = full_w / panel_w
        fx = int(ix * scale)
        fy = int(iy * (orig_h / ph))
        # left half of comparison = original image
        if fx > orig_w:
            fx = fx - orig_w - 20  # right half = cleaned image, still same coords
        fx = max(0, min(fx, orig_w - 1))
        fy = max(0, min(fy, orig_h - 1))
        return fx, fy

    # ── preview helpers ───────────────────────────────────────────────────────

    def _show_original(self):
        if self._original is None:
            return
        bgr = cv2.cvtColor(self._original, cv2.COLOR_GRAY2BGR)
        self._pixmap = self._bgr_to_pixmap(bgr)
        self._preview_label.setPixmap(
            self._pixmap.scaled(self._preview_label.size(),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _show_comparison(self):
        if self._cleaned is None:
            return
        comparison = build_comparison_image(self._original, self._cleaned, self._track_mask)
        self._pixmap = self._bgr_to_pixmap(comparison)
        self._preview_label.setPixmap(
            self._pixmap.scaled(self._preview_label.size(),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    @staticmethod
    def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_pixmap") and self._pixmap:
            self._preview_label.setPixmap(
                self._pixmap.scaled(self._preview_label.size(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )


class _ClickableImageLabel(QLabel):
    def __init__(self, click_cb):
        super().__init__()
        self._click_cb = click_cb
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._click_cb(event.x(), event.y())