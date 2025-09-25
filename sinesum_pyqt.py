# sinesum2_pyqt_topplots_split.py
# PyQt GUI for "Sum of Sines" — large plots on TOP,
# bottom area split: compact controls (LEFT) and big table (RIGHT).
# Works with PyQt6 or PyQt5 (auto-fallback).

from __future__ import annotations
import json, os, sys, math, logging
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

# ---- Qt bindings (prefer PyQt6; fall back to PyQt5) ----
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QComboBox, QSpinBox, QSlider, QTableWidget,
        QTableWidgetItem, QFileDialog, QMessageBox, QLineEdit, QGroupBox
    )
    from PyQt6.QtCore import Qt
    PYQT6 = True
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QComboBox, QSpinBox, QSlider, QTableWidget,
        QTableWidgetItem, QFileDialog, QMessageBox, QLineEdit, QGroupBox
    )
    from PyQt5.QtCore import Qt
    PYQT6 = False

# ---- Optional audio backends ----
_SD_OK = False
try:
    import sounddevice as sd
    _SD_OK = True
except Exception:
    _SD_OK = False


_SCIPY_WAV_OK = False
try:
    from scipy.io import wavfile
    _SCIPY_WAV_OK = True
except Exception:
    _SCIPY_WAV_OK = False

# ---- Matplotlib embedding ----
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---- Logger setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def zero_line(ax):
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.grid(True, ls="--", alpha=0.5)


class PlotCanvas(FigureCanvas):
    """Big canvas with 3 panels: spectral (left), current (top-right), combined (bottom-right)."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 7), layout="constrained", facecolor="white")
        super().__init__(self.fig)
        self.setParent(parent)

        gs = self.fig.add_gridspec(
            2, 2, width_ratios=[1.0, 1.65], height_ratios=[1.0, 1.8], wspace=0.22, hspace=0.05
        )
        self.ax_spec = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_current = self.fig.add_subplot(gs[0, 1])
        self.ax_combined = self.fig.add_subplot(gs[1, 1])

        for ax in [self.ax_current, self.ax_combined]:
            ax.set_facecolor("white")
        self.ax_spec.set_facecolor("white")

    def draw_all(self, t, A, P, k_idx):
        N = len(A); k = k_idx

        # Combined
        x = np.zeros_like(t)
        for n in range(N):
            x += A[n] * np.sin(2*np.pi*(n+1)*t + P[n])
        self.ax_combined.clear()
        self.ax_combined.plot(t, x, lw=1.25)
        zero_line(self.ax_combined)
        maxX = max(1e-4, np.max(np.abs(x)), float(A[k])) * 1.1
        self.ax_combined.set_xlim(0, t[-1])
        self.ax_combined.set_ylim(-maxX, maxX)
        self.ax_combined.set_title("Combined Signal")

        # Current harmonic
        y = A[k] * np.sin(2*np.pi*(k+1)*t + P[k])
        self.ax_current.clear()
        self.ax_current.plot(t, y, lw=1.25)
        zero_line(self.ax_current)
        self.ax_current.set_xlim(0, t[-1])
        self.ax_current.set_ylim(-maxX, maxX)
        self.ax_current.set_title(f"Harmonic {k+1}")

        # Spectral profile (3D stems)
        self.ax_spec.clear()
        self.ax_spec.set_title("Spectral Profile", pad=8)
        self.ax_spec.set_xlabel("Harmonic", labelpad=8)
        self.ax_spec.set_ylabel("Phase", labelpad=10)
        self.ax_spec.set_zlabel("Amplitude", labelpad=6)
        self.ax_spec.grid(True)
        if N > 0:
            for i in range(N):
                h = i + 1
                a = float(A[i]); ph = float(P[i] % (2*np.pi))
                self.ax_spec.plot([h, h], [ph, ph], [0, a], lw=2)
                self.ax_spec.scatter([h], [ph], [a], s=28)
            hc = k + 1
            ac = float(A[k]); pc = float(P[k] % (2*np.pi))
            self.ax_spec.plot([hc, hc], [pc, pc], [0, ac], lw=3)
            self.ax_spec.scatter([hc], [pc], [ac], s=60)
        self.ax_spec.set_xlim(0, N + 1)
        self.ax_spec.set_ylim(0, 2*np.pi)
        zmax = (np.max(A) if N else 0) * 1.1
        self.ax_spec.set_zlim(0, zmax)

        self.draw_idle()


class SineSumWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing SineSumWindow")
        self.setWindowTitle("Sum of Sines (PyQt)")
        self.resize(1300, 820)

        # ---- state
        self.num_harmonics = 10
        self.current_harmonic = 1  # 1-indexed UI
        self.amplitudes = np.zeros(self.num_harmonics, dtype=float)
        self.phases = np.zeros(self.num_harmonics, dtype=float)
        logger.debug(f"Initial state: {self.num_harmonics} harmonics, current harmonic: {self.current_harmonic}")

        self.maxT = 2.0
        self.dt = 1/200.0
        self.t = np.arange(0, self.maxT + 1e-12, self.dt)

        self.Fs = 44100
        self.num_seconds = 2.0
        self.f0 = 500.0

        # ---- root layout
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # TOP plots (large)
        self.canvas = PlotCanvas(self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Expanding)
        root.addWidget(self.canvas, stretch=5)

        # BOTTOM split: LEFT controls (compact) + RIGHT big table
        bottom = QHBoxLayout()
        root.addLayout(bottom, stretch=2)

        # LEFT controls group
        controls = QGroupBox("Controls")
        controls.setStyleSheet("QGroupBox { font-weight: bold; }")
        controls.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                               QtWidgets.QSizePolicy.Policy.Preferred)
        bottom.addWidget(controls, stretch=3)  # smaller than table
        grid = QGridLayout(controls)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        row = 0

        # Row A: Start Over with [N]
        self.btn_start = QPushButton("Start Over"); self.btn_start.setFixedWidth(100)
        self.spin_N = QSpinBox(); self.spin_N.setRange(1, 999); self.spin_N.setValue(self.num_harmonics); self.spin_N.setFixedWidth(70)
        grid.addWidget(self.btn_start, row, 0)
        grid.addWidget(QLabel("with"), row, 1)
        grid.addWidget(self.spin_N, row, 2)
        grid.addWidget(QLabel("harmonics"), row, 3)
        row += 1

        # Row B: Adjusting Harmonic Number
        grid.addWidget(QLabel("Adjusting Harmonic Number:"), row, 0, 1, 4)
        row += 1
        self.btn_prev = QPushButton("Previous"); self.btn_prev.setFixedWidth(90)
        self.combo_k = QComboBox(); self.combo_k.setFixedWidth(80)
        self.btn_next = QPushButton("Next"); self.btn_next.setFixedWidth(70)
        grid.addWidget(self.btn_prev, row, 0)
        grid.addWidget(self.combo_k, row, 1, 1, 2)
        grid.addWidget(self.btn_next, row, 3)
        row += 1

        # Row C: Amplitude
        grid.addWidget(QLabel("Adjust Amplitude:"), row, 0, 1, 4)
        row += 1
        self.slider_amp = QSlider(Qt.Orientation.Horizontal); self.slider_amp.setRange(0, 200)
        self.edit_amp = QLineEdit("0.0"); self.edit_amp.setFixedWidth(90)
        grid.addWidget(self.slider_amp, row, 0, 1, 3)
        grid.addWidget(self.edit_amp, row, 3)
        row += 1

        # Row D: Phase
        grid.addWidget(QLabel("Adjust Phase:"), row, 0, 1, 4)
        row += 1
        self.slider_phase = QSlider(Qt.Orientation.Horizontal); self.slider_phase.setRange(0, 6283)
        self.edit_phase = QLineEdit("0.0"); self.edit_phase.setFixedWidth(90)
        grid.addWidget(self.slider_phase, row, 0, 1, 3)
        grid.addWidget(self.edit_phase, row, 3)
        row += 1

        # Row E: Save/Load/About + Play
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Save"); self.btn_save.setFixedWidth(70)
        self.btn_load = QPushButton("Load"); self.btn_load.setFixedWidth(70)
        self.btn_about = QPushButton("About"); self.btn_about.setFixedWidth(70)
        self.btn_play = QPushButton("Play Sound"); self.btn_play.setFixedWidth(110)
        self.btn_save_sound = QPushButton("Save WAV"); self.btn_save_sound.setFixedWidth(90)
        btn_row.addWidget(self.btn_save); btn_row.addWidget(self.btn_load); btn_row.addWidget(self.btn_about)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_save_sound)
        grid.addLayout(btn_row, row, 0, 1, 4)

        # RIGHT: Big table
        right_box = QGroupBox("Harmonics")
        right_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        bottom.addWidget(right_box, stretch=5)  # larger than controls
        v = QVBoxLayout(right_box)
        self.table = QTableWidget(self.num_harmonics, 3)
        self.table.setHorizontalHeaderLabels(["Harmonic", "Amplitude", "Phase"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(160)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)
        v.addWidget(self.table)

        # wire up
        self.btn_start.clicked.connect(self.on_start_over)
        self.spin_N.valueChanged.connect(self.on_change_N)
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)
        self.combo_k.currentIndexChanged.connect(self.on_combo_changed)
        self.table.cellClicked.connect(self.on_table_clicked)
        self.slider_amp.valueChanged.connect(self.on_slider_amp)
        self.slider_phase.valueChanged.connect(self.on_slider_phase)
        self.edit_amp.editingFinished.connect(self.on_edit_amp)
        self.edit_phase.editingFinished.connect(self.on_edit_phase)
        self.btn_play.clicked.connect(self.on_play)
        self.btn_save_sound.clicked.connect(self.on_save_audio)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_load.clicked.connect(self.on_load)
        self.btn_about.clicked.connect(self.on_about)

        # init
        self._refresh_combo()
        self._sync_controls()
        self._refresh_table()
        self._redraw()

    # ---------- helpers ----------
    @property
    def k0(self):  # 0-index
        return max(0, min(self.num_harmonics - 1, self.current_harmonic - 1))

    def _refresh_combo(self):
        self.combo_k.blockSignals(True)
        self.combo_k.clear()
        for n in range(1, self.num_harmonics + 1):
            self.combo_k.addItem(str(n))
        self.combo_k.setCurrentIndex(self.k0)
        self.combo_k.blockSignals(False)

    def _refresh_table(self):
        self.table.setRowCount(self.num_harmonics)
        for r in range(self.num_harmonics):
            self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.table.setItem(r, 1, QTableWidgetItem(f"{self.amplitudes[r]:.4f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{self.phases[r]:.4f}"))
        self.table.resizeColumnsToContents()
        self.table.selectRow(self.k0)

    def _sync_controls(self):
        self.btn_prev.setEnabled(self.k0 > 0)
        self.btn_next.setEnabled(self.k0 < self.num_harmonics - 1)
        self.spin_N.blockSignals(True); self.spin_N.setValue(self.num_harmonics); self.spin_N.blockSignals(False)
        self.combo_k.blockSignals(True); self.combo_k.setCurrentIndex(self.k0); self.combo_k.blockSignals(False)
        a = float(self.amplitudes[self.k0]); p = float(self.phases[self.k0])
        self.slider_amp.blockSignals(True); self.slider_amp.setValue(int(round(a * 100))); self.slider_amp.blockSignals(False)
        self.slider_phase.blockSignals(True); self.slider_phase.setValue(int(round((p % (2*math.pi)) * 1000))); self.slider_phase.blockSignals(False)
        self.edit_amp.setText(f"{a:.6g}"); self.edit_phase.setText(f"{p:.6g}")

    def _redraw(self):
        self.canvas.draw_all(self.t, self.amplitudes, self.phases, self.k0)

    def _resize_state(self, N):
        N = max(1, int(N))
        old_num = self.num_harmonics
        oldA, oldP = self.amplitudes, self.phases
        self.amplitudes = np.zeros(N, dtype=float)
        self.phases = np.zeros(N, dtype=float)
        m = min(len(oldA), N)
        self.amplitudes[:m] = oldA[:m]
        self.phases[:m] = oldP[:m]
        self.num_harmonics = N
        self.current_harmonic = 1
        logger.debug(f"Resized state from {old_num} to {N} harmonics, preserved {m} values")

    # ---------- slots ----------
    def on_change_N(self, val):
        self._resize_state(val); self._refresh_combo(); self._sync_controls(); self._refresh_table(); self._redraw()

    def on_start_over(self):
        self._resize_state(self.spin_N.value()); self._refresh_combo(); self._sync_controls(); self._refresh_table(); self._redraw()

    def on_prev(self):
        if self.k0 > 0:
            self.current_harmonic -= 1
            logger.debug(f"Navigation: Previous harmonic selected ({self.current_harmonic})")
            self._sync_controls(); self._refresh_table(); self._redraw()

    def on_next(self):
        if self.k0 < self.num_harmonics - 1:
            self.current_harmonic += 1
            logger.debug(f"Navigation: Next harmonic selected ({self.current_harmonic})")
            self._sync_controls(); self._refresh_table(); self._redraw()

    def on_combo_changed(self, idx):
        if idx < 0: return
        old_harmonic = self.current_harmonic
        self.current_harmonic = idx + 1
        logger.debug(f"Combo selection: Changed from harmonic {old_harmonic} to {self.current_harmonic}")
        self._sync_controls(); self._refresh_table(); self._redraw()

    def on_table_clicked(self, row, _col):
        old_harmonic = self.current_harmonic
        self.current_harmonic = row + 1
        logger.debug(f"Table selection: Changed from harmonic {old_harmonic} to {self.current_harmonic}")
        self._sync_controls(); self._refresh_table(); self._redraw()

    def on_slider_amp(self, value):
        a = value / 100.0
        old_a = self.amplitudes[self.k0]
        self.amplitudes[self.k0] = a
        self.edit_amp.setText(f"{a:.6g}")
        logger.debug(f"Amplitude slider: Harmonic {self.current_harmonic} changed from {old_a:.4f} to {a:.4f}")
        self._refresh_table(); self._redraw()

    def on_slider_phase(self, value):
        p = value / 1000.0
        old_p = self.phases[self.k0]
        self.phases[self.k0] = p
        self.edit_phase.setText(f"{p:.6g}")
        logger.debug(f"Phase slider: Harmonic {self.current_harmonic} changed from {old_p:.4f} to {p:.4f}")
        self._refresh_table(); self._redraw()

    def on_edit_amp(self):
        old_a = self.amplitudes[self.k0]
        try: a = float(self.edit_amp.text())
        except Exception: 
            a = old_a
            logger.warning(f"Invalid amplitude input for harmonic {self.current_harmonic}, reverting to {a:.4f}")
        self.amplitudes[self.k0] = a; self.slider_amp.setValue(int(round(a * 100)))
        if a != old_a:
            logger.debug(f"Amplitude edit: Harmonic {self.current_harmonic} changed from {old_a:.4f} to {a:.4f}")
        self._refresh_table(); self._redraw()

    def on_edit_phase(self):
        old_p = self.phases[self.k0]
        try: p = float(self.edit_phase.text())
        except Exception: 
            p = old_p
            logger.warning(f"Invalid phase input for harmonic {self.current_harmonic}, reverting to {p:.4f}")
        self.phases[self.k0] = p; self.slider_phase.setValue(int(round((p % (2*math.pi)) * 1000)))
        if p != old_p:
            logger.debug(f"Phase edit: Harmonic {self.current_harmonic} changed from {old_p:.4f} to {p:.4f}")
        self._refresh_table(); self._redraw()

    def on_play(self):
        logger.info("Playing audio with current harmonic configuration")
        t = np.arange(0, self.num_seconds, 1/self.Fs)
        x = np.zeros_like(t)
        for n in range(self.num_harmonics):
            x += self.amplitudes[n] * np.sin(2*np.pi*(n+1)*self.f0*t + self.phases[n])
        denom = np.max(np.abs(x)) + 0.05
        x = (x / denom).astype(np.float32)
        logger.debug(f"Generated audio signal: {len(x)} samples, max amplitude: {np.max(np.abs(x)):.4f}")
        if _SD_OK:
            sd.stop(ignore_errors=True); sd.play(x, samplerate=self.Fs, blocking=False)
            logger.info(f"Playing audio via sounddevice: {self.num_seconds}s at {self.Fs}Hz")
        else:
            logger.warning("sounddevice library not available for audio playback")
            QMessageBox.information(self, "Library Error", f"Sound Device Library not found.\n Install 'sounddevice' via pip.")

    def on_save_audio(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save WAV Audio", "sinesum2_sound.wav", "WAV (*.wav)")
        if not fname: 
            logger.info("Audio save cancelled by user")
            return
        logger.info(f"Saving audio to: {fname}")
        t = np.arange(0, self.num_seconds, 1/self.Fs)
        x = np.zeros_like(t)
        for n in range(self.num_harmonics):
            x += self.amplitudes[n] * np.sin(2*np.pi*(n+1)*self.f0*t + self.phases[n])
        denom = np.max(np.abs(x)) + 0.05
        x = (x / denom).astype(np.float32)
        path = fname
        if _SCIPY_WAV_OK:
            wavfile.write(path, self.Fs, (x * 32767).astype(np.int16))
            logger.info(f"Successfully saved WAV file: {path}")
            QMessageBox.information(self, "Audio", f"Saved WAV to:\n{path}")
        else:
            logger.error("scipy library not available for WAV file saving")
            QMessageBox.information(self, "Library Error", f"scipy Library not found.\n Install 'scipy' via pip.")

    def on_save(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Sine Sum Project",
                                               "sinesum2_project.json", "JSON (*.json)")
        if not fname: 
            logger.info("Project save cancelled by user")
            return
        logger.info(f"Saving project to: {fname}")
        try:
            data = {"Amplitudes": self.amplitudes.tolist(), "Phases": self.phases.tolist()}
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully saved project: {os.path.basename(fname)} with {self.num_harmonics} harmonics")
            QMessageBox.information(self, "Saved", os.path.basename(fname))
        except Exception as e:
            logger.error(f"Failed to save project file {fname}: {e}")
            QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")

    def on_load(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Sine Sum Project", "", "JSON (*.json)")
        if not fname: 
            logger.info("Project load cancelled by user")
            return
        logger.info(f"Loading project from: {fname}")
        try:
            with open(fname) as f:
                d = json.load(f)
            A = np.array(d.get("Amplitudes", []), float)
            P = np.array(d.get("Phases", []), float)
        except Exception as e:
            logger.error(f"Failed to read project file {fname}: {e}")
            QMessageBox.critical(self, "Error", f"Could not read file:\n{e}"); return
        if len(A) == 0 or len(A) != len(P):
            logger.error(f"Invalid project file format: Amplitudes={len(A)}, Phases={len(P)}")
            QMessageBox.critical(self, "Error", "Invalid JSON: need equal-length Amplitudes & Phases."); return
        logger.info(f"Successfully loaded project with {len(A)} harmonics")
        self._resize_state(len(A))
        self.amplitudes[:] = A; self.phases[:] = P
        self._refresh_combo(); self._sync_controls(); self._refresh_table(); self._redraw()
        QMessageBox.information(self, "Loaded", os.path.basename(fname))

    def on_about(self):
        logger.info("About dialog opened")
        QMessageBox.information(
            self, "About",
            "Sum of Sines — PyQt UI with large plots on top and a split controls area:\n"
            "compact controls on the left, harmonics table on the right."
        )


def main():
    logger.info("Starting Sum of Sines PyQt application")
    app = QApplication(sys.argv)
    w = SineSumWindow()
    w.show()
    logger.info("Application window shown, entering event loop")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
