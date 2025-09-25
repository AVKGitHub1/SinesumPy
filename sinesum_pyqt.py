# sinesum2_pyqt.py
# PyQt GUI remake of MATLAB "sinesum2" with dropdowns, table, and embedded Matplotlib.
# Works with PyQt6 or PyQt5 (auto-fallback). Audio via sounddevice (optional),
# otherwise a WAV file is written using scipy or the stdlib wave module.

from __future__ import annotations
import json, os, sys, math, wave, struct
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

# ---- Qt bindings (prefer PyQt6; fall back to PyQt5) ----
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
        QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QLineEdit
    )
    from PyQt6.QtCore import Qt
    PYQT6 = True
except Exception:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
        QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QLineEdit
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


def nice_zero_line(ax):
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.grid(True, ls="--", alpha=0.5)


class PlotCanvas(FigureCanvas):
    """Three panels: current harmonic (top-right), combined (bottom-right), spectral profile (left)."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)

        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.0, 1.6], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.3)
        self.ax_spec = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_current = self.fig.add_subplot(gs[0, 1])
        self.ax_combined = self.fig.add_subplot(gs[1, 1])

        self.fig.patch.set_facecolor((0.85, 0.85, 0.85))
        for ax in [self.ax_current, self.ax_combined]:
            ax.set_facecolor("white")
        self.ax_spec.set_facecolor("white")

    def draw_plots(self, t, amplitudes, phases, current_idx):
        N = len(amplitudes)
        k = current_idx  # 0-indexed
        # --- Combined ---
        x = np.zeros_like(t)
        for n in range(N):
            x += amplitudes[n] * np.sin(2*np.pi*(n+1)*t + phases[n])

        self.ax_combined.clear()
        self.ax_combined.plot(t, x, lw=1.2)
        nice_zero_line(self.ax_combined)
        maxX = max(1e-4, np.max(np.abs(x)), float(amplitudes[k]))
        self.ax_combined.set_xlim(0, t[-1])
        self.ax_combined.set_ylim(-maxX, maxX)
        self.ax_combined.set_title("Combined Signal")
        self.ax_combined.set_facecolor("white")

        # --- Current harmonic ---
        self.ax_current.clear()
        y = amplitudes[k] * np.sin(2*np.pi*(k+1)*t + phases[k])
        self.ax_current.plot(t, y, lw=1.2)
        nice_zero_line(self.ax_current)
        self.ax_current.set_xlim(0, t[-1])
        self.ax_current.set_ylim(-maxX, maxX)
        self.ax_current.set_title(f"Harmonic {k+1}")
        self.ax_current.set_facecolor("white")

        # --- Spectral profile (3D stems) ---
        self.ax_spec.clear()
        self.ax_spec.set_title("Spectral Profile", pad=10)
        self.ax_spec.set_xlabel("Harmonic", labelpad=8)
        self.ax_spec.set_ylabel("Phase", labelpad=10)
        self.ax_spec.set_zlabel("Amplitude", labelpad=6)
        self.ax_spec.set_facecolor("white")
        self.ax_spec.grid(True)

        if N > 0:
            for i in range(N):
                h = i + 1
                a = float(amplitudes[i])
                ph = float(phases[i] % (2*np.pi))
                self.ax_spec.plot([h, h], [ph, ph], [0, a], lw=2)
                self.ax_spec.scatter([h], [ph], [a], s=30)
            hc = k + 1
            ac = float(amplitudes[k]); pc = float(phases[k] % (2*np.pi))
            self.ax_spec.plot([hc, hc], [pc, pc], [0, ac], lw=3)
            self.ax_spec.scatter([hc], [pc], [ac], s=60)

        self.ax_spec.set_xlim(0, N + 1)
        self.ax_spec.set_ylim(0, 2*np.pi)
        zmax = (np.max(amplitudes) if N else 0) + 1.0
        self.ax_spec.set_zlim(0, zmax)

        self.draw_idle()


class SineSumWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sum of Sines (PyQt)")
        self.resize(1200, 750)

        # --- state (mirrors MATLAB handles.*) ---
        self.num_harmonics = 5
        self.current_harmonic = 0   # 0-indexed for Python
        self.amplitudes = np.zeros(self.num_harmonics, dtype=float)
        self.phases = np.zeros(self.num_harmonics, dtype=float)

        self.maxT = 2.0
        self.dt = 1/200.0
        self.t = np.arange(0, self.maxT + 1e-12, self.dt)

        self.Fs = 44100
        self.num_seconds = 2.0
        self.f0 = 500.0

        # ---- central widget + layout scaffold ----
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # === Row 1: Start Over / num harmonics ===
        row1 = QHBoxLayout()
        row1.addStretch(1)
        self.btn_start = QPushButton("Start Over")
        self.spin_N = QSpinBox()
        self.spin_N.setRange(1, 999)
        self.spin_N.setValue(self.num_harmonics)
        lbl_with = QLabel("with"); lbl_harm = QLabel("harmonics")
        row1.addWidget(self.btn_start)
        row1.addSpacing(8)
        row1.addWidget(lbl_with)
        row1.addWidget(self.spin_N)
        row1.addWidget(lbl_harm)
        row1.addStretch(1)
        root.addLayout(row1)

        # === Row 2: Adjusting Harmonic Number (Prev / dropdown / Next) ===
        row2 = QHBoxLayout()
        row2.addStretch(1)
        row2.addWidget(QLabel("Adjusting Harmonic Number:"))
        row2.addSpacing(12)
        self.btn_prev = QPushButton("Previous")
        self.combo_k = QComboBox()
        self.btn_next = QPushButton("Next")
        row2.addWidget(self.btn_prev)
        row2.addWidget(self.combo_k)
        row2.addWidget(self.btn_next)
        row2.addStretch(1)
        root.addLayout(row2)

        # fill combo
        self._refresh_combo_items()

        # === Row 3: Amplitude & Phase controls ===
        row3 = QGridLayout(); root.addLayout(row3)
        # Amplitude
        row3.addWidget(QLabel("Adjust Amplitude:"), 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        self.slider_amp = QSlider(Qt.Orientation.Horizontal)
        self.slider_amp.setRange(0, 200)  # 0..2.0
        self.slider_amp.setValue(0)
        self.edit_amp = QLineEdit("0.0")
        self.edit_amp.setFixedWidth(100)
        row3.addWidget(self.slider_amp, 1, 0, 1, 2)
        row3.addWidget(self.edit_amp, 1, 2)

        # Phase
        row3.addWidget(QLabel("Adjust Phase:"), 0, 3, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        self.slider_phase = QSlider(Qt.Orientation.Horizontal)
        self.slider_phase.setRange(0, 6283)  # interpret as milli-radians (0..2π*1000)
        self.slider_phase.setValue(0)
        self.edit_phase = QLineEdit("0.0")
        self.edit_phase.setFixedWidth(100)
        row3.addWidget(self.slider_phase, 1, 3, 1, 2)
        row3.addWidget(self.edit_phase, 1, 5)

        # === Row 4: Action buttons ===
        row4 = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_load = QPushButton("Load")
        self.btn_about = QPushButton("About")
        self.btn_play = QPushButton("Play Sound")
        row4.addWidget(self.btn_save)
        row4.addWidget(self.btn_load)
        row4.addWidget(self.btn_about)
        row4.addStretch(1)
        row4.addWidget(self.btn_play)
        root.addLayout(row4)

        # === Row 5+: Left (table + spectral) & Right (plots) ===
        main_split = QHBoxLayout(); root.addLayout(main_split, 1)

        left_col = QVBoxLayout(); main_split.addLayout(left_col, 1)
        # Table (bigger)
        self.table = QTableWidget(self.num_harmonics, 3)
        self.table.setHorizontalHeaderLabels(["Harmonic", "Amplitude", "Phase"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(180)
        self._refresh_table()
        left_col.addWidget(self.table)

        # Spectral profile + plots on the right
        self.canvas = PlotCanvas(self)
        main_split.addWidget(self.canvas, 2)

        # Spectral profile is already part of canvas on the left of the grid

        # ---- wiring signals ----
        self.btn_start.clicked.connect(self.on_start_over)
        self.spin_N.valueChanged.connect(self.on_change_N)

        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)
        self.combo_k.currentIndexChanged.connect(self.on_combo_changed)
        self.table.cellClicked.connect(self.on_table_row_clicked)

        self.slider_amp.valueChanged.connect(self.on_slider_amp)
        self.slider_phase.valueChanged.connect(self.on_slider_phase)
        self.edit_amp.editingFinished.connect(self.on_edit_amp_done)
        self.edit_phase.editingFinished.connect(self.on_edit_phase_done)

        self.btn_play.clicked.connect(self.on_play)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_load.clicked.connect(self.on_load)
        self.btn_about.clicked.connect(self.on_about)

        # initial draw
        self._sync_controls_from_state()
        self._redraw()

    # ----------------- helpers -----------------
    def _refresh_combo_items(self):
        self.combo_k.blockSignals(True)
        self.combo_k.clear()
        for n in range(1, self.num_harmonics + 1):
            self.combo_k.addItem(str(n))
        self.combo_k.setCurrentIndex(self.current_harmonic)
        self.combo_k.blockSignals(False)

    def _refresh_table(self):
        self.table.setRowCount(self.num_harmonics)
        for r in range(self.num_harmonics):
            self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.table.setItem(r, 1, QTableWidgetItem(f"{self.amplitudes[r]:.4f}"))
            self.table.setItem(r, 2, QTableWidgetItem(f"{self.phases[r]:.4f}"))
        self.table.resizeColumnsToContents()

    def _sync_controls_from_state(self):
        # enable/disable prev/next
        self.btn_prev.setEnabled(self.current_harmonic > 0)
        self.btn_next.setEnabled(self.current_harmonic < self.num_harmonics - 1)

        self.spin_N.blockSignals(True)
        self.spin_N.setValue(self.num_harmonics)
        self.spin_N.blockSignals(False)

        self.combo_k.blockSignals(True)
        self.combo_k.setCurrentIndex(self.current_harmonic)
        self.combo_k.blockSignals(False)

        # amplitude slider 0..200 -> 0..2.0
        a = float(self.amplitudes[self.current_harmonic])
        p = float(self.phases[self.current_harmonic])
        self.slider_amp.blockSignals(True)
        self.slider_amp.setValue(int(round(a * 100)))
        self.slider_amp.blockSignals(False)
        self.edit_amp.setText(f"{a:.6g}")

        self.slider_phase.blockSignals(True)
        self.slider_phase.setValue(int(round((p % (2*math.pi)) * 1000)))
        self.slider_phase.blockSignals(False)
        self.edit_phase.setText(f"{p:.6g}")

        # highlight row in table
        self.table.selectRow(self.current_harmonic)

    def _redraw(self):
        self.canvas.draw_plots(self.t, self.amplitudes, self.phases, self.current_harmonic)
        self._refresh_table()

    def _resize_state(self, newN: int):
        newN = max(1, int(newN))
        oldA, oldP = self.amplitudes, self.phases
        self.amplitudes = np.zeros(newN, dtype=float)
        self.phases = np.zeros(newN, dtype=float)
        m = min(len(oldA), newN)
        self.amplitudes[:m] = oldA[:m]
        self.phases[:m] = oldP[:m]
        self.num_harmonics = newN
        self.current_harmonic = 0

    # ----------------- slots -----------------
    def on_change_N(self, val: int):
        self._resize_state(val)
        self._refresh_combo_items()
        self._sync_controls_from_state()
        self._redraw()

    def on_start_over(self):
        self._resize_state(self.spin_N.value())
        self._refresh_combo_items()
        self._sync_controls_from_state()
        self._redraw()

    def on_prev(self):
        if self.current_harmonic > 0:
            self.current_harmonic -= 1
            self._sync_controls_from_state()
            self._redraw()

    def on_next(self):
        if self.current_harmonic < self.num_harmonics - 1:
            self.current_harmonic += 1
            self._sync_controls_from_state()
            self._redraw()

    def on_combo_changed(self, idx: int):
        if idx < 0: 
            return
        self.current_harmonic = idx
        self._sync_controls_from_state()
        self._redraw()

    def on_table_row_clicked(self, row: int, col: int):
        self.current_harmonic = row
        self._sync_controls_from_state()
        self._redraw()

    def on_slider_amp(self, value: int):
        a = value / 100.0
        self.amplitudes[self.current_harmonic] = a
        self.edit_amp.setText(f"{a:.6g}")
        self._redraw()

    def on_slider_phase(self, value: int):
        p = value / 1000.0  # radians
        self.phases[self.current_harmonic] = p
        self.edit_phase.setText(f"{p:.6g}")
        self._redraw()

    def on_edit_amp_done(self):
        try:
            a = float(self.edit_amp.text())
        except Exception:
            a = self.amplitudes[self.current_harmonic]
        self.amplitudes[self.current_harmonic] = a
        self.slider_amp.setValue(int(round(a * 100)))
        self._redraw()

    def on_edit_phase_done(self):
        try:
            p = float(self.edit_phase.text())
        except Exception:
            p = self.phases[self.current_harmonic]
        self.phases[self.current_harmonic] = p
        self.slider_phase.setValue(int(round((p % (2*math.pi)) * 1000)))
        self._redraw()

    def on_play(self):
        t = np.arange(0, self.num_seconds, 1/self.Fs)
        x = np.zeros_like(t)
        for n in range(self.num_harmonics):
            x += self.amplitudes[n] * np.sin(2*np.pi*(n+1)*self.f0*t + self.phases[n])
        denom = np.max(np.abs(x)) + 0.05
        x = (x / denom).astype(np.float32)

        if _SD_OK:
            sd.stop(ignore_errors=True)
            sd.play(x, samplerate=self.Fs, blocking=False)
        else:
            path = os.path.join(os.getcwd(), "sinesum2_output.wav")
            if _SCIPY_WAV_OK:
                wavfile.write(path, self.Fs, (x * 32767).astype(np.int16))
            else:
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.Fs)
                    for s in (x * 32767):
                        wf.writeframesraw(struct.pack("<h", int(np.clip(s, -32768, 32767))))
            QMessageBox.information(self, "Audio", f"Saved WAV to:\n{path}")

    def on_save(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Sine Sum Project",
                                               "sinesum2_project.json", "JSON (*.json)")
        if not fname:
            return
        data = {"Amplitudes": self.amplitudes.tolist(), "Phases": self.phases.tolist()}
        try:
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Saved", os.path.basename(fname))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file:\n{e}")

    def on_load(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Sine Sum Project",
                                               "", "JSON (*.json)")
        if not fname:
            return
        try:
            with open(fname) as f:
                d = json.load(f)
            A = np.array(d.get("Amplitudes", []), float)
            P = np.array(d.get("Phases", []), float)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file:\n{e}")
            return

        if len(A) == 0 or len(A) != len(P):
            QMessageBox.critical(self, "Error", "Invalid JSON: need equal-length Amplitudes & Phases.")
            return

        self._resize_state(len(A))
        self.amplitudes[:] = A
        self.phases[:] = P
        self._refresh_combo_items()
        self._sync_controls_from_state()
        self._redraw()
        QMessageBox.information(self, "Loaded", os.path.basename(fname))

    def on_about(self):
        QMessageBox.information(self, "About",
            "Sum of Sines — PyQt port with dropdowns, table, and embedded Matplotlib.\n"
            "Based on the EE261 'sinesum2' concept. "
            "Audio via sounddevice (if available) or WAV export."
        )


def main():
    app = QApplication(sys.argv)
    w = SineSumWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
