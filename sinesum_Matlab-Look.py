# sinesum2_matlablook.py
# MATLAB GUIDE "sinesum2" look-alike in Python (matplotlib UI)
# Requires: numpy, matplotlib, tk; optional: sounddevice (for live audio) or scipy (to write WAV)
# Execute: pip install numpy matplotlib tk sounddevice scipy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import json, os, wave, struct, logging, sys

# Optional audio backends
_SD_OK = False
try:
    import sounddevice as sd
    _SD_OK = True
except Exception:
    _SD_OK = False

try:
    from scipy.io import wavfile
    _SCIPY_WAV_OK = True
except Exception:
    _SCIPY_WAV_OK = False

# ---- Logger setup ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

MATLAB_GRAY = (0.85, 0.85, 0.85)
EDGE_GRAY = (0.4, 0.4, 0.4)

class ClassicButton(Button):
    """Button with 'disabled' support & GUIDE-ish style."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = True
        for spine in self.ax.spines.values():
            spine.set_color(EDGE_GRAY)
        self.ax.set_facecolor((0.93, 0.93, 0.93))
        self.ax.figure.canvas.draw_idle()

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)
        self.ax.patch.set_alpha(1.0 if self.enabled else 0.4)
        self.label.set_alpha(1.0 if self.enabled else 0.4)
        self.ax.figure.canvas.draw_idle()

    def _click(self, event):
        # swallow clicks when disabled
        if not self.enabled:
            return
        return super()._click(event)


class SineSumApp:
    def __init__(self):
        logger.info("Initializing SineSumApp")
        # state mirrors MATLAB handles.*
        self.num_harmonics = 1
        self.current_harmonic = 1  # 1-indexed
        self.amplitudes = np.zeros(self.num_harmonics)
        self.phases = np.zeros(self.num_harmonics)
        logger.debug(f"Initial state: {self.num_harmonics} harmonics, current harmonic: {self.current_harmonic}")

        # timebase for plots
        self.maxT = 2.0
        self.dt = 1/200.0
        self.t = np.arange(0, self.maxT + 1e-12, self.dt)

        # audio
        self.Fs = 44100
        self.num_seconds = 2.0
        self.f0 = 500.0

        # figure layout like screenshot
        self.fig = plt.figure("Sum of Sines", figsize=(12, 8), facecolor=MATLAB_GRAY)

        # --- Top row: Start Over with [ N ] harmonics ---
        self.btn_start = ClassicButton(self.fig.add_axes([0.31, 0.92, 0.12, 0.045]), "Start Over")
        self.ax_label_with = self.fig.add_axes([0.44, 0.92, 0.05, 0.045]); self.ax_label_with.axis("off")
        self.ax_label_with.text(0, 0.5, "with", va="center", fontsize=10)
        self.tb_num = TextBox(self.fig.add_axes([0.50, 0.925, 0.06, 0.035]), "", initial=str(self.num_harmonics))
        self.ax_label_harm = self.fig.add_axes([0.57, 0.92, 0.10, 0.045]); self.ax_label_harm.axis("off")
        self.ax_label_harm.text(0, 0.5, "harmonics", va="center", fontsize=10)

        # --- Adjusting Harmonic Number row (centered) ---
        self.ax_label_adj = self.fig.add_axes([0.31, 0.87, 0.22, 0.04]); self.ax_label_adj.axis("off")
        self.ax_label_adj.text(0, 0.5, "Adjusting Harmonic Number:", va="center", fontsize=10)
        self.btn_prev = ClassicButton(self.fig.add_axes([0.31, 0.83, 0.10, 0.04]), "Previous")
        self.tb_cur = TextBox(self.fig.add_axes([0.42, 0.835, 0.10, 0.035]), "", initial=str(self.current_harmonic))
        self.btn_next = ClassicButton(self.fig.add_axes([0.53, 0.83, 0.10, 0.04]), "Next")

        # --- Sliders row (amplitude left, phase right) ---
        self.ax_label_amp = self.fig.add_axes([0.20, 0.78, 0.18, 0.04]); self.ax_label_amp.axis("off")
        self.ax_label_amp.text(0, 0.5, "Adjust Amplitude:", va="center", fontsize=10)
        self.s_amp = Slider(self.fig.add_axes([0.20, 0.755, 0.26, 0.03]), "", 0.0, 2.0, valinit=0.0)
        self.tb_amp = TextBox(self.fig.add_axes([0.50, 0.755, 0.08, 0.032]), "", initial="0")

        self.ax_label_ph = self.fig.add_axes([0.60, 0.78, 0.15, 0.04]); self.ax_label_ph.axis("off")
        self.ax_label_ph.text(0, 0.5, "Adjust Phase:", va="center", fontsize=10)
        self.s_phase = Slider(self.fig.add_axes([0.60, 0.755, 0.26, 0.03]), "", 0.0, 2*np.pi, valinit=0.0)
        self.tb_phase = TextBox(self.fig.add_axes([0.9, 0.755, 0.08, 0.032]), "", initial="0")

        # Play button centered under sliders
        

        # --- Left column: listbox (top) & spectral profile (bottom) ---
        self.ax_list = self.fig.add_axes([0.10, 0.49, 0.30, 0.20]); self.ax_list.axis("off")
        self.ax_spec = self.fig.add_axes([0.01, 0.03, 0.5, 0.4], projection='3d')

        # --- Right column: Harmonic plot (top) & Combined plot (bottom) ---
        self.ax_current = self.fig.add_axes([0.53, 0.48, 0.43, 0.2])
        self.ax_combined = self.fig.add_axes([0.53, 0.03, 0.43, 0.35])

        # Menu-ish buttons (Save/Load/About) tucked near listbox
        self.btn_save = ClassicButton(self.fig.add_axes([0.10, 0.705, 0.09, 0.04]), "Save")
        self.btn_load = ClassicButton(self.fig.add_axes([0.20, 0.705, 0.09, 0.04]), "Load")
        self.btn_play = ClassicButton(self.fig.add_axes([0.30, 0.705, 0.09, 0.04]), "Play Sound")
        self.btn_save_sound = ClassicButton(self.fig.add_axes([0.40, 0.705, 0.09, 0.04]), "Save Sound")
        self.btn_about = ClassicButton(self.fig.add_axes([0.50, 0.705, 0.09, 0.04]), "About")

        # Wiring callbacks
        self.btn_start.on_clicked(self.cb_start_over)
        self.btn_prev.on_clicked(self.cb_prev)
        self.btn_next.on_clicked(self.cb_next)
        self.tb_num.on_submit(self.cb_num_submit)
        self.tb_cur.on_submit(self.cb_cur_submit)

        self.s_amp.on_changed(self.cb_amp_slider)
        self.s_phase.on_changed(self.cb_phase_slider)
        self.tb_amp.on_submit(self.cb_amp_edit)
        self.tb_phase.on_submit(self.cb_phase_edit)

        self.btn_play.on_clicked(self.cb_play)
        self.btn_save_sound.on_clicked(self.cb_save_sound)
        self.btn_save.on_clicked(self.cb_save)
        self.btn_load.on_clicked(self.cb_load)
        self.btn_about.on_clicked(self.cb_about)

        # initial UI sync & draw
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()

    # ---------- state helpers ----------
    def _resize(self, new_n):
        new_n = max(1, int(new_n))
        old_num = self.num_harmonics
        oldA, oldP = self.amplitudes, self.phases
        self.amplitudes = np.zeros(new_n)
        self.phases = np.zeros(new_n)
        m = min(len(oldA), new_n)
        self.amplitudes[:m] = oldA[:m]
        self.phases[:m] = oldP[:m]
        self.num_harmonics = new_n
        self.current_harmonic = 1
        logger.debug(f"Resized state from {old_num} to {new_n} harmonics, preserved {m} values")

    def _sync_nav_enabled(self):
        self.btn_prev.set_enabled(self.current_harmonic > 1)
        self.btn_next.set_enabled(self.current_harmonic < self.num_harmonics)

    def _push_to_controls(self):
        # write current harmonic/a/p to widgets (don’t wrap)
        k = self.current_harmonic - 1
        self.tb_num.set_val(str(self.num_harmonics))
        self.tb_cur.set_val(str(self.current_harmonic))
        self.s_amp.set_val(float(self.amplitudes[k]))
        self.tb_amp.set_val(f"{self.amplitudes[k]:.6g}")
        self.s_phase.set_val(float(self.phases[k] % (2*np.pi)))
        self.tb_phase.set_val(f"{self.phases[k]:.6g}")

    # ---------- plotting ----------
    def _update_display(self):
        t = self.t
        # combined
        x = np.zeros_like(t)
        for n in range(1, self.num_harmonics+1):
            x += self.amplitudes[n-1] * np.sin(2*np.pi*n*t + self.phases[n-1])
        self.ax_combined.cla()
        self.ax_combined.plot(t, x, lw=1.2)
        self.ax_combined.axhline(0, ls="--", lw=0.8, color='k')
        self.ax_combined.grid(True, ls="--", alpha=0.5)
        maxX = max(1e-4, np.max(np.abs(x)), float(self.amplitudes[self.current_harmonic-1])*1.1)
        self.ax_combined.set_xlim(0, self.maxT)
        self.ax_combined.set_ylim(-maxX, maxX)
        self.ax_combined.set_title("Combined Signal")
        self.ax_combined.set_facecolor('white')

        # current harmonic
        k = self.current_harmonic
        y = self.amplitudes[k-1] * np.sin(2*np.pi*k*t + self.phases[k-1])
        self.ax_current.cla()
        self.ax_current.plot(t, y, lw=1.2)
        self.ax_current.axhline(0, ls="--", lw=0.8, color='k')
        self.ax_current.grid(True, ls="--", alpha=0.5)
        self.ax_current.set_xlim(0, self.maxT)
        self.ax_current.set_ylim(-maxX, maxX)
        self.ax_current.set_title(f"Harmonic {k}")
        self.ax_current.set_facecolor('white')

        # “listbox”
        self.ax_list.cla(); self.ax_list.axis("off")
        header = f"{'Harmonic':<10}{'Amplitude':>12}{'Phase':>12}"
        lines = [header]
        for n in range(1, self.num_harmonics+1):
            lines.append(f"{n:<10d}{self.amplitudes[n-1]:>12.4f}{self.phases[n-1]:>12.4f}")
        txt = "\n".join(lines)
        self.ax_list.text(0.0, 1.0, txt, va="top", family="monospace", fontsize=9, color='k',
                          bbox=dict(facecolor='white', edgecolor=EDGE_GRAY, boxstyle='square,pad=0.3'))

        # spectral profile (3D stem)
        self.ax_spec.cla()
        self.ax_spec.set_facecolor(MATLAB_GRAY)
        if self.num_harmonics > 0:
            for i in range(self.num_harmonics):
                h = i + 1
                a = float(self.amplitudes[i])
                ph = float(self.phases[i] % (2*np.pi))
                self.ax_spec.plot([h, h], [ph, ph], [0, a], lw=2)
                self.ax_spec.scatter([h], [ph], [a], s=25)
            hc = self.current_harmonic
            ac = float(self.amplitudes[hc-1])
            pc = float(self.phases[hc-1] % (2*np.pi))
            self.ax_spec.plot([hc, hc], [pc, pc], [0, ac], lw=3)
            self.ax_spec.scatter([hc], [pc], [ac], s=40)
        self.ax_spec.set_xlim(0, self.num_harmonics + 1)
        self.ax_spec.set_ylim(0, 2*np.pi)
        zmax = (np.max(self.amplitudes) if self.num_harmonics else 0)*1.1
        self.ax_spec.set_zlim(0, zmax)
        self.ax_spec.set_title("Spectral Profile", pad=10)
        self.ax_spec.set_xlabel("Harmonic")
        self.ax_spec.set_ylabel("Phase")
        self.ax_spec.set_zlabel("Amplitude")
        self.ax_spec.grid(True)

        self.fig.canvas.draw_idle()

    # ---------- callbacks ----------
    def cb_start_over(self, _):
        new_harmonics = self._parse_int(self.tb_num.text, default=1)
        logger.info(f"Starting over with {new_harmonics} harmonics")
        self._resize(new_harmonics)
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()

    def cb_prev(self, _):
        if self.current_harmonic > 1:
            self.current_harmonic -= 1
            logger.debug(f"Navigation: Previous harmonic selected ({self.current_harmonic})")
            self._sync_nav_enabled()
            self._push_to_controls()
            self._update_display()

    def cb_next(self, _):
        if self.current_harmonic < self.num_harmonics:
            self.current_harmonic += 1
            logger.debug(f"Navigation: Next harmonic selected ({self.current_harmonic})")
            self._sync_nav_enabled()
            self._push_to_controls()
            self._update_display()

    def cb_num_submit(self, text):
        self._resize(self._parse_int(text, default=self.num_harmonics))
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()

    def cb_cur_submit(self, text):
        k = np.clip(self._parse_int(text, default=self.current_harmonic), 1, self.num_harmonics)
        self.current_harmonic = int(k)
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()

    def cb_amp_slider(self, val):
        old_val = self.amplitudes[self.current_harmonic-1]
        self.amplitudes[self.current_harmonic-1] = float(val)
        self.tb_amp.set_val(f"{float(val):.6g}")
        logger.debug(f"Amplitude slider: Harmonic {self.current_harmonic} changed from {old_val:.4f} to {val:.4f}")
        self._update_display()

    def cb_phase_slider(self, val):
        old_val = self.phases[self.current_harmonic-1]
        self.phases[self.current_harmonic-1] = float(val)
        self.tb_phase.set_val(f"{float(val):.6g}")
        logger.debug(f"Phase slider: Harmonic {self.current_harmonic} changed from {old_val:.4f} to {val:.4f}")
        self._update_display()

    def cb_amp_edit(self, text):
        v = self._parse_float(text, default=self.amplitudes[self.current_harmonic-1])
        self.amplitudes[self.current_harmonic-1] = v
        # expand slider range if needed
        if v < self.s_amp.valmin or v > self.s_amp.valmax:
            vmin, vmax = min(self.s_amp.valmin, v), max(self.s_amp.valmax, v)
            self.s_amp.ax.set_xlim(vmin, vmax)
            self.s_amp.valmin, self.s_amp.valmax = vmin, vmax
        self.s_amp.set_val(v)
        self._update_display()

    def cb_phase_edit(self, text):
        v = self._parse_float(text, default=self.phases[self.current_harmonic-1])
        self.phases[self.current_harmonic-1] = v
        self.s_phase.set_val(v % (2*np.pi))
        self._update_display()

    def cb_play(self, _):
        logger.info(f"Playing audio with {self.num_harmonics} harmonics, f0={self.f0}Hz")
        t = np.arange(0, self.num_seconds, 1/self.Fs)
        x = np.zeros_like(t)
        for n in range(1, self.num_harmonics+1):
            x += self.amplitudes[n-1] * np.sin(2*np.pi*n*self.f0*t + self.phases[n-1])
        denom = np.max(np.abs(x)) + 0.05
        x = (x/denom).astype(np.float32)
        logger.debug(f"Generated audio signal: {len(x)} samples, max amplitude: {np.max(np.abs(x)):.4f}")

        if _SD_OK:
            sd.stop(ignore_errors=True)
            sd.play(x, samplerate=self.Fs, blocking=False)
            logger.info(f"Playing audio via sounddevice: {self.num_seconds}s at {self.Fs}Hz")
        else:
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                messagebox.showinfo(
                    "Warning",
                    "sounddevice not available, cannot play audio.\nInstall using 'pip install sounddevice'"
                )
                root.destroy()
            except ImportError:
                logger.warning("tkinter not available, cannot show message box")
            logger.warning("sounddevice not available, cannot play audio. install using 'pip install sounddevice'")
    
    def cb_save_sound(self, _):
        # Import tkinter for file dialog
        TK_Installed = False
        try:
            import tkinter as tk
            TK_Installed = True
            from tkinter import filedialog
            
            # Create a temporary root window (hidden)
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Get the directory of the current Python file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Open file dialog to save JSON file
            fname = filedialog.asksaveasfilename(
                title="Save Sine Sum Project",
                initialdir=script_dir,
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                initialfile="sinesum2_sound.wav"
            )
            
            # Destroy the temporary root window
            root.destroy()
            
            # If no file was selected, return
            if not fname:
                logger.info("Save cancelled by user")
                return
                
        except ImportError:
            # Fallback to hardcoded filename if tkinter not available
            fname = "sinesum2_sound.wav"
            logger.warning("tkinter not available, using default filename")
        
        path = fname
        if _SCIPY_WAV_OK:
            wavfile.write(path, self.Fs, (x * 32767).astype(np.int16))
            logger.info(f"Wrote WAV file using scipy: {os.path.abspath(path)}")
        else:
            if TK_Installed:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()  # Hide the main window
                messagebox.showinfo(
                    "Warning",
                    "scipy not available"
                )
                root.destroy()
                logger.warning("scipy not available, cannot save WAV file. install using 'pip install scipy'")
            else:
                logger.warning("scipy not available, cannot save WAV file. install using 'pip install scipy'")
            

    def cb_save(self, _):
        # Import tkinter for file dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Create a temporary root window (hidden)
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Get the directory of the current Python file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Open file dialog to save JSON file
            fname = filedialog.asksaveasfilename(
                title="Save Sine Sum Project",
                initialdir=script_dir,
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile="sinesum2_project.json"
            )
            
            # Destroy the temporary root window
            root.destroy()
            
            # If no file was selected, return
            if not fname:
                logger.info("Save cancelled by user")
                return
                
        except ImportError:
            # Fallback to hardcoded filename if tkinter not available
            fname = "sinesum2_project.json"
            logger.warning("tkinter not available, using default filename")
        
        logger.info(f"Saving project to: {fname}")
        try:
            data = {"Amplitudes": self.amplitudes.tolist(), "Phases": self.phases.tolist()}
            with open(fname, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully saved project: {os.path.basename(fname)} with {self.num_harmonics} harmonics")
        except Exception as e:
            logger.error(f"Could not save file {fname}: {e}")

    def cb_load(self, _):
        # Import tkinter for file dialog
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Create a temporary root window (hidden)
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Get the directory of the current Python file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Open file dialog to select JSON file
            fname = filedialog.askopenfilename(
                title="Load Sine Sum Project",
                initialdir=script_dir,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                defaultextension=".json"
            )
            
            # Destroy the temporary root window
            root.destroy()
            
            # If no file was selected, return
            if not fname:
                logger.info("Load cancelled by user")
                return
                
        except ImportError:
            # Fallback to hardcoded filename if tkinter not available
            fname = "sinesum2_project.json"
            logger.warning("tkinter not available, using default filename")
        
        if not os.path.exists(fname):
            logger.error(f"File {fname} not found")
            return
            
        logger.info(f"Loading project from: {fname}")
        try:
            with open(fname) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {fname}")
            return
        except Exception as e:
            logger.error(f"Could not read file {fname}: {e}")
            return
            
        A = np.array(d.get("Amplitudes", []), float)
        P = np.array(d.get("Phases", []), float)
        if len(A) == 0 or len(A) != len(P):
            logger.error(f"Invalid file format: Amplitudes={len(A)}, Phases={len(P)}")
            return
        logger.info(f"Successfully loaded project with {len(A)} harmonics")
        self._resize(len(A))
        self.amplitudes[:] = A
        self.phases[:] = P
        self.current_harmonic = 1
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()

    def cb_about(self, _):
        # Show about dialog
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            # Create a temporary root window (hidden)
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Show info message box
            messagebox.showinfo(
                "About",
                "Matlab code to visualize sum of harmonics adapted to python by Abhishek Karve. \n" \
                "https://github.com/AVKGitHub1/SinesumPy"
            )
            
            # Destroy the temporary root window
            root.destroy()
            
        except ImportError:
            # Fallback to console print if tkinter not available
            logger.info("About dialog opened (tkinter not available): Matlab code to generate harmonics adapted to python by Abhishek.")

    # ---------- utils ----------
    @staticmethod
    def _parse_int(s, default=1):
        try:
            return int(float(str(s).strip()))
        except Exception:
            return default

    @staticmethod
    def _parse_float(s, default=0.0):
        try:
            return float(str(s).strip())
        except Exception:
            return default

    def run(self):
        # GUIDE-like dashed grid aesthetic + thin axes
        for ax in [self.ax_current, self.ax_combined]:
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
        plt.show()


if __name__ == "__main__":
    logger.info("Starting Sum of Sines matplotlib application")
    app = SineSumApp()
    logger.info("Application window created, starting GUI")
    app.run()
