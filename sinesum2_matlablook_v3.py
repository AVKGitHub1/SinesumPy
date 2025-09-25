# sinesum2_matlablook_v3.py
# MATLAB GUIDE "sinesum2" look-alike (layout tuned)
# Deps: numpy, matplotlib; optional: sounddevice or scipy (for WAV save)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import json, os, wave, struct

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
        if not self.enabled:  # swallow clicks when disabled
            return
        return super()._click(event)


class SineSumApp:
    def __init__(self):
        # state mirrors MATLAB handles.*
        self.num_harmonics = 5
        self.current_harmonic = 1  # 1-indexed
        self.amplitudes = np.zeros(self.num_harmonics)
        self.phases = np.zeros(self.num_harmonics)

        # timebase for plots
        self.maxT = 2.0
        self.dt = 1/200.0
        self.t = np.arange(0, self.maxT + 1e-12, self.dt)

        # audio
        self.Fs = 44100
        self.num_seconds = 2.0
        self.f0 = 500.0

        # figure layout like GUIDE
        self.fig = plt.figure("Sum of Sines", figsize=(14, 8), facecolor=MATLAB_GRAY)

        # --- Top row: Start Over with [ N ] harmonics ---
        self.btn_start = ClassicButton(self.fig.add_axes([0.33, 0.92, 0.12, 0.045]), "Start Over")
        self.ax_label_with = self.fig.add_axes([0.46, 0.92, 0.05, 0.045]); self.ax_label_with.axis("off")
        self.ax_label_with.text(0, 0.5, "with", va="center", fontsize=10)
        self.tb_num = TextBox(self.fig.add_axes([0.52, 0.925, 0.06, 0.035]), "", initial=str(self.num_harmonics))
        self.ax_label_harm = self.fig.add_axes([0.59, 0.92, 0.10, 0.045]); self.ax_label_harm.axis("off")
        self.ax_label_harm.text(0, 0.5, "harmonics", va="center", fontsize=10)

        # --- Adjusting Harmonic Number row (centered) ---
        self.ax_label_adj = self.fig.add_axes([0.33, 0.87, 0.22, 0.04]); self.ax_label_adj.axis("off")
        self.ax_label_adj.text(0, 0.5, "Adjusting Harmonic Number:", va="center", fontsize=10)
        self.btn_prev = ClassicButton(self.fig.add_axes([0.33, 0.83, 0.10, 0.04]), "Previous")
        self.tb_cur = TextBox(self.fig.add_axes([0.44, 0.835, 0.10, 0.035]), "", initial=str(self.current_harmonic))
        self.btn_next = ClassicButton(self.fig.add_axes([0.55, 0.83, 0.10, 0.04]), "Next")

        # --- Sliders row (amplitude left, phase right) ---
        self.ax_label_amp = self.fig.add_axes([0.18, 0.78, 0.18, 0.04]); self.ax_label_amp.axis("off")
        self.ax_label_amp.text(0, 0.5, "Adjust Amplitude:", va="center", fontsize=10)
        self.s_amp = Slider(self.fig.add_axes([0.18, 0.755, 0.30, 0.03]), "", 0.0, 2.0, valinit=0.0)
        # Moved amplitude numeric **right** a bit more than before
        self.tb_amp = TextBox(self.fig.add_axes([0.49, 0.755, 0.09, 0.032]), "", initial="0")

        self.ax_label_ph = self.fig.add_axes([0.60, 0.78, 0.15, 0.04]); self.ax_label_ph.axis("off")
        self.ax_label_ph.text(0, 0.5, "Adjust Phase:", va="center", fontsize=10)
        self.s_phase = Slider(self.fig.add_axes([0.60, 0.755, 0.30, 0.03]), "", 0.0, 2*np.pi, valinit=0.0)
        # Moved phase numeric **right** too
        self.tb_phase = TextBox(self.fig.add_axes([0.91, 0.755, 0.09, 0.032]), "", initial="0")

        # Play button centered under sliders
        self.btn_play = ClassicButton(self.fig.add_axes([0.46, 0.705, 0.12, 0.04]), "Play Sound")

        # --- Left column: bigger table (top) & larger spectral profile (bottom) ---
        # Bigger table: wider (0.34→0.38) and taller (0.20→0.24)
        self.ax_list = self.fig.add_axes([0.08, 0.47, 0.38, 0.24]); self.ax_list.axis("off")

        # Spectral profile: larger panel; make sure labels are inside white panel by adding labelpad
        self.ax_spec = self.fig.add_axes([0.08, 0.09, 0.42, 0.34], projection='3d')

        # --- Right column: Harmonic plot (top) & Combined plot (bottom) ---
        self.ax_current = self.fig.add_axes([0.55, 0.47, 0.40, 0.24])
        self.ax_combined = self.fig.add_axes([0.55, 0.10, 0.40, 0.24])

        # Menu-ish buttons near table
        self.btn_save = ClassicButton(self.fig.add_axes([0.08, 0.705, 0.10, 0.04]), "Save")
        self.btn_load = ClassicButton(self.fig.add_axes([0.19, 0.705, 0.10, 0.04]), "Load")
        self.btn_about = ClassicButton(self.fig.add_axes([0.30, 0.705, 0.10, 0.04]), "About")

        # Wire callbacks
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
        self.btn_save.on_clicked(self.cb_save)
        self.btn_load.on_clicked(self.cb_load)
        self.btn_about.on_clicked(lambda *_: print("Sum of Sines — EE261 look-alike (Python)."))

        # initial UI sync & draw
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()

    # ---------- state helpers ----------
    def _resize(self, new_n):
        new_n = max(1, int(new_n))
        oldA, oldP = self.amplitudes, self.phases
        self.amplitudes = np.zeros(new_n)
        self.phases = np.zeros(new_n)
        m = min(len(oldA), new_n)
        self.amplitudes[:m] = oldA[:m]
        self.phases[:m] = oldP[:m]
        self.num_harmonics = new_n
        self.current_harmonic = 1

    def _sync_nav_enabled(self):
        self.btn_prev.set_enabled(self.current_harmonic > 1)
        self.btn_next.set_enabled(self.current_harmonics_left())

    def current_harmonics_left(self):
        return self.current_harmonic < self.num_harmonics

    def _push_to_controls(self):
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
        maxX = max(1e-4, np.max(np.abs(x)), float(self.amplitudes[self.current_harmonic-1]))
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

        # bigger “listbox”
        self.ax_list.cla(); self.ax_list.axis("off")
        header = f"{'Harmonic':<10}{'Amplitude':>14}{'Phase':>14}"
        lines = [header]
        for n in range(1, self.num_harmonics+1):
            lines.append(f"{n:<10d}{self.amplitudes[n-1]:>14.4f}{self.phases[n-1]:>14.4f}")
        txt = "\n".join(lines)
        self.ax_list.text(0.0, 1.0, txt, va="top", family="monospace", fontsize=10, color='black',
                          bbox=dict(facecolor='white', edgecolor=EDGE_GRAY, boxstyle='square,pad=0.35'))

        # spectral profile (bigger; labels inside panel)
        self.ax_spec.cla()
        self.ax_spec.set_facecolor('white')
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
        zmax = (np.max(self.amplitudes) if self.num_harmonics else 0) + 1.0
        self.ax_spec.set_zlim(0, zmax)
        self.ax_spec.set_title("Spectral Profile", pad=8)
        # labelpad pulls labels inside the panel; slightly bigger tick label size for readability
        self.ax_spec.set_xlabel("Harmonic", labelpad=8)
        self.ax_spec.set_ylabel("Phase", labelpad=10)
        self.ax_spec.set_zlabel("Amplitude", labelpad=4)
        self.ax_spec.tick_params(axis='both', which='major', labelsize=8)
        self.ax_spec.grid(True)

        self.fig.canvas.draw_idle()

    # ---------- callbacks ----------
    def cb_start_over(self, _):
        self._resize(self._parse_int(self.tb_num.text, default=1))
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()

    def cb_prev(self, _):
        if self.current_harmonic > 1:
            self.current_harmonic -= 1
            self._sync_nav_enabled()
            self._push_to_controls()
            self._update_display()

    def cb_next(self, _):
        if self.current_harmonic < self.num_harmonics:
            self.current_harmonic += 1
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
        self.amplitudes[self.current_harmonic-1] = float(val)
        self.tb_amp.set_val(f"{float(val):.6g}")
        self._update_display()

    def cb_phase_slider(self, val):
        self.phases[self.current_harmonic-1] = float(val)
        self.tb_phase.set_val(f"{float(val):.6g}")
        self._update_display()

    def cb_amp_edit(self, text):
        v = self._parse_float(text, default=self.amplitudes[self.current_harmonic-1])
        self.amplitudes[self.current_harmonic-1] = v
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
        t = np.arange(0, self.num_seconds, 1/self.Fs)
        x = np.zeros_like(t)
        for n in range(1, self.num_harmonics+1):
            x += self.amplitudes[n-1] * np.sin(2*np.pi*n*self.f0*t + self.phases[n-1])
        denom = np.max(np.abs(x)) + 0.05
        x = (x/denom).astype(np.float32)
        if _SD_OK:
            sd.stop(ignore_errors=True)
            sd.play(x, samplerate=self.Fs, blocking=False)
        else:
            path = "sinesum2_output.wav"
            if _SCIPY_WAV_OK:
                wavfile.write(path, self.Fs, (x * 32767).astype(np.int16))
            else:
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.Fs)
                    for s in (x * 32767):
                        wf.writeframesraw(struct.pack("<h", int(np.clip(s, -32768, 32767))))
            print(f"[Info] sounddevice not available; wrote {os.path.abspath(path)}")

    def cb_save(self, _):
        data = {"Amplitudes": self.amplitudes.tolist(), "Phases": self.phases.tolist()}
        with open("sinesum2_project.json", "w") as f:
            json.dump(data, f, indent=2)
        print("[Saved] sinesum2_project.json")

    def cb_load(self, _):
        fname = "sinesum2_project.json"
        if not os.path.exists(fname):
            print("[Error] sinesum2_project.json not found.")
            return
        with open(fname) as f:
            d = json.load(f)
        A = np.array(d.get("Amplitudes", []), float)
        P = np.array(d.get("Phases", []), float)
        if len(A) == 0 or len(A) != len(P):
            print("[Error] Invalid file (Amplitudes/Phases).")
            return
        self._resize(len(A))
        self.amplitudes[:] = A
        self.phases[:] = P
        self.current_harmonic = 1
        self._sync_nav_enabled()
        self._push_to_controls()
        self._update_display()
        print("[Loaded] sinesum2_project.json")

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

    def current_harmonics_left(self):
        return self.current_harmonic < self.num_harmonics

    def run(self):
        for ax in [self.ax_current, self.ax_combined]:
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
        plt.show()


if __name__ == "__main__":
    SineSumApp().run()
