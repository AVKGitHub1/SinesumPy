# sinesum2.py
# Python port of the MATLAB GUIDE app "sinesum2"
# Dependencies: numpy, matplotlib, sounddevice (optional), scipy (optional for WAV), or wave+struct fallback.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import json
import os
import sys

# ---- Optional audio backends ----
_SD_OK = False
try:
    import sounddevice as sd
    _SD_OK = True
except Exception:
    _SD_OK = False

_SICPY_WAV_OK = False
try:
    from scipy.io import wavfile
    _SICPY_WAV_OK = True
except Exception:
    _SICPY_WAV_OK = False

import wave, struct  # as a last-resort WAV writer


class SineSumGUI:
    def __init__(self):
        # ---- State (equivalent to MATLAB handles.*) ----
        self.num_harmonics = 1
        self.current_harmonic = 1  # 1-indexed like MATLAB
        self.amplitudes = np.zeros(self.num_harmonics, dtype=float)
        self.phases = np.zeros(self.num_harmonics, dtype=float)

        # timebase used in update_display
        self.maxT = 2.0
        self.intervalT = 1/200.0
        self.t = np.arange(0, self.maxT + 1e-12, self.intervalT)

        # audio params
        self.Fs = 44100
        self.num_seconds = 2.0
        self.fundamental_freq = 500.0  # Hz

        # ---- Figure & layout ----
        self.fig = plt.figure("Sum of Sines", figsize=(12, 8))

        # Axes for plots
        self.ax_combined = self.fig.add_axes([0.08, 0.56, 0.55, 0.38])
        self.ax_current = self.fig.add_axes([0.08, 0.12, 0.55, 0.34])
        self.ax_spec = self.fig.add_axes([0.67, 0.44, 0.30, 0.50], projection='3d')

        # “Listbox” area (we’ll render text lines)
        self.ax_list = self.fig.add_axes([0.67, 0.12, 0.30, 0.28])
        self.ax_list.axis("off")

        # ---- Controls (TextBoxes, Sliders, Buttons) ----
        # num harmonics edit + Start Over
        self.tb_num = TextBox(self.fig.add_axes([0.08, 0.96, 0.08, 0.035]),
                              label="Harmonics:", initial=str(self.num_harmonics))
        self.btn_start = Button(self.fig.add_axes([0.17, 0.96, 0.08, 0.035]), "Start Over")
        self.btn_start.on_clicked(self.on_start_over)

        # current harmonic TextBox + prev/next
        self.tb_cur = TextBox(self.fig.add_axes([0.28, 0.96, 0.06, 0.035]),
                              label="Harmonic:", initial=str(self.current_harmonic))
        self.btn_prev = Button(self.fig.add_axes([0.36, 0.96, 0.06, 0.035]), "Prev")
        self.btn_next = Button(self.fig.add_axes([0.43, 0.96, 0.06, 0.035]), "Next")
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)
        self.tb_cur.on_submit(self.on_current_submit)

        # amplitude slider + edit
        # amplitude in [0, 2] by default; you can extend if you like
        self.s_amp = Slider(self.fig.add_axes([0.67, 0.97, 0.30, 0.02]),
                            label="Amplitude", valmin=0.0, valmax=2.0, valinit=0.0)
        self.s_amp.on_changed(self.on_amp_slider)
        self.tb_amp = TextBox(self.fig.add_axes([0.67, 0.935, 0.30, 0.03]),
                              label="Amplitude edit:", initial="0.0")
        self.tb_amp.on_submit(self.on_amp_edit)

        # phase slider + edit (0..2π)
        self.s_phase = Slider(self.fig.add_axes([0.67, 0.89, 0.30, 0.02]),
                              label="Phase (rad)", valmin=0.0, valmax=2*np.pi, valinit=0.0)
        self.s_phase.on_changed(self.on_phase_slider)
        self.tb_phase = TextBox(self.fig.add_axes([0.67, 0.855, 0.30, 0.03]),
                                label="Phase edit:", initial="0.0")
        self.tb_phase.on_submit(self.on_phase_edit)

        # Play, Save, Load, About
        self.btn_play = Button(self.fig.add_axes([0.67, 0.80, 0.09, 0.035]), "Play")
        self.btn_save = Button(self.fig.add_axes([0.77, 0.80, 0.09, 0.035]), "Save")
        self.btn_load = Button(self.fig.add_axes([0.87, 0.80, 0.09, 0.035]), "Load")
        self.btn_about = Button(self.fig.add_axes([0.67, 0.75, 0.29, 0.035]), "About")

        self.btn_play.on_clicked(self.on_play)
        self.btn_save.on_clicked(self.on_save)
        self.btn_load.on_clicked(self.on_load)
        self.btn_about.on_clicked(self.on_about)

        # Initialize
        self.set_current_harmonic(1)
        self.update_display()

    # ---- Helpers equivalent to MATLAB init / set_current_harmonic ----
    def init_with_count(self, new_count: int):
        self.num_harmonics = max(1, int(new_count))
        old_amp = self.amplitudes
        old_ph = self.phases
        self.amplitudes = np.zeros(self.num_harmonics, dtype=float)
        self.phases = np.zeros(self.num_harmonics, dtype=float)
        # preserve old values up to min length
        m = min(len(old_amp), self.num_harmonics)
        self.amplitudes[:m] = old_amp[:m]
        self.phases[:m] = old_ph[:m]

        self.current_harmonic = 1
        self.tb_num.set_val(str(self.num_harmonics))
        self.set_current_harmonic(1)
        self.update_display()

    def set_current_harmonic(self, k: int):
        k = max(1, min(self.num_harmonics, int(k)))
        self.current_harmonic = k
        # push values into sliders and edits
        a = float(self.amplitudes[k-1])
        p = float(self.phases[k-1])
        self.s_amp.set_val(a)
        # set_val also triggers on_changed; mute repaint storms by just allowing it
        self.tb_amp.set_val(f"{a:.6g}")
        self.s_phase.set_val(p)
        self.tb_phase.set_val(f"{p:.6g}")
        self.tb_cur.set_val(str(k))

    # ---- Display update (combined, current, spectral, list) ----
    def update_display(self):
        t = self.t
        # Combined
        x = np.zeros_like(t)
        for n in range(1, self.num_harmonics+1):
            x += self.amplitudes[n-1] * np.sin(2*np.pi*n*t + self.phases[n-1])
        self.ax_combined.cla()
        self.ax_combined.plot(t, x)
        self.ax_combined.grid(True)
        maxX = np.max(np.abs(x))
        maxX = max(maxX, 1e-4, float(self.amplitudes[self.current_harmonic-1]))
        self.ax_combined.set_xlim(0, self.maxT)
        self.ax_combined.set_ylim(-maxX, maxX)
        self.ax_combined.set_title("Combined Signal")
        self.ax_combined.set_xlabel("t [s]")

        # Current harmonic
        k = self.current_harmonic
        y = self.amplitudes[k-1] * np.sin(2*np.pi*k*t + self.phases[k-1])
        self.ax_current.cla()
        self.ax_current.plot(t, y)
        self.ax_current.grid(True)
        self.ax_current.set_xlim(0, self.maxT)
        self.ax_current.set_ylim(-maxX, maxX)
        self.ax_current.set_title(f"Harmonic {k}")
        self.ax_current.set_xlabel("t [s]")

        # Spectral profile (3D “stem”)
        self.ax_spec.cla()
        if self.num_harmonics > 0:
            hs = np.arange(1, self.num_harmonics+1)
            # draw stems
            for i, h in enumerate(hs):
                amp = float(self.amplitudes[i])
                ph = float(self.phases[i])
                # vertical line for “stem”
                self.ax_spec.plot([h, h], [ph, ph], [0.0, amp], lw=2)
                self.ax_spec.scatter([h], [ph], [amp], s=30)
            # highlight current harmonic in a different marker
            hc = self.current_harmonic
            ampC = float(self.amplitudes[hc-1])
            phC = float(self.phases[hc-1])
            self.ax_spec.plot([hc, hc], [phC, phC], [0.0, ampC], lw=3)
            self.ax_spec.scatter([hc], [phC], [ampC], s=60)
        self.ax_spec.set_xlim(0, self.num_harmonics + 1)
        self.ax_spec.set_ylim(0, 2*np.pi)
        self.ax_spec.set_zlim(0, (np.max(self.amplitudes) if self.num_harmonics else 0) + 1.0)
        self.ax_spec.set_title("Spectral Profile")
        self.ax_spec.set_xlabel("Harmonic")
        self.ax_spec.set_ylabel("Phase [rad]")
        self.ax_spec.set_zlabel("Amplitude")

        # Text “listbox”
        self.ax_list.cla()
        self.ax_list.axis("off")
        lines = ["Harmonic    Amplitude      Phase"]
        for n in range(1, self.num_harmonics+1):
            lines.append(f"{n:>3d}          {self.amplitudes[n-1]:>10.6f}   {self.phases[n-1]:>10.6f}")
        txt = "\n".join(lines)
        self.ax_list.text(0.0, 1.0, txt, va="top", family="monospace", fontsize=9)
        # indicate selection
        if self.num_harmonics >= 1:
            # draw a small marker next to current line
            y0 = 1.0 - (self.current_harmonic) * (1.0 / (len(lines)+0.5))
            self.ax_list.text(-0.05, 1.0 - 0.052 * self.current_harmonic, "➤", va="top", fontsize=10)

        self.fig.canvas.draw_idle()

    # ---- Callbacks ----
    def on_start_over(self, _):
        try:
            nh = int(float(self.tb_num.text))
        except Exception:
            nh = 1
        self.init_with_count(nh)

    def on_prev(self, _):
        if self.current_harmonic > 1:
            self.set_current_harmonic(self.current_harmonic - 1)
            self.update_display()

    def on_next(self, _):
        if self.current_harmonic < self.num_harmonics:
            self.set_current_harmonic(self.current_harmonic + 1)
            self.update_display()

    def on_current_submit(self, text):
        try:
            k = int(float(text))
        except Exception:
            k = self.current_harmonic
        self.set_current_harmonic(k)
        self.update_display()

    def on_amp_slider(self, val):
        k = self.current_harmonic
        self.amplitudes[k-1] = float(val)
        self.tb_amp.set_val(f"{float(val):.6g}")
        self.update_display()

    def on_phase_slider(self, val):
        k = self.current_harmonic
        self.phases[k-1] = float(val)
        self.tb_phase.set_val(f"{float(val):.6g}")
        self.update_display()

    def on_amp_edit(self, text):
        try:
            v = float(text)
        except Exception:
            v = float(self.amplitudes[self.current_harmonic-1])
        self.amplitudes[self.current_harmonic-1] = v
        # clamp slider range if needed
        if v < self.s_amp.valmin or v > self.s_amp.valmax:
            vmin = min(self.s_amp.valmin, v)
            vmax = max(self.s_amp.valmax, v)
            self.s_amp.ax.set_xlim(vmin, vmax)
            self.s_amp.valmin = vmin
            self.s_amp.valmax = vmax
        self.s_amp.set_val(v)
        self.update_display()

    def on_phase_edit(self, text):
        try:
            v = float(text)
        except Exception:
            v = float(self.phases[self.current_harmonic-1])
        # wrap to [0, 2π) for consistency with slider (optional)
        if np.isfinite(v):
            v = v % (2*np.pi)
        self.phases[self.current_harmonic-1] = v
        self.s_phase.set_val(v)
        self.update_display()

    def synthesize_audio(self):
        t = np.arange(0, self.num_seconds, 1.0/self.Fs)
        x = np.zeros_like(t)
        for n in range(1, self.num_harmonics+1):
            x += self.amplitudes[n-1] * np.sin(2*np.pi*n*self.fundamental_freq*t + self.phases[n-1])
        # normalize like MATLAB: max(x)+0.05 to avoid clipping silence
        denom = np.max(np.abs(x)) + 0.05
        if denom <= 0:
            denom = 1.0
        x = x / denom
        return x.astype(np.float32)

    def on_play(self, _):
        x = self.synthesize_audio()
        if _SD_OK:
            sd.stop(ignore_errors=True)
            sd.play(x, samplerate=self.Fs, blocking=False)
        else:
            # write a temp WAV and print path
            path = os.path.abspath("sinesum2_output.wav")
            if _SICPY_WAV_OK:
                wavfile.write(path, self.Fs, (x * 32767).astype(np.int16))
            else:
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.Fs)
                    for s in (x * 32767):
                        wf.writeframesraw(struct.pack("<h", int(np.clip(s, -32768, 32767))))
            print(f"[Info] sounddevice not available; wrote WAV to: {path}")

    def on_save(self, _):
        # Save to JSON (amplitudes, phases). You can also use .npz; JSON is human-readable.
        data = {
            "Amplitudes": self.amplitudes.tolist(),
            "Phases": self.phases.tolist(),
        }
        # simple prompt-less save next to script; edit as desired
        fname = "sinesum2_project.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Saved] {fname}")

    def on_load(self, _):
        fname = "sinesum2_project.json"
        if not os.path.exists(fname):
            print(f"[Error] {fname} not found. Place it next to this script.")
            return
        with open(fname, "r") as f:
            data = json.load(f)
        amps = np.array(data.get("Amplitudes", []), dtype=float)
        ph = np.array(data.get("Phases", []), dtype=float)
        if len(amps) == 0 or len(amps) != len(ph):
            print("[Error] Invalid file: amplitudes and phases must exist and have equal length.")
            return
        self.init_with_count(len(amps))
        self.amplitudes[:] = amps
        self.phases[:] = ph
        self.set_current_harmonic(1)
        self.update_display()
        print(f"[Loaded] {fname}")

    def on_about(self, _):
        print("Sum of Sines — Python port of EE261 (2006) demo. Use sliders/edits, Prev/Next, Start Over, Save/Load, Play.")

    def run(self):
        plt.show()


if __name__ == "__main__":
    app = SineSumGUI()
    app.run()
