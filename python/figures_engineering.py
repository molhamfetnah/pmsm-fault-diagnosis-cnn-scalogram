"""Generate the engineering figure pack for the report (PMSM, control, faults,
detection, motor comparison). Pure matplotlib — reproducible, no external assets.

Run:  .venv/bin/python -m python.figures_engineering
Output: docs/figures/*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle, FancyArrowPatch, FancyBboxPatch, Arc

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
BLUE, RED, GREEN, ORANGE, PURPLE, GREY = "#2c6fbb", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#7f8c8d"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# 1) PMSM construction cross-section --------------------------------------- #
def pmsm_construction():
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.add_patch(Circle((0, 0), 1.30, fc="#dfe6ee", ec="#34495e", lw=2))   # stator yoke
    ax.add_patch(Circle((0, 0), 1.00, fc="white", ec="#34495e", lw=1.2))   # stator bore
    # stator slots / 3-phase windings (12 slots, colored A/B/C)
    phase_c = [BLUE, RED, GREEN]
    for k in range(12):
        a = np.deg2rad(k * 30)
        ax.add_patch(Circle((1.13 * np.cos(a), 1.13 * np.sin(a)), 0.085,
                            fc=phase_c[k % 3], ec="k", lw=0.4))
    ax.add_patch(Circle((0, 0), 0.78, fc="#fceabb", ec="#b9770e", lw=1.5))  # rotor
    # surface magnets N/S
    for k in range(4):
        a0 = k * 90
        col = "#e74c3c" if k % 2 == 0 else "#3498db"
        ax.add_patch(Wedge((0, 0), 0.78, a0 + 8, a0 + 82, width=0.16, fc=col, ec="k", lw=0.5))
        am = np.deg2rad(a0 + 45)
        ax.text(0.68 * np.cos(am), 0.68 * np.sin(am), "N" if k % 2 == 0 else "S",
                ha="center", va="center", color="white", fontweight="bold")
    ax.add_patch(Circle((0, 0), 0.18, fc="#95a5a6", ec="k"))                # shaft
    ax.annotate("Stator yoke (laminated iron)", (0, 1.30), (1.5, 1.7),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.annotate("3-phase windings\n(A/B/C in slots)", (1.13, 0), (1.55, 0.4),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.annotate("Air gap", (0.89, 0.0), (1.4, -0.5), arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.annotate("Permanent magnets\n(rotor)", (0, -0.70), (-2.3, -1.2),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.annotate("Shaft", (0, 0), (-2.0, 0.3), arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Fig. PMSM construction (surface-mounted, 4-pole)", fontsize=11)
    save(fig, "fig_pmsm_construction.png")


# 2) Back-EMF: trapezoidal (BLDC) vs sinusoidal (PMSM) --------------------- #
def backemf():
    t = np.linspace(0, 2 * np.pi, 500)
    sine = np.sin(t)
    trap = np.clip(2.2 * np.sin(t), -1, 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(t, trap, color=RED, lw=2); ax[0].set_title("BLDC — trapezoidal back-EMF")
    ax[1].plot(t, sine, color=BLUE, lw=2); ax[1].set_title("PMSM — sinusoidal back-EMF")
    for a in ax:
        a.set_xlabel("electrical angle (rad)"); a.set_ylabel("back-EMF (pu)")
        a.grid(alpha=0.3); a.axhline(0, color="k", lw=0.5)
    save(fig, "fig_backemf_comparison.png")


# 3) Three-phase currents producing a rotating field ---------------------- #
def rotating_field():
    t = np.linspace(0, 2 * np.pi, 500)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for ph, c, lab in zip([0, -2*np.pi/3, 2*np.pi/3], [BLUE, RED, GREEN], "ABC"):
        ax[0].plot(t, np.sin(t + ph), color=c, label=f"phase {lab}")
    ax[0].legend(fontsize=8); ax[0].set_title("Balanced 3-phase currents (120° apart)")
    ax[0].set_xlabel("electrical angle"); ax[0].set_ylabel("current (pu)"); ax[0].grid(alpha=0.3)
    # resultant MMF vector at several instants
    ax[1].set_aspect("equal")
    for ang, alpha in zip(np.linspace(0, 2*np.pi, 9)[:-1], np.linspace(0.25, 1, 8)):
        ax[1].add_patch(FancyArrowPatch((0, 0), (np.cos(ang), np.sin(ang)),
                        arrowstyle="-|>", mutation_scale=12, color=PURPLE, alpha=alpha))
    ax[1].add_patch(Circle((0, 0), 1, fill=False, ls="--", ec=GREY))
    ax[1].set_xlim(-1.3, 1.3); ax[1].set_ylim(-1.3, 1.3)
    ax[1].set_title("Resultant MMF: a rotating field"); ax[1].axis("off")
    save(fig, "fig_rotating_field.png")


# 4) FOC control block diagram -------------------------------------------- #
def foc_block():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    def box(x, y, w, h, text, color="#eaf2fb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                     fc=color, ec="#34495e", lw=1.3))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=12, color="#34495e", lw=1.2))
    box(0.0, 1.1, 1.3, 0.7, "Speed PI\n(ω* → iq*)")
    box(1.7, 1.1, 1.3, 0.7, "Current PI\n(id*, iq*)")
    box(3.4, 1.1, 1.4, 0.7, "Inverse Park\n(dq → αβ)")
    box(5.2, 1.1, 1.3, 0.7, "SVPWM")
    box(6.9, 1.1, 1.5, 0.7, "3-phase\ninverter", "#fdebd0")
    box(8.8, 1.1, 1.2, 0.7, "PMSM", "#d5f5e3")
    box(5.2, 0.0, 1.5, 0.6, "Clarke+Park\n(abc → dq)")
    box(8.8, 0.0, 1.2, 0.6, "Encoder /\nsensorless", "#f9e0e0")
    for (x1, x2) in [(1.3, 1.7), (3.0, 3.4), (4.8, 5.2), (6.5, 6.9), (8.4, 8.8)]:
        arrow(x1, 1.45, x2, 1.45)
    arrow(9.4, 1.1, 9.4, 0.6)                 # motor -> sensor
    arrow(8.8, 0.3, 6.7, 0.3)                 # sensor -> clarke/park
    arrow(5.2, 0.3, 1.2, 0.3); ax.text(3.2, 0.4, "i_d, i_q feedback", fontsize=7, color=RED)
    arrow(0.6, 0.3, 0.6, 1.1)
    ax.text(8.0, 1.95, "current sensors on each phase", fontsize=7, color=BLUE)
    ax.set_xlim(-0.2, 10.3); ax.set_ylim(-0.3, 2.2); ax.axis("off")
    ax.set_title("Fig. Field-Oriented Control (FOC) of a PMSM", fontsize=11)
    save(fig, "fig_foc_block.png")


# 5) Three-phase 2-level inverter ----------------------------------------- #
def inverter():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot([0, 8], [3.6, 3.6], "k", lw=2); ax.plot([0, 8], [0.2, 0.2], "k", lw=2)
    ax.text(-0.3, 3.6, "+DC", va="center"); ax.text(-0.3, 0.2, "−DC", va="center")
    for i, (x, c, lab) in enumerate(zip([2, 4, 6], [BLUE, RED, GREEN], "ABC")):
        for y in (2.7, 1.0):
            ax.add_patch(Rectangle((x-0.22, y-0.18), 0.44, 0.36, fc="#eaf2fb", ec="k"))
            ax.plot([x-0.12, x+0.12], [y-0.10, y+0.10], "k", lw=1.2)   # switch stroke
            ax.plot([x+0.05, x+0.12, x+0.12], [y+0.10, y+0.10, y+0.03], "k", lw=1.2)  # arrowhead
        ax.plot([x, x], [3.6, 2.9], "k"); ax.plot([x, x], [2.5, 1.2], "k")
        ax.plot([x, x], [0.8, 0.2], "k")
        ax.plot([x, x+0.0], [1.85, 1.85], "k")
        ax.add_patch(FancyArrowPatch((x, 1.85), (x+1.0, 1.85), arrowstyle="-|>",
                     mutation_scale=10, color=c))
        ax.text(x+1.05, 1.85, f"phase {lab}", color=c, va="center", fontsize=8)
    ax.text(4, 4.0, "6 power switches (IGBT/MOSFET) — SPWM/SVPWM gating", ha="center", fontsize=8)
    ax.set_xlim(-0.6, 8.5); ax.set_ylim(-0.1, 4.3); ax.axis("off")
    ax.set_title("Fig. Two-level three-phase voltage-source inverter", fontsize=11)
    save(fig, "fig_inverter.png")


# 6) Inter-turn fault mechanism ------------------------------------------- #
def interturn_mechanism():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    # healthy coil
    xs = np.linspace(0.3, 3.0, 400)
    ax.plot(xs, 2.4 + 0.25*np.sin(xs*8), color=BLUE, lw=2)
    ax.text(1.6, 3.0, "Healthy winding (intact turn insulation)", color=BLUE, fontsize=8, ha="center")
    # faulty coil with shorted turns
    ax.plot(xs, 1.0 + 0.25*np.sin(xs*8), color="#555", lw=2)
    ax.add_patch(Circle((1.6, 1.0), 0.16, fc=RED, ec="k", zorder=5))
    ax.text(1.6, 1.0, "✗", color="white", ha="center", va="center", zorder=6)
    ax.add_patch(FancyArrowPatch((1.35, 0.78), (1.85, 0.78), connectionstyle="arc3,rad=-0.9",
                 arrowstyle="-|>", mutation_scale=12, color=RED))
    ax.text(1.6, 0.35, "shorted turns → large circulating fault current → local heating",
            color=RED, fontsize=8, ha="center")
    ax.text(3.25, 1.0, "heat ↑", fontsize=10, color=RED, fontweight="bold")
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 3.4); ax.axis("off")
    ax.set_title("Fig. Inter-turn stator short-circuit mechanism", fontsize=11)
    save(fig, "fig_interturn_mechanism.png")


# 7) MCSA spectrum: healthy vs inter-turn --------------------------------- #
def mcsa_spectrum():
    from python.simulate import build_signal, F0
    fs, dur = 10000, 1.0
    rng = np.random.default_rng(0)
    h = build_signal("Healthy", 0.0, fs=fs, duration=dur, rng=rng)
    f = build_signal("InterTurn", 0.7, fs=fs, duration=dur, rng=rng)
    freqs = np.fft.rfftfreq(len(h), 1/fs)
    keep = freqs <= 400
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.semilogy(freqs[keep], np.abs(np.fft.rfft(h))[keep] + 1, color=BLUE, lw=1.2, label="Healthy")
    ax.semilogy(freqs[keep], np.abs(np.fft.rfft(f))[keep] + 1, color=RED, lw=1.0, alpha=0.8, label="Inter-turn")
    ax.axvline(F0, color="k", ls="--", lw=0.7)
    ax.text(F0+3, ax.get_ylim()[1]*0.4, "f₀ (50 Hz)", fontsize=8)
    for h_ord in (3, 5, 7):
        ax.axvline(h_ord*F0, color=ORANGE, ls=":", lw=0.7)
        ax.text(h_ord*F0+2, 30, f"{h_ord}f₀", fontsize=7, color=ORANGE)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("magnitude (log)")
    ax.set_title("Fig. MCSA — fault raises harmonics/side-bands (the classic manual method)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    save(fig, "fig_mcsa_spectrum.png")


# 8) Torque–speed characteristic ------------------------------------------ #
def torque_speed():
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    n = np.linspace(0, 2, 400); base = 1.0
    T = np.where(n <= base, 1.0, base/np.clip(n, 1e-3, None))
    ax.plot(n, T, color=BLUE, lw=2)
    ax.fill_between(n[n<=base], 0, T[n<=base], color=BLUE, alpha=0.1)
    ax.axvline(base, color=GREY, ls="--")
    ax.text(0.45, 0.5, "Constant-torque\nregion", ha="center", fontsize=8)
    ax.text(1.5, 0.5, "Field-weakening\n(constant power)", ha="center", fontsize=8)
    ax.set_xlabel("speed (pu)"); ax.set_ylabel("torque (pu)")
    ax.set_title("Fig. PMSM torque–speed envelope"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.2)
    save(fig, "fig_torque_speed.png")


# 9) Fault-detection methods taxonomy ------------------------------------- #
def detection_taxonomy():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="#34495e"))
        ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=8)
    box(4.3, 3.4, 2.4, 0.6, "PMSM fault detection", "#d6eaf8")
    cats = [("Manual / offline\n(thermal cam, Megger,\ninsulation/surge test)", "#fdebd0", 0.2),
            ("Signature (online)\nMCSA-FFT,\nvibration FFT/envelope", "#fcf3cf", 3.1),
            ("Model-based\nobservers, residuals,\nparameter estimation", "#e8daef", 6.0),
            ("AI / data-driven\nscalogram + CNN\n(THIS WORK)", "#d5f5e3", 8.9)]
    for t, c, x in cats:
        box(x, 1.7, 2.6, 1.0, t, c)
        ax.add_patch(FancyArrowPatch((5.5, 3.4), (x+1.3, 2.7), arrowstyle="-|>",
                     mutation_scale=10, color="#34495e"))
    notes = ["needs shutdown,\nperiodic, skilled", "fixed thresholds,\nfooled by load/speed",
             "needs accurate\nmotor model", "learns features,\nrobust, automatic"]
    for (t, c, x), nt in zip(cats, notes):
        ax.text(x+1.3, 1.45, nt, ha="center", va="top", fontsize=7, color="#555")
    ax.set_xlim(0, 11.7); ax.set_ylim(0.4, 4.2); ax.axis("off")
    ax.set_title("Fig. Landscape of PMSM fault-detection methods", fontsize=11)
    save(fig, "fig_detection_taxonomy.png")


# 10) Motor family comparison (table figure) ------------------------------ #
def motor_comparison():
    cols = ["PMSM", "PMDC", "BLDC", "Induction"]
    rows = ["Commutation", "Back-EMF", "Efficiency", "Torque density",
            "Control complexity", "Maintenance", "Rotor current", "Typical use"]
    data = [
        ["Electronic (inverter)", "Mechanical brushes", "Electronic (inverter)", "Electronic/DOL"],
        ["Sinusoidal", "—(DC)", "Trapezoidal", "Sinusoidal"],
        ["Very high", "Medium", "High", "Medium"],
        ["Very high", "Low", "High", "Medium"],
        ["High (FOC)", "Low", "Medium", "Medium/High"],
        ["Low (no brushes)", "High (brushes)", "Low", "Low"],
        ["None", "Yes (armature)", "None", "Induced (slip)"],
        ["EV, robotics, servo", "Toys, simple drives", "Drones, fans", "Pumps, industry"],
    ]
    fig, ax = plt.subplots(figsize=(11, 3.8)); ax.axis("off")
    tbl = ax.table(cellText=data, rowLabels=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    for j in range(len(cols)):           # highlight PMSM column
        tbl[(0, j)].set_facecolor("#34495e"); tbl[(0, j)].set_text_props(color="white")
    for i in range(1, len(rows)+1):
        tbl[(i, 0)].set_facecolor("#eaf2fb")
    ax.set_title("Fig. PMSM vs its relatives (PMDC / BLDC / Induction)", fontsize=11, pad=14)
    save(fig, "fig_motor_comparison.png")


# 11) Why CWT+CNN improves on FFT-threshold ------------------------------- #
def method_improvement():
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    sev = np.linspace(0, 1, 100)
    fft_thr = 1/(1+np.exp(-(sev-0.55)*12))      # detects only above a threshold
    cnn = 1/(1+np.exp(-(sev-0.25)*12))          # detects earlier
    ax.plot(sev*100, fft_thr*100, color=ORANGE, lw=2, label="FFT + fixed threshold (manual)")
    ax.plot(sev*100, cnn*100, color=GREEN, lw=2, label="CWT scalogram + CNN (this work)")
    ax.fill_between(sev*100, fft_thr*100, cnn*100, where=cnn>fft_thr, color=GREEN, alpha=0.12)
    ax.set_xlabel("fault severity (%)"); ax.set_ylabel("detection probability (%)")
    ax.set_title("Fig. Earlier detection: learned time–frequency features vs fixed thresholds")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.text(35, 30, "earlier\ndetection\nzone", color=GREEN, fontsize=8, ha="center")
    save(fig, "fig_method_improvement.png")


if __name__ == "__main__":
    pmsm_construction(); backemf(); rotating_field(); foc_block(); inverter()
    interturn_mechanism(); mcsa_spectrum(); torque_speed(); detection_taxonomy()
    motor_comparison(); method_improvement()
    print("done — figures in", OUT)
