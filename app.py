"""
PMSM Fault Diagnosis — Comprehensive Interactive Dashboard
==========================================================
A step-by-step, hands-on walkthrough of the entire project: from a raw motor
signal, through the Continuous Wavelet Transform, to a CNN diagnosis — with a
live test lab so you can try the model by hand.

Run:  ./run_demo.sh      (or)   .venv/bin/python -m streamlit run app.py
"""
import os
import json
import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.cm as cm
from PIL import Image
import streamlit as st

from python.config import load_config
from python.simulate import build_signal, F0, FR
from python.scalogram import compute_scalogram
from python.ingest_real import segment_signal

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PMSM Fault Diagnosis — Interactive",
                   page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
cfg = load_config()
CLASSES = cfg["classes"]
TARGET_FS = cfg["target_fs"]
WINDOW_SECONDS = cfg["window_seconds"]
IMAGE_SIZE = cfg["image_size"]
WAVELET = cfg.get("wavelet_py", "cmor1.5-1.0")
N_SCALES = cfg.get("n_scales", 128)
PATHS = cfg["paths"]

CLASS_COLORS = {"Healthy": "#2ecc71", "InterTurn": "#e74c3c",
                "Demagnetization": "#f39c12", "Overload": "#9b59b6"}

st.markdown("""
<style>
.big-metric {background:#0e1117;border:1px solid #2a2f3a;border-radius:12px;padding:14px 18px;}
.step-badge {display:inline-block;background:#1f6feb;color:#fff;border-radius:50%;
             width:28px;height:28px;text-align:center;line-height:28px;font-weight:700;margin-right:8px;}
.pill {display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;}
hr {margin:0.6rem 0;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders (degrade gracefully when data/models are absent)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(name):
    import tensorflow as tf
    p = os.path.join(PATHS["models"], name)
    return tf.keras.models.load_model(p) if os.path.exists(p) else None


@st.cache_data
def load_manifest_safe():
    p = PATHS["manifest"]
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


@st.cache_data
def load_json_safe(path):
    return json.load(open(path)) if os.path.exists(path) else None


def render_scalogram(signal, fs, wavelet=WAVELET, n_scales=N_SCALES, size=IMAGE_SIZE, cmap="jet"):
    """Return (|CWT| matrix, PIL image) — the exact pipeline used in training."""
    A = compute_scalogram(signal, fs, wavelet=wavelet, n_scales=n_scales)
    An = A / (A.max() + 1e-12)
    rgb = (getattr(cm, cmap)(An)[..., :3] * 255).astype("uint8")
    img = Image.fromarray(rgb).resize((size, size))
    return A, img


def predict(model, img):
    arr = np.expand_dims(np.array(img) / 255.0, 0)
    p = model.predict(arr, verbose=0)[0]
    return p


def signal_plot(signal, fs, title, color="#3498db"):
    t = np.arange(len(signal)) / fs
    fig = go.Figure(go.Scatter(x=t, y=signal, mode="lines", line=dict(color=color, width=1)))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude", title=title,
                      margin=dict(l=0, r=0, t=35, b=0), height=300)
    return fig


def prob_bars(p):
    fig = go.Figure(go.Bar(x=p * 100, y=CLASSES, orientation="h",
                           text=[f"{v*100:.1f}%" for v in p], textposition="auto",
                           marker=dict(color=[CLASS_COLORS.get(c, "#888") for c in CLASSES])))
    fig.update_layout(xaxis_title="Confidence (%)", xaxis=dict(range=[0, 100]),
                      height=220, margin=dict(l=0, r=0, t=10, b=0))
    return fig


MANIFEST = load_manifest_safe()
MODELS = {"current": load_model("cnn_current.keras"),
          "vibration": load_model("cnn_vibration.keras"),
          "fusion": load_model("cnn_fusion.keras")}

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚡ PMSM Diagnosis")
st.sidebar.caption("Interactive project walkthrough & test lab")
SECTIONS = [
    "🏠 Home & Pipeline",
    "🩺 The Problem (PMSM & Faults)",
    "🔬 Signal Lab: Fourier vs Wavelet",
    "🖼️ Scalogram Studio",
    "🗂️ Dataset Explorer",
    "🧠 The CNN Model",
    "📈 Results & Ablations",
    "🧪 Test Lab (try it by hand)",
    "📚 Concepts & Defense Prep",
]
page = st.sidebar.radio("Walk through the project:", SECTIONS)
st.sidebar.markdown("---")
# environment status
def ok(b): return "🟢" if b else "🔴"
st.sidebar.markdown(f"""**Environment**
{ok(MANIFEST is not None)} real manifest ({0 if MANIFEST is None else len(MANIFEST)} rows)
{ok(MODELS['current'] is not None)} current model
{ok(MODELS['vibration'] is not None)} vibration model
{ok(MODELS['fusion'] is not None)} fusion model""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Authors:** Mulham Fetna · Mohammad Zein Qabbani")
st.sidebar.caption("«Supervisor / University» — fill on the cover")


# =========================================================================== #
# 1. HOME & PIPELINE
# =========================================================================== #
if page == "🏠 Home & Pipeline":
    st.title("⚡ PMSM Fault Diagnosis using Wavelet Scalograms & CNNs")
    st.markdown("#### Detect motor faults by turning signals into images and classifying them with deep learning.")

    m = load_json_safe("results/real_metrics.json")
    c1, c2, c3, c4 = st.columns(4)
    if m:
        c1.metric("Vibration — balanced acc", f"{m['vibration']['balanced_accuracy']:.2f}", "best channel")
        c2.metric("Fusion — balanced acc", f"{m['fusion']['balanced_accuracy']:.2f}")
        c3.metric("Current — balanced acc", f"{m['current']['balanced_accuracy']:.2f}", "weak channel", delta_color="inverse")
    c4.metric("Scalograms", "3,150", "real KAIST")

    st.markdown("### The pipeline — one signal's journey")
    steps = [
        ("1", "Signal", "Current @100 kHz / vibration @25.6 kHz from the motor."),
        ("2", "Segment", "Decimate to 10 kHz, cut into 0.5 s windows (50% overlap)."),
        ("3", "CWT", "Continuous Wavelet Transform (complex Morlet, 128 scales)."),
        ("4", "Scalogram", "Render |CWT| as a 224×224 RGB time–frequency image."),
        ("5", "CNN", "3 conv blocks classify the image's texture."),
        ("6", "Diagnosis", "Healthy / Inter-turn (+ Demag / Overload synthetic)."),
    ]
    cols = st.columns(6)
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f"<span class='step-badge'>{n}</span>**{t}**", unsafe_allow_html=True)
        col.caption(d)

    st.info("Use the sidebar to walk through each stage. Every section is interactive — "
            "the **Scalogram Studio** and **Test Lab** let you run the real pipeline by hand.")

    st.markdown("### What makes this project solid")
    a, b, c = st.columns(3)
    a.markdown("**Real data** — KAIST PMSM (current+vibration), not just synthetic.")
    a.markdown("**Leakage-free** — split by recording, never by window.")
    b.markdown("**Honest metrics** — balanced accuracy / macro-F1 on an imbalanced test set.")
    b.markdown("**Reproducible** — one config, one manifest, 38 tests, CI.")
    c.markdown("**Python-only** — PyWavelets + TensorFlow (MATLAB optional).")
    c.markdown("**Bilingual** — EN + AR report & slides in `docs/build/`.")


# =========================================================================== #
# 2. THE PROBLEM
# =========================================================================== #
elif page == "🩺 The Problem (PMSM & Faults)":
    st.title("🩺 The Problem: PMSM Motors & Their Faults")
    st.markdown("""
**PMSM** = Permanent Magnet Synchronous Motor. The rotor's permanent magnets lock
onto the stator's rotating magnetic field and spin **synchronously** — no slip, no
rotor current → very high efficiency. They drive EVs, robots, aircraft actuators.
Controlled by **Field-Oriented Control (FOC)**: current is split into `i_d` (flux,
kept ≈0) and `i_q` (torque).
""")
    st.markdown("### The faults we classify")
    tabs = st.tabs(CLASSES)
    info = {
        "Healthy": ("✅ Normal operation", "Clean fundamental at 50 Hz plus the usual odd harmonics. Baseline class."),
        "InterTurn": ("🔴 Inter-turn short (our main fault)", "Insulation between winding turns breaks → a shorted loop carries a large current, overheats, and escalates to total winding failure within minutes. Introduces asymmetry, extra harmonics and side-bands."),
        "Demagnetization": ("🟠 Demagnetization", "Rotor magnets weaken (heat/age) → lower back-EMF and torque; sub-harmonics and f0±k·fr side-bands appear. (Synthetic only — absent from the real dataset.)"),
        "Overload": ("🟣 Overload", "Sustained operation above rated load → raised fundamental + even harmonics + more noise. A degraded operating state. (Synthetic only.)"),
    }
    for tab, c in zip(tabs, CLASSES):
        title, desc = info[c]
        with tab:
            st.markdown(f"#### {title}")
            st.write(desc)
            rng = np.random.default_rng(0)
            sig = build_signal(c, 0.5 if c == "InterTurn" else 0.0, fs=TARGET_FS,
                               duration=WINDOW_SECONDS, rng=rng)
            st.plotly_chart(signal_plot(sig, TARGET_FS, f"Synthetic {c} current signal",
                            color=CLASS_COLORS.get(c, "#3498db")), use_container_width=True)

    st.markdown("### Why two sensors — current *and* vibration?")
    cc, vv = st.columns(2)
    cc.markdown("**Current** — free (sensors already in every FOC drive), but the closed-loop controller *suppresses* fault signatures → weaker signal.")
    vv.markdown("**Vibration** — needs an accelerometer, but responds *directly* to the fault → much cleaner signature. (Our results confirm vibration wins.)")


# =========================================================================== #
# 3. SIGNAL LAB: FOURIER vs WAVELET
# =========================================================================== #
elif page == "🔬 Signal Lab: Fourier vs Wavelet":
    st.title("🔬 Signal Lab — why wavelets beat Fourier here")
    st.markdown("Build a signal, then compare what the **FFT** sees (frequencies, no time) "
                "vs what the **scalogram** sees (frequencies *and* time).")

    c1, c2, c3 = st.columns(3)
    klass = c1.selectbox("Condition", CLASSES, index=1)
    sev = c2.slider("Inter-turn severity", 0.0, 1.0, 0.5, 0.05)
    noise = c3.slider("Extra noise", 0.0, 0.5, 0.0, 0.05)
    rng = np.random.default_rng(1)
    sig = build_signal(klass, sev, fs=TARGET_FS, duration=WINDOW_SECONDS, rng=rng)
    if noise:
        sig = sig + rng.normal(0, noise * np.std(sig), len(sig))

    st.plotly_chart(signal_plot(sig, TARGET_FS, "Time-domain signal"), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### FFT — frequency only (blind to *when*)")
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), 1 / TARGET_FS)
        keep = freqs <= 1000
        fig = go.Figure(go.Scatter(x=freqs[keep], y=spec[keep], line=dict(color="#e67e22")))
        fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                          height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Fundamental at {F0:.0f} Hz; faults add harmonics/side-bands. "
                   "But if a fault were transient, FFT would smear it — it can't say *when*.")
    with colB:
        st.markdown("#### Scalogram — frequency **and** time")
        A, img = render_scalogram(sig, TARGET_FS)
        st.image(img, caption="224×224 CWT scalogram (Morlet)", use_container_width=True)
        st.caption("Vertical = frequency, horizontal = time, color = energy. "
                   "Low freq = fine frequency/coarse time; high freq = the reverse "
                   "(the uncertainty principle, Δt·Δf ≥ const).")


# =========================================================================== #
# 4. SCALOGRAM STUDIO
# =========================================================================== #
elif page == "🖼️ Scalogram Studio":
    st.title("🖼️ Scalogram Studio — see how every parameter changes the image")
    st.markdown("This is the exact CWT used in training. Change the knobs and watch the "
                "raw `|CWT|` matrix and the final RGB image update.")

    c1, c2, c3 = st.columns(3)
    klass = c1.selectbox("Condition", CLASSES, index=1)
    sev = c1.slider("Severity (Inter-turn)", 0.0, 1.0, 0.5, 0.05)
    n_scales = c2.slider("Number of CWT scales", 16, 256, N_SCALES, 16)
    cmap = c2.selectbox("Colormap", ["jet", "viridis", "magma", "turbo"], 0)
    size = c3.select_slider("Image size (px)", [64, 96, 128, 160, 224], IMAGE_SIZE)
    bw = c3.slider("Morlet bandwidth", 0.5, 3.0, 1.5, 0.5)

    rng = np.random.default_rng(2)
    sig = build_signal(klass, sev, fs=TARGET_FS, duration=WINDOW_SECONDS, rng=rng)
    wavelet = f"cmor{bw}-1.0"
    A, img = render_scalogram(sig, TARGET_FS, wavelet=wavelet, n_scales=n_scales, size=size, cmap=cmap)

    colL, colR = st.columns([3, 2])
    with colL:
        st.markdown("**Raw |CWT| coefficient matrix** (scales × time)")
        heat = px.imshow(A, aspect="auto", color_continuous_scale=cmap,
                         labels=dict(x="time samples", y="scale index", color="|CWT|"))
        heat.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(heat, use_container_width=True)
    with colR:
        st.markdown("**Final model input image**")
        st.image(img, caption=f"{size}×{size} · {wavelet} · {n_scales} scales", use_container_width=True)
        st.metric("Matrix shape", f"{A.shape[0]} × {A.shape[1]}")
        st.caption("Scales are geometric 4→256; each maps to a pseudo-frequency. "
                   "More scales = finer vertical resolution but slower.")


# =========================================================================== #
# 5. DATASET EXPLORER
# =========================================================================== #
elif page == "🗂️ Dataset Explorer":
    st.title("🗂️ Dataset Explorer — the real KAIST data")
    if MANIFEST is None:
        st.warning("No `data/manifest.csv` found (the dataset is gitignored). "
                   "Run the ingestion pipeline locally to populate it. "
                   "Showing the documented composition instead.")
        st.markdown("""
| Channel | Healthy | InterTurn | Recordings (H/IT) |
|---|---|---|---|
| current | 200 | 1350 | 4 / 27 |
| vibration | 200 | 1400 | 4 / 28 |
""")
    else:
        df = MANIFEST
        st.markdown(f"**{len(df)} segments** · {df['signal_type'].nunique()} channels · "
                    f"{df['recording_id'].nunique()} recordings")
        c1, c2 = st.columns(2)
        cross = df.groupby(["signal_type", "class"]).size().reset_index(name="segments")
        fig = px.bar(cross, x="signal_type", y="segments", color="class", barmode="group",
                     color_discrete_map=CLASS_COLORS, title="Segments per class & channel")
        c1.plotly_chart(fig, use_container_width=True)
        if "split" in df.columns:
            sp = df.groupby(["split", "class"]).size().reset_index(name="segments")
            fig2 = px.bar(sp, x="split", y="segments", color="class", barmode="group",
                          color_discrete_map=CLASS_COLORS, title="Segments per split (leakage-free, by recording)")
            c2.plotly_chart(fig2, use_container_width=True)
        st.warning("⚖️ Note the **imbalance**: ~4 healthy vs ~60 fault recordings. "
                   "This is why we balance the training set and report **balanced accuracy**.")

        st.markdown("### Browse real scalograms")
        b1, b2, b3 = st.columns(3)
        ch = b1.selectbox("Channel", sorted(df["signal_type"].unique()))
        cl = b2.selectbox("Class", sorted(df[df.signal_type == ch]["class"].unique()))
        n = b3.slider("How many", 3, 12, 6)
        folder = os.path.join(PATHS["scalograms"], ch, cl)
        imgs = sorted(glob.glob(os.path.join(folder, "*.png")))[:n]
        if imgs:
            for row in range(0, len(imgs), 3):
                for col, ip in zip(st.columns(3), imgs[row:row + 3]):
                    col.image(ip, caption=os.path.basename(ip)[:22], use_container_width=True)
        else:
            st.info(f"No rendered scalograms in `{folder}` (run `make scalograms`).")


# =========================================================================== #
# 6. THE CNN MODEL
# =========================================================================== #
elif page == "🧠 The CNN Model":
    st.title("🧠 The CNN Model")
    st.markdown("""
A compact CNN, sized deliberately for a few-thousand-image dataset.
""")
    st.code("""Input 224×224×3
 → Conv2D(32,3×3) → ReLU → MaxPool        # learn edges / bands
 → Conv2D(64,3×3) → ReLU → MaxPool        # learn textures
 → Conv2D(128,3×3) → ReLU → MaxPool       # learn fault motifs
 → Flatten → Dropout(0.5)                 # regularize
 → Dense(128) → ReLU
 → Dense(num_classes) → Softmax           # class probabilities""", language="text")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Optimizer** Adam")
    c1.markdown("**Loss** sparse categorical cross-entropy")
    c2.markdown("**Early stopping** patience 5, restore best")
    c2.markdown("**Augment** random horizontal flip")
    c3.markdown("**Balancing** undersample majority (train/val)")
    c3.markdown("**Fusion** two branches + global avg pooling")

    if MODELS["current"] is not None:
        with st.expander("📐 Live model.summary() (current channel)"):
            lines = []
            MODELS["current"].summary(print_fn=lambda x: lines.append(x))
            st.code("\n".join(lines), language="text")
            st.metric("Total parameters", f"{MODELS['current'].count_params():,}")

    st.markdown("### Training curves")
    cur = load_json_safe("results/history_current.json")
    vib = load_json_safe("results/history_vibration.json")
    cols = st.columns(2)
    for col, hist, name in [(cols[0], cur, "current"), (cols[1], vib, "vibration")]:
        if hist:
            fig = go.Figure()
            if "accuracy" in hist:
                fig.add_trace(go.Scatter(y=hist["accuracy"], name="train acc"))
            if "val_accuracy" in hist:
                fig.add_trace(go.Scatter(y=hist["val_accuracy"], name="val acc"))
            fig.update_layout(title=f"{name} training", xaxis_title="epoch",
                              yaxis_title="accuracy", height=300, margin=dict(l=0, r=0, t=35, b=0))
            col.plotly_chart(fig, use_container_width=True)
        else:
            col.info(f"No history_{name}.json yet.")


# =========================================================================== #
# 7. RESULTS & ABLATIONS
# =========================================================================== #
elif page == "📈 Results & Ablations":
    st.title("📈 Results & Ablation Studies")
    m = load_json_safe("results/real_metrics.json")
    if m:
        rows = [{"Channel": k.capitalize(), "Test acc": v["accuracy"],
                 "Balanced acc": v["balanced_accuracy"], "Macro-F1": v["macro_f1_2class"],
                 "Healthy recall": v["healthy_recall"], "InterTurn recall": v["interturn_recall"]}
                for k, v in m.items()]
        dfm = pd.DataFrame(rows).sort_values("Balanced acc", ascending=False)
        st.table(dfm.set_index("Channel").style.format("{:.2f}"))
        fig = px.bar(dfm, x="Channel", y="Balanced acc", color="Channel", text="Balanced acc",
                     range_y=[0, 1.05], title="Balanced accuracy by channel")
        st.plotly_chart(fig, use_container_width=True)
        st.success("**Key finding:** vibration scalograms detect inter-turn faults perfectly "
                   "on held-out recordings (1.00); current is much weaker (0.69).")
    st.caption("⚠️ Limitation: only 4 distinct healthy recordings exist, so the perfect "
               "vibration score can't fully exclude a recording-identity shortcut.")

    st.markdown("### Confusion matrices")
    c1, c2 = st.columns(2)
    if os.path.exists("results/confusion_real_2class.png"):
        c1.image("results/confusion_real_2class.png", caption="Current vs Vibration (2-class)", use_container_width=True)
    if os.path.exists("results/example_scalograms_real.png"):
        c2.image("results/example_scalograms_real.png", caption="Real scalograms", use_container_width=True)

    st.markdown("### Ablation experiments")
    exp = load_json_safe("results/experiments_real.json")
    if exp:
        t1, t2, t3 = st.tabs(["Balancing", "Image size", "Learning curve"])
        with t1:
            d = pd.DataFrame(exp["balancing"])
            st.plotly_chart(px.bar(d, x="channel", y="balanced_acc", color="balanced",
                            barmode="group", range_y=[0, 1.05],
                            title="Effect of class balancing"), use_container_width=True)
            st.caption("Without balancing the current model collapses (healthy recall → 0).")
        with t2:
            d = pd.DataFrame(exp["image_size"])
            st.plotly_chart(px.line(d, x="image_size", y="balanced_acc", color="channel",
                            markers=True, range_y=[0, 1.05],
                            title="Effect of scalogram image size"), use_container_width=True)
            st.caption("Bigger images help the weak current channel; vibration is saturated.")
        with t3:
            d = pd.DataFrame(exp["learning_curve"])
            st.plotly_chart(px.line(d, x="n_train", y="balanced_acc", color="channel",
                            markers=True, range_y=[0, 1.05],
                            title="Effect of training-set size"), use_container_width=True)
            st.caption("Vibration needs ~46 images; current is flat — its limit is signal "
                       "quality, not quantity.")
    elif os.path.exists("results/learning_curve.png"):
        st.image("results/learning_curve.png", use_container_width=True)


# =========================================================================== #
# 8. TEST LAB
# =========================================================================== #
elif page == "🧪 Test Lab (try it by hand)":
    st.title("🧪 Test Lab — run the real pipeline by hand")
    mode = st.radio("Mode", ["Synthetic generator", "Real test sample", "Upload .npy", "🎮 Quiz me"],
                    horizontal=True)

    # ---- choose a signal + channel + ground truth -------------------------- #
    signal, channel, truth = None, "current", "Unknown"

    if mode == "Synthetic generator":
        c1, c2, c3 = st.columns(3)
        truth = c1.selectbox("Condition (ground truth)", CLASSES)
        sev = c2.slider("Severity", 0.0, 1.0, 0.5, 0.05) if truth == "InterTurn" else 0.0
        channel = c3.radio("Channel / model", ["current", "vibration"], horizontal=True)
        if st.button("⚙️ Generate & diagnose", type="primary"):
            signal = build_signal(truth, sev, fs=TARGET_FS, duration=WINDOW_SECONDS,
                                  rng=np.random.default_rng())

    elif mode == "Real test sample":
        if MANIFEST is None:
            st.warning("Needs `data/manifest.csv` + rendered scalograms (run the pipeline locally).")
        else:
            test = MANIFEST[MANIFEST.get("split", "") == "test"] if "split" in MANIFEST else MANIFEST
            c1, c2 = st.columns(2)
            channel = c1.selectbox("Channel / model", sorted(test["signal_type"].unique()))
            sub = test[test.signal_type == channel]
            cl = c2.selectbox("True class", sorted(sub["class"].unique()))
            sub = sub[sub["class"] == cl]
            if st.button("🎲 Pick a random held-out sample & diagnose", type="primary") and len(sub):
                row = sub.sample(1, random_state=int(np.random.randint(1e6))).iloc[0]
                truth = row["class"]
                segp = os.path.join(PATHS["raw"], "segments", row["signal_id"] + ".npy")
                if os.path.exists(segp):
                    signal = np.load(segp)
                elif os.path.exists(str(row.get("scalogram_path", ""))):
                    st.session_state["_preimg"] = row["scalogram_path"]
                    st.session_state["_truth"] = truth
                    st.session_state["_channel"] = channel

    elif mode == "Upload .npy":
        c1, c2 = st.columns(2)
        up = c1.file_uploader("1-D signal (.npy)", type=["npy"])
        channel = c2.radio("Channel / model", ["current", "vibration"], horizontal=True)
        if up and st.button("Run pipeline", type="primary"):
            signal = np.load(up)
            need = int(TARGET_FS * WINDOW_SECONDS)
            signal = signal[:need] if len(signal) >= need else np.pad(signal, (0, need - len(signal)))

    elif mode == "🎮 Quiz me":
        st.markdown("**Can you beat the AI?** A scalogram is shown — you guess, then the model guesses.")
        if st.button("🃏 New round"):
            tc = np.random.choice(CLASSES)
            sv = float(np.random.uniform(0.2, 0.9)) if tc == "InterTurn" else 0.0
            sig = build_signal(tc, sv, fs=TARGET_FS, duration=WINDOW_SECONDS, rng=np.random.default_rng())
            st.session_state["quiz_sig"] = sig
            st.session_state["quiz_truth"] = tc
            st.session_state["quiz_channel"] = np.random.choice(["current", "vibration"])
        if "quiz_sig" in st.session_state:
            _, qimg = render_scalogram(st.session_state["quiz_sig"], TARGET_FS)
            st.image(qimg, width=260, caption="What condition is this?")
            guess = st.selectbox("Your guess", CLASSES, key="quiz_guess")
            if st.button("Reveal"):
                signal = st.session_state["quiz_sig"]
                channel = st.session_state["quiz_channel"]
                truth = st.session_state["quiz_truth"]
                st.session_state["_quiz_guess"] = guess

    # ---- run the model + show the 3-step result ---------------------------- #
    preimg = st.session_state.pop("_preimg", None)
    if preimg:
        signal, truth, channel = "img", st.session_state.pop("_truth"), st.session_state.pop("_channel")

    if signal is not None:
        model = MODELS.get(channel)
        if model is None:
            st.error(f"Model `cnn_{channel}.keras` not found — train it first (`make train SIGNAL={channel}`).")
        else:
            if preimg:
                img = Image.open(preimg).resize((IMAGE_SIZE, IMAGE_SIZE))
                st.image(img, width=240, caption="held-out scalogram")
            else:
                colS, colI = st.columns([3, 2])
                colS.plotly_chart(signal_plot(signal, TARGET_FS, "Step 1 — signal"), use_container_width=True)
                _, img = render_scalogram(signal, TARGET_FS)
                colI.image(img, caption="Step 2 — scalogram", use_container_width=True)

            p = predict(model, img)
            pred = CLASSES[int(np.argmax(p))]
            conf = float(np.max(p)) * 100
            st.markdown("### Step 3 — CNN diagnosis")
            r1, r2 = st.columns([1, 2])
            r1.metric("AI diagnosis", pred, f"{conf:.1f}% confidence")
            r2.plotly_chart(prob_bars(p), use_container_width=True)
            if truth != "Unknown":
                quiz_guess = st.session_state.pop("_quiz_guess", None)
                if quiz_guess is not None:
                    st.write(f"🧑 You guessed **{quiz_guess}** · 🤖 AI said **{pred}** · ✅ truth **{truth}**")
                    you = "✅" if quiz_guess == truth else "❌"
                    ai = "✅" if pred == truth else "❌"
                    st.write(f"You {you} — AI {ai}")
                elif pred == truth:
                    st.success(f"✅ Correct — ground truth is **{truth}**")
                else:
                    st.error(f"❌ Wrong — ground truth is **{truth}** (model said {pred})")
    else:
        st.info("Pick inputs above and press the button to run signal → scalogram → CNN.")


# =========================================================================== #
# 9. CONCEPTS & DEFENSE
# =========================================================================== #
elif page == "📚 Concepts & Defense Prep":
    st.title("📚 Concepts & Defense Prep")
    st.markdown("Quick-reference for the viva. Full guide: `docs/defense-study-guide.md` (EN/AR).")

    with st.expander("🎤 60-second elevator pitch", expanded=True):
        st.write("PMSM motors develop inter-turn stator shorts that can destroy the winding. "
                 "We cut the current/vibration signal into 0.5 s windows, turn each into a "
                 "**wavelet scalogram** (a time–frequency image), and a **CNN** classifies it. "
                 "On real KAIST data, **vibration reaches balanced accuracy 1.00** on unseen "
                 "recordings; current is far weaker (0.69). Python-only, reproducible, tested. "
                 "We report balanced accuracy because the data is imbalanced.")

    qa = [
        ("Why scalograms instead of the raw signal?",
         "They expose fault signatures as spatial patterns and let us use CNNs, which excel at images."),
        ("Why wavelets not FFT?",
         "FFT is blind to time and assumes stationarity; fault signals are non-stationary. Wavelets give frequency-dependent time–frequency resolution."),
        ("Vibration is 1.00 — overfitting?",
         "Split is leakage-free by recording and the score is on held-out recordings. Caveat: only 4 healthy recordings, so we present it as a limitation, not a finished product."),
        ("Why balanced accuracy not accuracy?",
         "Test set is 50 healthy / 200 faulty; always-'faulty' scores 80% but detects no healthy motor. Balanced accuracy exposes that."),
        ("Why is current so weak?",
         "The inter-turn current signature is weak at these severities and FOC suppresses it. The learning curve is flat → it's a signal-quality limit, not data quantity."),
        ("Do you still need MATLAB?",
         "No — everything runs in Python (PyWavelets + TensorFlow). MATLAB is an optional alternative."),
        ("What is data leakage and how avoided?",
         "Near-identical windows of one recording in both train and test inflate scores. We split by recording_id, stratified, per channel."),
    ]
    for q, a in qa:
        with st.expander("❓ " + q):
            st.write(a)

    st.markdown("### Numbers to memorize")
    st.markdown("""
- Dataset: KAIST `rgn5brrgrn` — current @100 kHz, vibration @25.6 kHz → decimated 10 kHz
- Windows 0.5 s, 50% overlap; 50 seg/recording; **3,150** scalograms
- Morlet `cmor1.5-1.0`, **128** scales, **224×224**; split 70/15/15 by recording
- Results: vibration **1.00**, current **0.69**, fusion **0.88** (balanced acc)
- 38 tests · CI · GPU (P620, ~17×) · Python-only
""")

st.markdown("---")
st.caption("PMSM Fault Diagnosis dashboard · github.com/molhamfetnah/pmsm-fault-diagnosis-cnn-scalogram")
