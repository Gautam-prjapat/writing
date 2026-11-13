import streamlit as st

# Must be the first Streamlit call
st.set_page_config(
    page_title="Handwriting Style Transfer",
    page_icon="✍️",
    layout="wide",
)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU; remove if GPU available

import io
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gdown

# ------------------------------------------------------------------
# Model weights download (Google Drive)
# ------------------------------------------------------------------
WEIGHTS_PATH = "./weights/model_weights.h5"
WEIGHTS_FILE_ID = "1U1h_sfTx2F51vmBGnELfZvKenospyITo"


def ensure_weights():
    if os.path.exists(WEIGHTS_PATH):
        return
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with st.spinner("Downloading model weights (first run only)…"):
        try:
            gdown.download(
                id=WEIGHTS_FILE_ID,
                output=WEIGHTS_PATH,
                quiet=False,
                fuzzy=True,
            )
        except Exception as e:
            st.error(
                "Could not download model weights from Google Drive.\n\n"
                f"Details: {e}\n\n"
                "Make sure the file is publicly shared and that the ID is correct."
            )
            st.stop()


# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading diffusion model…")
def load_model():
    import tensorflow as tf
    import nn

    model = nn.DiffusionWriter(num_layers=2, c1=128, c2=192, c3=256)
    # Warm-up call so the model builds variables before loading
    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)
    model.load_weights(WEIGHTS_PATH)
    return model


@st.cache_resource(show_spinner="Loading style extractor…")
def load_style_extractor():
    import nn
    return nn.StyleExtractor()


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def read_writer_image(file_obj):
    """Read an uploaded image file and preprocess it exactly like training."""
    import preprocessing

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name
    try:
        img = preprocessing.read_img(tmp_path, 96)
    finally:
        os.unlink(tmp_path)
    return img


def default_writer_image():
    """Pick a random asset from ./assets, or fall back to a blank canvas."""
    import preprocessing

    if os.path.isdir("./assets"):
        assets = [
            f for f in os.listdir("./assets")
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        if assets:
            return preprocessing.read_img(
                os.path.join("./assets", assets[0]), 96
            )
    # Blank white canvas — the model will produce a "generic" style
    return np.full((96, 400, 1), 255, dtype=np.uint8)


def strokes_to_png(strokes, linewidth=1.5):
    """Render a stroke array (N,3) — (dx,dy,pen_lift) — to PNG bytes."""
    pos = np.cumsum(strokes[:, :2], axis=0)
    prev_ind = 0

    W = float(np.max(pos[:, 0]) - np.min(pos[:, 0])) + 1e-6
    H = float(np.max(pos[:, 1]) - np.min(pos[:, 1])) + 1e-6
    fig_w = float(np.clip(10.0 * W / H, 6.0, 18.0))

    fig, ax = plt.subplots(figsize=(fig_w, 4))
    for ind, val in enumerate(strokes[:, 2]):
        if val > 0.5:
            if ind > prev_ind:
                ax.plot(
                    pos[prev_ind:ind, 0],
                    pos[prev_ind:ind, 1],
                    color="black",
                    linewidth=linewidth,
                )
            prev_ind = ind
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def run_generation(
    text, style_vector, char_spacing, word_spacing, italics, boldness, seqlen=None
):
    """Run diffusion → apply style controllers → return (raw, styled, linewidth)."""
    import tensorflow as tf
    import utils
    import style_controller

    model = load_model()
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()

    if not seqlen or seqlen <= 0:
        seqlen = len(text) * 16
    # Sequence length must be divisible by 8
    seqlen = seqlen - (seqlen % 8) + 8

    # 1. Base generation (all controls zeroed inside run_batch_inference)
    raw = utils.run_batch_inference(
        model,
        beta_set,
        text,
        style_vector,
        tokenizer=tokenizer,
        time_steps=seqlen,
        diffusion_mode="new",
        show_samples=False,
        path=None,
        char_spacing=0.0,
        word_spacing=0.0,
        italics=0.0,
        boldness=0.0,
    )[0]

    # 2. Mathematical style controllers
    config = style_controller.StyleConfig()
    styled, lw = style_controller.apply_all_styles(
        raw,
        text,
        char_spacing=char_spacing,
        word_spacing=word_spacing,
        italics=italics,
        boldness=boldness,
        config=config,
    )
    return raw, styled, lw


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("✍️ Handwriting Style Transfer")
st.caption(
    "Diffusion-based handwriting generation with four mathematical style controllers: "
    "character spacing, word spacing, italics (slant) and boldness."
)

ensure_weights()

with st.sidebar:
    st.header("📝 Input")
    text = st.text_input("Text to generate", value="Hello World", max_chars=200)

    st.header("🎨 Style Controls")
    char_spacing = st.slider(
        "Character spacing", -1.0, 1.0, 0.0, 0.05,
        help="-1 = tighter, 0 = original, +1 = wider"
    )
    word_spacing = st.slider(
        "Word spacing", -1.0, 1.0, 0.0, 0.05,
        help="-1 = tighter, 0 = original, +1 = wider"
    )
    italics = st.slider(
        "Italics (slant)", -1.0, 1.0, 0.0, 0.05,
        help="-1 = backward slant, 0 = upright, +1 = forward slant"
    )
    boldness = st.slider(
        "Boldness", -1.0, 1.0, 0.0, 0.05,
        help="-1 = thinner stroke, 0 = original, +1 = thicker stroke"
    )

    st.header("⚙️ Options")
    seqlen = st.number_input(
        "Sequence length (0 = auto)",
        min_value=0,
        max_value=4000,
        value=0,
        step=8,
        help="Auto = 16 × len(text), rounded up to a multiple of 8.",
    )
    writer_file = st.file_uploader(
        "Writer style reference (optional)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )
    compare = st.checkbox("Show original alongside styled", value=True)
    generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)

# Session state for the last result so the download buttons survive reruns
if "result" not in st.session_state:
    st.session_state.result = None


if generate_btn:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            with st.spinner("Generating handwriting — this can take a minute…"):
                style_extractor = load_style_extractor()

                if writer_file is not None:
                    writer_img = read_writer_image(writer_file)
                else:
                    writer_img = default_writer_image()

                import tensorflow as tf
                style_vec = style_extractor(tf.expand_dims(writer_img, 0))

                raw, styled, lw = run_generation(
                    text,
                    style_vec,
                    char_spacing,
                    word_spacing,
                    italics,
                    boldness,
                    seqlen if seqlen > 0 else None,
                )

                st.session_state.result = {
                    "orig_png": strokes_to_png(raw, linewidth=1.5),
                    "mod_png": strokes_to_png(styled, linewidth=lw),
                    "text": text,
                    "compare": compare,
                }
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.exception(e)

# -------- Render results --------
if st.session_state.result is not None:
    res = st.session_state.result

    if res["compare"]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(res["orig_png"], use_container_width=True)
            st.download_button(
                "📥 Download original",
                data=res["orig_png"],
                file_name="handwriting_original.png",
                mime="image/png",
                use_container_width=True,
            )
        with col2:
            st.subheader("Styled")
            st.image(res["mod_png"], use_container_width=True)
            st.download_button(
                "📥 Download styled",
                data=res["mod_png"],
                file_name="handwriting_styled.png",
                mime="image/png",
                use_container_width=True,
            )
    else:
        st.subheader("Generated")
        st.image(res["mod_png"], use_container_width=True)
        st.download_button(
            "📥 Download image",
            data=res["mod_png"],
            file_name="handwriting.png",
            mime="image/png",
            use_container_width=True,
        )
else:
    st.info(
        "Configure the controls on the left and click **✨ Generate** to create handwriting."
    )
