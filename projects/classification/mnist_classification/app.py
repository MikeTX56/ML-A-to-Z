import streamlit as st
import numpy as np
from joblib import load
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Canvas", layout="centered")
st.title("Write on Canvas")

MODEL_DEFAULT = "knn_model.joblib"


@st.cache_resource
def load_model(model_path: str):
    return load(model_path)


def _shift_to_center(arr: np.ndarray) -> np.ndarray:
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return arr
    cx, cy = xs.mean(), ys.mean()
    shift_x = int(round(13.5 - cx))
    shift_y = int(round(13.5 - cy))

    h, w = arr.shape
    shifted = np.zeros_like(arr)
    x_from = max(0, -shift_x)
    x_to = min(w, w - shift_x)
    y_from = max(0, -shift_y)
    y_to = min(h, h - shift_y)
    shifted[y_from + shift_y : y_to + shift_y, x_from + shift_x : x_to + shift_x] = arr[
        y_from:y_to, x_from:x_to
    ]
    return shifted


def preprocess_canvas(image_data: np.ndarray) -> np.ndarray:
    img = Image.fromarray(image_data.astype("uint8"), mode="RGBA")
    img = img.convert("L")
    img = Image.fromarray(255 - np.array(img))

    arr = np.array(img)
    arr = (arr > 20).astype("uint8") * 255

    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 28 * 28), dtype="float32")

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    cropped = img.crop((left, top, right + 1, bottom + 1))

    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(20 * h / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(20 * w / h)))
    resized = cropped.resize((new_w, new_h), resample=Image.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    centered = _shift_to_center(np.array(canvas))
    out = centered.astype("float32") / 255.0
    return out.reshape(1, -1)


with st.sidebar:
    st.subheader("Canvas Settings")
    stroke_width = st.slider("Stroke width", 2, 20, 8)
    canvas_size = st.slider("Canvas size", 200, 600, 400, step=50)
    model_path = st.text_input("Model path", value=MODEL_DEFAULT)

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)

with col1:
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data, caption="Canvas", clamp=True)

with col2:
    if canvas_result.image_data is not None:
        processed = preprocess_canvas(canvas_result.image_data)
        st.image(processed.reshape(28, 28), caption="28x28", clamp=True)

if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Draw a digit first.")
    else:
        model_file = Path(model_path)
        if not model_file.exists():
            st.error(f"Model not found: {model_file}")
        else:
            model = load_model(str(model_file))
            processed = preprocess_canvas(canvas_result.image_data)
            pred = model.predict(processed)[0]
            st.success(f"Prediction: {pred}")