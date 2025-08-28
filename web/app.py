import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path

CLASS_NAMES = ["Normal", "Pneumonia"]     # 0 = Normal, 1 = Pneumonia
TARGET_SIZE = (224, 224)
MODEL_PATH  = Path("model/chest_xray_adanmodel.h5")

@st.cache_resource(show_spinner="🔄 Đang tải mô hình…")
def load_model(path: Path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

st.set_page_config(page_title="X‑ray Pneumonia Detector", page_icon="🩺")
st.title("🩺 Pneumonia Identification System")
uploaded = st.file_uploader("Chọn ảnh X‑ray", type=["jpg", "jpeg", "png"])

def preprocess(img_pil: Image.Image) -> np.ndarray:
    """Resize *không crop*, chuẩn hoá 0‑1, RGB 3‑kênh."""
    img = img_pil.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0     
    return arr[np.newaxis, ...]                       

if uploaded is None:
    st.info("👈 Vui lòng tải ảnh trước.")
else:
    img_pil = Image.open(uploaded)
    st.image(img_pil, caption="Ảnh đã tải", use_column_width=True)

    prob  = float(model.predict(preprocess(img_pil))[0][0])  # sigmoid
    label = "Pneumonia" if prob >= 0.65 else "Normal"
    conf  = prob*100 if label=="Pneumonia" else (1-prob)*100

    st.subheader(f"Kết quả: **{label}**")
    st.write(f"Độ tin cậy: **{conf:.2f}%**")
    with st.expander("Chi tiết đầu ra"):
        st.json({"Pneumonia": f"{prob*100:.2f}%",
                 "Normal":    f"{(1-prob)*100:.2f}%"})
