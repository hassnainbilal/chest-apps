import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import urllib.request
import json
import os

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩻",
    layout="centered"
)

# =========================================================
# MODEL SETTINGS
# =========================================================
# Your local project folder
PROJECT_DIR = Path(r"C:\Users\Hp\Desktop\chest-xray-streamlit")

MODEL_NAME = "chest_xray_cnn_model_best.keras"
MODEL_PATH = PROJECT_DIR / MODEL_NAME

# GitHub release information
GITHUB_API_URL = (
    "https://api.github.com/repos/"
    "hassnainbilal/chest-apps/releases/tags/v1.0.0"
)

# =========================================================
# DOWNLOAD MODEL FROM GITHUB RELEASE
# =========================================================
def download_model():
    # 1. Use local model if it already exists.
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10 * 1024 * 1024:
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    st.info("⬇️ AI model not found locally. Finding the model in GitHub Release...")

    try:
        # Get the release information from GitHub.
        request = urllib.request.Request(
            GITHUB_API_URL,
            headers={
                "User-Agent": "chest-xray-streamlit-app",
                "Accept": "application/vnd.github+json"
            }
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            release = json.loads(response.read().decode("utf-8"))

        assets = release.get("assets", [])

        # Find the Keras model asset.
        model_asset = None

        for asset in assets:
            name = asset.get("name", "")
            if name == MODEL_NAME:
                model_asset = asset
                break

        # If exact name is not present, find the first .keras asset.
        if model_asset is None:
            for asset in assets:
                name = asset.get("name", "")
                if name.lower().endswith(".keras"):
                    model_asset = asset
                    break

        if model_asset is None:
            asset_names = [a.get("name", "") for a in assets]
            raise FileNotFoundError(
                "No .keras model was found in GitHub Release v1.0.0. "
                f"Release assets found: {asset_names}"
            )

        download_url = model_asset.get("browser_download_url")

        if not download_url:
            raise ValueError("GitHub did not provide a download URL for the model asset.")

        st.info(f"Downloading: {model_asset.get('name')}")

        # Download the actual Release asset.
        urllib.request.urlretrieve(
            download_url,
            str(MODEL_PATH)
        )

    except Exception as e:
        if MODEL_PATH.exists():
            try:
                MODEL_PATH.unlink()
            except Exception:
                pass

        st.error("❌ Unable to download the AI model.")
        st.code(str(e))
        st.info(
            "If you are running locally, put "
            f"'{MODEL_NAME}' inside:\n{PROJECT_DIR}"
        )
        st.stop()

    # Verify the downloaded file.
    if not MODEL_PATH.exists():
        st.error("❌ Model file was not found after download.")
        st.stop()

    file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

    if file_size_mb < 10:
        st.error(
            f"❌ Downloaded model appears invalid. "
            f"File size: {file_size_mb:.2f} MB"
        )
        try:
            MODEL_PATH.unlink()
        except Exception:
            pass
        st.stop()

    st.success(f"✅ AI model ready ({file_size_mb:.2f} MB)")


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    download_model()

    try:
        return tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )
    except Exception as e:
        st.error("❌ Unable to load the AI model.")
        st.exception(e)
        st.stop()


# =========================================================
# IMAGE PREPROCESSING
# =========================================================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# =========================================================
# PREDICTION
# =========================================================
def predict_image(image):
    model = load_model()

    processed_image = preprocess_image(image)

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    return prediction


# =========================================================
# USER INTERFACE
# =========================================================
st.title("🩻 Chest X-Ray AI")
st.write(
    "Upload a chest X-ray image to get an AI model prediction."
)

uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    try:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded X-ray",
            use_container_width=True
        )

    except Exception as e:
        st.error("❌ Unable to open the uploaded image.")
        st.exception(e)
        st.stop()

    if st.button("🔍 Predict", type="primary"):

        with st.spinner("Analyzing chest X-ray..."):

            try:
                prediction = predict_image(image)
                probabilities = np.asarray(prediction)[0]

                if probabilities.size != 2:
                    raise ValueError(
                        f"Expected 2 model outputs, but got {probabilities.size}."
                    )

                predicted_class = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_class])

            except Exception as e:
                st.error("❌ Error while making prediction.")
                st.exception(e)
                st.stop()

        class_names = ["NORMAL", "PNEUMONIA"]
        label = class_names[predicted_class]

        st.markdown("---")

        if label == "NORMAL":
            st.success(f"✅ Prediction: {label}")
        else:
            st.error(f"⚠️ Prediction: {label}")

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.write("### Class Probabilities")
        st.write(f"Normal: {probabilities[0] * 100:.2f}%")
        st.write(f"Pneumonia: {probabilities[1] * 100:.2f}%")

st.markdown("---")
st.caption("Chest X-Ray AI • CNN-based image classification")
st.warning(
    "This AI prediction is for educational/research purposes "
    "and should not be used as a medical diagnosis."
)
