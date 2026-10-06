
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import urllib.request


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

# GitHub Release se AI model download hoga
MODEL_URL = (
    "https://github.com/hassnainbilal/chest-xray-app/"
    "releases/download/v1.0.0/chest_xray_cnn_model_best.keras"
)

# Model temporary/local app folder mein save hoga
MODEL_PATH = Path(__file__).parent / "chest_xray_cnn_model_best.keras"


# =========================================================
# DOWNLOAD MODEL
# =========================================================

def download_model():

    if MODEL_PATH.exists():
        return

    st.info("Downloading AI model from GitHub Release...")

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            str(MODEL_PATH)
        )

    except Exception as e:
        st.error("Unable to download the AI model.")
        st.code(str(e))
        st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    download_model()

    try:
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )

        return model

    except Exception as e:
        st.error("Unable to load the AI model.")
        st.exception(e)
        st.stop()


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# MAKE PREDICTION
# =========================================================

def predict_image(image):

    model = load_model()

    processed_image = preprocess_image(
        image
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    return prediction


# =========================================================
# EXTRACT PREDICTION VALUE
# =========================================================

def extract_prediction_value(prediction):

    if isinstance(
        prediction,
        (list, tuple)
    ):

        if len(prediction) == 0:
            raise ValueError(
                "The model returned an empty prediction."
            )

        prediction = prediction[0]

    array = np.asarray(
        prediction
    )

    array = np.squeeze(
        array
    )

    if array.ndim == 0:
        return float(
            array.item()
        )

    if array.size == 1:
        return float(
            array.reshape(-1)[0]
        )

    flat_array = array.reshape(-1)

    if flat_array.size == 2:
        return float(
            flat_array[1]
        )

    return float(
        np.max(flat_array)
    )


# =========================================================
# USER INTERFACE
# =========================================================

st.title(
    "🩻 Chest X-Ray AI"
)

st.write(
    "Upload a chest X-ray image to get an AI model prediction."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# SHOW IMAGE
# =========================================================

if uploaded_file:

    try:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded X-ray",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            "Unable to open the uploaded image."
        )

        st.exception(e)

        st.stop()


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "🔍 Predict",
        type="primary"
    ):

        with st.spinner(
            "Analyzing chest X-ray..."
        ):

            try:

                prediction = predict_image(
                    image
                )

                value = extract_prediction_value(
                    prediction
                )

            except Exception as e:

                st.error(
                    "An error occurred while making the prediction."
                )

                st.write(
                    "Prediction type:"
                )

                try:
                    st.code(
                        str(type(prediction))
                    )
                except:
                    pass

                try:

                    st.write(
                        "Prediction shape:"
                    )

                    st.code(
                        str(
                            np.asarray(
                                prediction
                            ).shape
                        )
                    )

                except:
                    pass

                st.exception(
                    e
                )

                st.stop()


        # =================================================
        # BINARY CLASSIFICATION
        # =================================================

        if 0 <= value <= 1:

            if value >= 0.5:

                label = (
                    "Positive / Abnormal"
                )

                confidence = value

            else:

                label = (
                    "Negative / Normal"
                )

                confidence = 1 - value


            # =================================================
            # RESULT
            # =================================================

            if value >= 0.5:

                st.error(
                    f"Prediction: {label}"
                )

            else:

                st.success(
                    f"Prediction: {label}"
                )


            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        # =================================================
        # OTHER MODEL OUTPUT
        # =================================================

        else:

            st.success(
                f"Prediction value: {value:.4f}"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "---"
)

st.caption(
    "Chest X-Ray AI • CNN-based image classification"
)
