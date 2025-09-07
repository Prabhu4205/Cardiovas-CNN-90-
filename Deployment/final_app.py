import streamlit as st
from Ecg import ECG
from PIL import Image
import numpy as np

MODEL_PATH = "CNN_ECG(90).h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
INPUT_LENGTH = 255  # Matches the new 255-feature input

# Initialize ECG object
ecg = ECG(MODEL_PATH, LABEL_ENCODER_PATH, input_length=INPUT_LENGTH)

st.title("Cardiovascular Disease Prediction from ECG Image")

uploaded_file = st.file_uploader("Choose an ECG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))
    st.image(img, caption="Uploaded ECG Image", use_container_width=True)

    # Convert to gray
    gray_img = ecg.GrayImgae(img)
    with st.expander("Gray Scale Image"):
        st.image(gray_img, use_container_width=True)

    # Divide leads and plot
    leads = ecg.DividingLeads(img)
    with st.expander("Dividing Leads"):
        st.image('Leads_1-12_figure.png', use_container_width=True)

    # Preprocess leads
    ecg.PreprocessingLeads(leads)
    with st.expander("Preprocessed Leads"):
        st.image('Preprocessed_Leads_1-12_figure.png', use_container_width=True)

    # Extract features and predict
    prediction = ecg.predict(uploaded_file)
    with st.expander("Prediction"):
        st.success(prediction)
