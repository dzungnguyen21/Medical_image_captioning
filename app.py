import streamlit as st
from PIL import Image
from models import MODELS
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🏥",
)

# --- CSS ---
st.markdown("""
<style>
body { background-color: #fafafa; }

.title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    font-size: 1rem;
    color: #666;
    margin-bottom: 2rem;
}

.start-container {
    max-width: 450px;
    margin-left: auto;
    margin-right: auto;
}

.stButton>button {
    background-color: #1f77b4 !important;
    color: white !important;
    border-radius: 8px !important;
    height: 3rem !important;
    font-size: 1.1rem !important;
}

.result-text {
    max-width: 400px;
    font-size: 1.1rem;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 3rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = None
if "caption" not in st.session_state:
    st.session_state.caption = None
if "model_selected" not in st.session_state:
    st.session_state.model_selected = list(MODELS.keys())[0]


# --- MODEL LOADING ---
@st.cache_resource
def load_hf_model(model_key):
    hf_name = MODELS[model_key]["hf_name"]
    if hf_name is None:
        return None
    return pipeline("image-to-text", model=hf_name, trust_remote_code=True)


def generate_caption(image, model_key):
    model = load_hf_model(model_key)
    if model is None:
        return f"(Model '{model_key}' chưa được triển khai.)"
    try:
        result = model(image)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Model error: {e}"


# --- HEADER ---
st.markdown("<div class='title'>Image Captioning</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image and see what the model thinks it is</div>", unsafe_allow_html=True)


# --- START SCREEN ---
def start_screen():
    st.set_page_config(layout="centered")
    with st.container(horizontal_alignment="center"):
        st.session_state.model_selected = st.selectbox(
            "Select model",
            list(MODELS.keys()),
            index=list(MODELS.keys()).index(st.session_state.model_selected)
        )
        st.info(MODELS[st.session_state.model_selected]["description"])

        uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp"])
        if uploaded_file:
            st.session_state.image_uploaded = Image.open(uploaded_file)
            st.session_state.caption = "thinking..."
            st.rerun()


# --- RESULT SCREEN ---
def result_screen():
    st.set_page_config(layout="wide")
    image = st.session_state.image_uploaded
    model_name = st.session_state.model_selected

    col1, col2, col3 = st.columns([3, 1, 3], vertical_alignment="center")

    with col1:
        st.image(image, caption="Uploaded Image")

    with col2:
        st.markdown('<div style="font-size: 3rem; text-align: center">➡️</div>', unsafe_allow_html=True)

    with col3:
        if st.session_state.caption == "thinking...":
            with st.spinner("Model is thinking..."):
                st.session_state.caption = generate_caption(image, model_name)
        st.markdown(f"<div class='result-text'><b style='font-size: 2rem'>{model_name} thinks it is...</b><br>{st.session_state.caption}</div>", unsafe_allow_html=True)

    # Buttons
    button_container = st.container(horizontal_alignment="center")
    with button_container:
        if st.button("Try another image"):
            st.session_state.image_uploaded = None
            st.session_state.caption = None
            st.rerun()

# --- MAIN FLOW ---


if st.session_state.image_uploaded is None:
    start_screen()
else:
    result_screen()


# --- FOOTER ---
st.markdown("<div class='footer'>Medical Image Captioning v1.0 | Powered by Streamlit</div>", unsafe_allow_html=True)

