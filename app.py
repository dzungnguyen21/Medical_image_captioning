import streamlit as st
from PIL import Image
from models_registry import MODEL_REGISTRY  # your new file
from functools import partial

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🏥",
)

# --- CSS ---
st.markdown("""
<style>
body { background-color: #fafafa; }
.title { text-align: center; font-size: 2.4rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.2rem; }
.subtitle { text-align: center; font-size: 1rem; color: #666; margin-bottom: 2rem; }
.stButton>button { background-color: #1f77b4 !important; color: white !important; border-radius: 8px !important; height: 3rem !important; font-size: 1.1rem !important; }
.result-text { max-width: 400px; font-size: 1.1rem; }
.footer { text-align: center; color: gray; margin-top: 3rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = None
if "caption" not in st.session_state:
    st.session_state.caption = None
if "model_selected" not in st.session_state:
    st.session_state.model_selected = list(MODEL_REGISTRY.keys())[0]
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""


# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_key):
    model_class = MODEL_REGISTRY[model_key]
    return model_class()


def generate_caption(image, model_key, prompt=None):
    model = load_model(model_key)
    try:
        if model.requires_prompt:
            return model.run(image, prompt=prompt)
        else:
            return model.run(image)
    except Exception as e:
        return f"Model error: {e}"


# --- HEADER ---
st.markdown("<div class='title'>Image Captioning</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image and see what the model thinks it is</div>", unsafe_allow_html=True)


# --- START SCREEN ---
def start_screen():
    st.set_page_config(layout="centered")
    with st.container():
        # Model selection
        st.session_state.model_selected = st.selectbox(
            "Select model",
            list(MODEL_REGISTRY.keys()),
            index=list(MODEL_REGISTRY.keys()).index(st.session_state.model_selected)
        )
        model = load_model(st.session_state.model_selected)
        st.info(model.description)

        # Prompt if required
        if model.requires_prompt:
            st.session_state.user_prompt = st.text_input(
                "Enter prompt/question",
                value=st.session_state.user_prompt or "Describe this image"
            )

        # Upload image
        uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp"])
        if uploaded_file:
            st.session_state.image_uploaded = Image.open(uploaded_file)
            # Add button to go to result screen
            if st.button("Generate Caption"):
                st.session_state.caption = "thinking..."
                st.session_state.result_ready = True
                st.rerun()


# --- RESULT SCREEN ---
def result_screen():
    st.set_page_config(layout="wide")
    image = st.session_state.image_uploaded
    model_name = st.session_state.model_selected
    user_prompt = st.session_state.user_prompt if hasattr(st.session_state, "user_prompt") else None

    col1, col2, col3 = st.columns([3, 1, 3], vertical_alignment="center")
    with col1:
        st.image(image, caption="Uploaded Image")
    with col2:
        st.markdown('<div style="font-size: 3rem; text-align: center">➡️</div>', unsafe_allow_html=True)
    with col3:
        if st.session_state.caption == "thinking...":
            with st.spinner("Model is thinking..."):
                st.session_state.caption = generate_caption(image, model_name, prompt=user_prompt)
        st.markdown(f"<div class='result-text'><b style='font-size: 2rem'>{model_name} thinks it is...</b><br>{st.session_state.caption}</div>", unsafe_allow_html=True)

    # Button to go back
    button_container = st.container()
    with button_container:
        if st.button("Try another image"):
            st.session_state.image_uploaded = None
            st.session_state.caption = None
            st.session_state.user_prompt = ""
            st.rerun()

# --- MAIN FLOW ---
if not hasattr(st.session_state, "result_ready"):
    st.session_state.result_ready = False

if st.session_state.image_uploaded is None or not st.session_state.result_ready:
    start_screen()
else:
    result_screen()

# --- FOOTER ---
st.markdown("<div class='footer'>Medical Image Captioning v1.0 | Powered by Streamlit</div>", unsafe_allow_html=True)

