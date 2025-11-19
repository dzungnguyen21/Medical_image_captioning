import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from models_registry import MODEL_REGISTRY
import time

# --- CONSTANTS ---
FIXED_PROMPT = "Describe this medical image in detail."

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    /* --- GLOBAL VARS --- */
    :root {
        --header-height: 3.5rem;
    }

    /* --- HIDE STREAMLIT UI ELEMENTS --- */
    .stDeployButton {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    footer {visibility: hidden;}

    /* Make the default header transparent so our sticky header looks integrated */
    header[data-testid="stHeader"] {
        background-color: transparent;
        z-index: 1;
    }

    /* --- STICKY HEADER --- */
    .sticky-header {
        position: sticky;
        /* We stick it to the top.
           Note: Streamlit's native header takes up approx 3rem.
           If we want it to stick 'under' that or 'over' it, we adjust top.
           Setting top: 0 usually works if we hide the decoration. */
        top: 0;
        z-index: 1000;
        background-color: var(--background-color);
        padding: 1rem 0;
        text-align: center;
        border-bottom: 1px solid var(--secondary-background-color);
        margin-bottom: 1rem;
        width: 100%;
    }

    .header-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f77b4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* --- CHAT BUBBLES --- */
    .stChatMessage { gap: 1rem; }

    .entity-label {
        font-weight: 600;
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }

    /* --- HERO / EMPTY STATE --- */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 4rem;
        text-align: center;
        opacity: 0.8;
    }
    .hero-icon { font-size: 5rem; margin-bottom: 1rem; }
    .hero-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; }

    /* --- LAYOUT TWEAKS --- */
    /* Add padding to top so sticky header doesn't overlap the first message immediately */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SCROLL SCRIPT (ANCHOR METHOD) ---
# This is the most reliable way to scroll in Streamlit.
# We inject this script which finds the 'end-of-chat' div and scrolls to it.
def scroll_to_bottom():
    js = """
    <script>
        var element = window.parent.document.getElementById("end-of-chat");
        if (element) {
            element.scrollIntoView({behavior: "smooth", block: "end"});
        }
    </script>
    """
    components.html(js, height=0, width=0)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_key):
    model_class = MODEL_REGISTRY[model_key]
    return model_class()

def get_response(image, model_key):
    model = load_model(model_key)
    try:
        if model.requires_prompt:
            return model.run(image, prompt=FIXED_PROMPT)
        else:
            return model.run(image)
    except Exception as e:
        return f"Error processing image: {str(e)}"

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    selected_model = st.selectbox("Select AI Model", list(MODEL_REGISTRY.keys()), index=0)
    current_model = load_model(selected_model)
    st.info(f"**Model:** {selected_model}\n\n{current_model.description}")
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.uploader_key += 1
        st.rerun()

# --- RENDER STICKY HEADER ---
# Visible only when chatting
if st.session_state.messages:
    st.markdown("""
        <div class='sticky-header'>
            <span class='header-text'>🏥 Medical Assistant</span>
        </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---

# 1. Empty State
if not st.session_state.messages:
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-icon'>🏥</div>
        <div class='hero-title'>Medical Imaging Assistant</div>
        <p>Select a model from the sidebar and attach a scan to start.</p>
    </div>
    """, unsafe_allow_html=True)

# 2. Chat History
for role, content, msg_type, entity_name in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(f"<div class='entity-label'>{entity_name}</div>", unsafe_allow_html=True)
        if msg_type == "image":
            st.image(content, width=300)
        else:
            st.markdown(content)

# 3. Logic Loop (Processing)
if st.session_state.messages and st.session_state.messages[-1][0] == "user":
    user_image = st.session_state.messages[-1][1]

    with st.chat_message("assistant"):
        st.markdown(f"<div class='entity-label'>{selected_model}</div>", unsafe_allow_html=True)
        with st.spinner("Analyzing..."):
            response_text = get_response(user_image, selected_model)
            st.markdown(response_text)

    st.session_state.messages.append(("assistant", response_text, "text", selected_model))
    st.session_state.uploader_key += 1
    st.rerun()

# 4. Input Area
st.divider()

col1, col2 = st.columns([0.85, 0.15], vertical_alignment="bottom")

with col1:
    uploaded_file = st.file_uploader(
        "Attachment",
        type=["png", "jpg", "jpeg", "bmp"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

with col2:
    # Button is always visible, but disabled if uploaded_file is None
    send_clicked = st.button(
        "Send",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None)
    )

# 5. Handle Send Click
if uploaded_file and send_clicked:
    image = Image.open(uploaded_file)
    st.session_state.messages.append(("user", image, "image", "USER"))
    st.rerun()

# --- SCROLL ANCHOR ---
# We place an invisible div here. The JS will scroll to this div.
st.markdown('<div id="end-of-chat"></div>', unsafe_allow_html=True)

# 6. Trigger Auto Scroll
scroll_to_bottom()
