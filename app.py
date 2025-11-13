import streamlit as st
from PIL import Image
import io
import base64

# Cấu hình trang
st.set_page_config(
    page_title="Medical Image Captioning",
    page_icon="🏥",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Hàm giả lập tạo caption (bạn có thể thay thế bằng model thực tế)
def generate_caption(image, model_name):
    """
    Hàm này sẽ được thay thế bằng model thực tế
    """
    captions = {
        "BLIP": f"A medical image showing anatomical structures. Model: {model_name}",
        "ViT-GPT2": f"This appears to be a diagnostic medical scan with various features. Model: {model_name}",
        "CheXNet": f"Medical imaging study demonstrating clinical findings. Model: {model_name}",
        "Custom Model": f"Image analysis complete using {model_name}."
    }
    return captions.get(model_name, "Unable to generate caption for this image.")

# Header
st.markdown('<div class="main-header">🏥 Medical Image Captioning Chatbot</div>', unsafe_allow_html=True)

# Sidebar - Model selection
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    selected_model = st.selectbox(
        "Chọn mô hình:",
        ["BLIP", "ViT-GPT2", "CheXNet", "Custom Model"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Thông tin mô hình")
    
    model_info = {
        "BLIP": "Bootstrapping Language-Image Pre-training - Model đa năng cho image captioning",
        "ViT-GPT2": "Vision Transformer kết hợp GPT-2 - Tạo mô tả chi tiết",
        "CheXNet": "Chuyên về phân tích X-quang ngực",
        "Custom Model": "Mô hình tùy chỉnh của bạn"
    }
    
    st.info(model_info[selected_model])
    
    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.chat_history = []
        st.session_state.current_image = None
        st.rerun()

# Khu vực chat history
chat_container = st.container()

with chat_container:
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['type'] == 'user':
                st.markdown(f'<div class="chat-message user-message"><b>👤 Bạn:</b><br/>{message["content"]}</div>', unsafe_allow_html=True)
                if message.get('image'):
                    st.image(message['image'], width=300)
            else:
                st.markdown(f'<div class="chat-message bot-message"><b>🤖 Bot ({message.get("model", "Unknown")}):</b><br/>{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.info("👋 Xin chào! Tôi là chatbot image captioning. Hãy tải ảnh và tôi sẽ tạo mô tả cho bạn.")

# Khu vực input ở dưới cùng
st.markdown("---")
st.markdown("### 📤 Tải ảnh và gửi")

# Tạo 3 cột: nút chọn model, upload file, nút enter
col1, col2, col3 = st.columns([2, 6, 1])

with col1:
    st.markdown(f"**Mô hình:** {selected_model}")

with col2:
    uploaded_file = st.file_uploader(
        "Chọn ảnh y tế",
        type=['png', 'jpg', 'jpeg', 'bmp', 'dicom'],
        key="image_uploader",
        label_visibility="collapsed"
    )

with col3:
    submit_button = st.button("➡️", help="Gửi ảnh để tạo caption")

# Xử lý khi người dùng gửi ảnh
if submit_button and uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    st.session_state.current_image = image
    
    # Thêm tin nhắn của user vào chat history
    st.session_state.chat_history.append({
        'type': 'user',
        'content': f'Đã tải lên ảnh: {uploaded_file.name}',
        'image': image
    })
    
    # Tạo caption
    with st.spinner(f'Đang phân tích ảnh với mô hình {selected_model}...'):
        caption = generate_caption(image, selected_model)
    
    # Thêm phản hồi của bot
    st.session_state.chat_history.append({
        'type': 'bot',
        'content': caption,
        'model': selected_model
    })
    
    st.rerun()

elif submit_button and uploaded_file is None:
    st.warning("⚠️ Vui lòng chọn một ảnh trước khi gửi!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Medical Image Captioning Chatbot v1.0 | Powered by Streamlit</div>",
    unsafe_allow_html=True
)
