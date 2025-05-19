# Giao_diện_garbage.py
import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn as nn
import timm
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import datetime
import os
import base64

# --- Giao diện nền trắng + chữ nổi ---
def set_white_background_and_styles():
    st.markdown("""
        <style>
        .stApp {
            background-color: white;
            font-family: 'Segoe UI', sans-serif;
            color: #111 !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, .css-1cpxqw2, .css-16idsys, .css-q8sbsg, .css-1v0mbdj {
            color: #111 !important;
        }
        .highlight-Fabric, .highlight-Glass, .highlight-Non-recyclable, 
        .highlight-Paper, .highlight-Recyclable-inorganic {
            color: #000 !important;
            animation: glow 1s ease-in-out infinite alternate;
        }
        @keyframes glow {
            0% { text-shadow: 0 0 5px #aaa; }
            100% { text-shadow: 0 0 20px #555; }
        }
        section[data-testid="stFileUploader"] div {
            background-color: #f0f0f0 !important;
            color: #111 !important;
            border: 1px solid #ccc !important;
        }
        .history-item {
    color: #111 !important;
    font-size: 16px;
    white-space: pre-line;
}

        </style>
    """, unsafe_allow_html=True)



set_white_background_and_styles()

# -------------------- Mô hình và xử lý ảnh ---------------------
base_model = timm.create_model("legacy_xception", pretrained=True)
base_model.global_pool = nn.Identity()
base_model.fc = nn.Identity()
for param in base_model.parameters():
    param.requires_grad = False
in_features = base_model.num_features

class XceptionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(XceptionModel, self).__init__()
        base_model = timm.create_model("legacy_xception", pretrained=True)
        base_model.global_pool = nn.Identity()
        base_model.fc = nn.Identity()
        for param in base_model.parameters():
            param.requires_grad = False
        in_features = base_model.num_features
        self.base = base_model
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=512)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XceptionModel(num_classes=5)
checkpoint = torch.load('C:\\Users\\OS\\Downloads\\Project_garbage\\Project_garbage\\models_trained\\best_cnn.pt', map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

class_names = ['Fabric', 'Glass', 'Non-recyclable', 'Paper', 'Recyclable-inorganic']
class_info = {
    'Fabric': "Vải vóc, quần áo cũ...",
    'Glass': "Chai thủy tinh, lọ...",
    'Non-recyclable': "Không thể tái chế, rác thải sinh hoạt...",
    'Paper': "Giấy báo, giấy in...",
    'Recyclable-inorganic': "Nhựa, lon, kim loại không phân hủy..."
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        score, pred = torch.max(probs, 1)
        label = class_names[pred.item()]
        return label, score.item(), class_info[label]

def save_history(label, score):
    folder = "history"
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "history.txt"), "a", encoding="utf-8") as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{now} - {label} - {round(score * 100, 2)}%\n")

# -------------------- Giao diện Streamlit ---------------------
st.title("♻️ Hệ thống phân loại rác thải thông minh")

option = st.radio("Chọn nguồn ảnh:", ["Tải ảnh", "Sử dụng webcam"])

if option == "Tải ảnh":
    uploaded_file = st.file_uploader("Tải ảnh rác cần phân loại", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh đầu vào", use_column_width=True)
        label, score, description = predict(image)
        st.markdown(f"""
        <div class='highlight-{label}' style="font-size:32px; font-weight:bold; text-align:center; margin-top:20px;">
            {label}
        </div>
        """, unsafe_allow_html=True)
        save_history(label, score)

elif option == "Sử dụng webcam":
    st.write("📷 Mở webcam để phân loại trực tiếp")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            label, score, description = predict(pil_image)
            cv2.putText(img, f"{label} ({round(score * 100, 1)}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            save_history(label, score)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="realtime",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )

# -------------------- Hiển thị lịch sử ---------------------
st.markdown("### 📝 Lịch sử dự đoán")
if os.path.exists("history/history.txt"):
    with open("history/history.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        history = "".join(lines[-10:])
    styled_history = "<br>".join(line.strip() for line in lines[-10:])
    st.markdown(f"<div class='history-item'>{styled_history}</div>", unsafe_allow_html=True)
else:
    st.write("Chưa có lịch sử.")
