import streamlit as st 
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Monitor de segurança: Detecção de EPIs", layout="wide")

st.title("Monitor de Segurança: Detecção de EPIs em Tempo Real")
st.markdown("---")

@st.cache_resource
def load_model():
   
    model = YOLO("best.pt")
    return model

model = load_model()

def process_frame(frame):

    results = model(frame, conf=0.5)
    
   
    for r in results:
        return r.plot() 
    return frame

st.header("1. Detecção em Imagem")

uploaded_file = st.file_uploader("Faça upload de uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    
  
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)

  
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    

    processed_bgr = process_frame(img_cv)
    

    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Resultado de Detecção")
        st.image(processed_rgb, caption="EPIs Detectados", use_container_width=True)

st.markdown("---")
st.header("2. Detecção em Tempo Real (Webcam)")
st.warning("Inserido em atualizações futuras")

st.markdown("---")
st.caption("Mikhael Soel")