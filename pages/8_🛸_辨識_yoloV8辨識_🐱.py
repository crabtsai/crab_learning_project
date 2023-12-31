import streamlit as st
from skimage import io
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt  # 新增這一行
from ultralytics.utils.plotting import Annotator
from PIL import Image #1024

st.title("上傳圖片辨識")
st.info("YOLO_V8)， 80 個類別的物體檢測，這些類別包括人，動物，交通工具，家具等")

uploaded_file = st.file_uploader("上傳圖片", type=['jpg', 'jpeg', 'png', 'gif'])
colors_rgb = {str(i): ((i * 30) % 256, (i * 50) % 256, (i * 70) % 256) for i in range(30)}

if uploaded_file is not None:
      
    image = io.imread(uploaded_file)
    if image.shape[-1] == 4:  # 如果通道數為4，通常是带有alpha通道的圖像
        image = image[:, :, :3]  # 去除alpha通道
    
    # 載入模型
    pretrained_weights_path = './model/yolov8n.pt'
    yolo = YOLO()
    yolo = yolo.load(pretrained_weights_path)
    try:
        results = yolo.predict(image)  #預測指令
        # 顯示結果
        st.subheader("檢測結果")
        colors_count = 0
        result_str = []
        # 顯示物件類別
        print(results[0].boxes.cls)
        # # print()
        for i, result in enumerate(results):
            annotator = Annotator(image)
            boxes = result.boxes
            for box in boxes:
                cls = box.cls
                xyxy = box.xyxy[0]
                result_str_1 = yolo.names[int(cls[0])]
                result_str.append(result_str_1)
                annotator.box_label(xyxy,yolo.names[int(cls[0])],colors_rgb[f'{colors_count}'])
                colors_count += 1
        img = annotator.result()
        # 顯示標註後的圖片
        st.image(img, caption="檢測結果", use_column_width=True)
        st.write(result_str)
        
    except Exception as e:
        print(f"Error during prediction: {e}")

