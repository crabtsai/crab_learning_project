import streamlit as st
from skimage import io
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt  # 新增這一行
from ultralytics.utils.plotting import Annotator
st.title("上傳圖片辨識")
st.info("訓練模型(YOLO_V8)")

uploaded_file = st.file_uploader("上傳圖片(.png)", type=['png', 'jpg'])
if uploaded_file is not None:
    # 读取上传的图像并调整大小
    image = io.imread(uploaded_file)
    if image.shape[-1] == 4:  # 如果通道数为4，通常是带有alpha通道的图像
        image = image[:, :, :3]  # 去除alpha通道
    
    # 使用 YOLO 进行对象检测
    pretrained_weights_path = './model/yolov8n.pt'
    
    yolo = YOLO()
    yolo = yolo.load(pretrained_weights_path)
    try:
        # 這裡可能不需要再次加載權重，因為上面已經在構造函數中指定了
        results = yolo.predict(image)  # 对图像进行预测
        # 显示检测结果
        st.subheader("檢測結果")
        
        result_skr = None
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
                annotator.box_label(xyxy,yolo.names[int(cls[0])],(50,125,50))
        img = annotator.result()
        # 顯示標註後的圖片

        st.image(img, caption="檢測結果", use_column_width=True)
        st.write(result_str)
        
    except Exception as e:
        print(f"Error during prediction: {e}")

