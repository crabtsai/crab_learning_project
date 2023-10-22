import streamlit as st
from skimage import io
from skimage.transform import resize
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# 加载模型
model = tf.keras.models.load_model('./model/yolov8n.pt')

st.title("上傳圖片辨識")
st.info("訓練模型(YOLO_V8)")

uploaded_file = st.file_uploader("上傳圖片(.png)", type=['png', 'jpg'])
if uploaded_file is not None:
    # 读取上传的图像并调整大小
    image = io.imread(uploaded_file)
    if image.shape[-1] == 4:  # 如果通道数为4，通常是带有alpha通道的图像
        image = image[:, :, :3]  # 去除alpha通道

    # 显示原始上传图像
    st.image(image, caption="上傳的圖像", use_column_width=True)

    # 使用 YOLO 进行对象检测
    yolo = YOLO("./model/yolov8n.pt")
    results = yolo.predict(image)  # 对图像进行预测

    # 显示检测结果
    st.subheader("檢測結果")
    for result in results.xyxy[0]:
        st.write(f"類別: {result[5]}, 置信度: {result[4]*100:.2f}%")
        st.image(result[0:4], caption="檢測結果", use_column_width=True)

