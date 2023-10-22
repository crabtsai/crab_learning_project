import streamlit as st
from skimage import io
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt  # 新增這一行

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
        
        # 使用 matplotlib 繪製檢測框
        fig, ax = plt.subplots()
        ax.imshow(image)
        
        for result in results.xyxy[0]:
            st.write(f"類別: {result[5]}, 置信度: {result[4]*100:.2f}%")
            
            # 繪製檢測框
            box = result[0:4]
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        # 顯示標註後的圖片
        st.pyplot(fig)
        
    except Exception as e:
        print(f"Error during prediction: {e}")


