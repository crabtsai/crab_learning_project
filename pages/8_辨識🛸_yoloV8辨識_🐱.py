import streamlit as st
from skimage import io
from skimage.transform import resize
import numpy as np
from PIL import Image, ImageDraw  # 新增這一行
from ultralytics import YOLO

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
    pretrained_weights_path = './model/yolov8n.pt'
    
    yolo = YOLO()
    yolo = yolo.load(pretrained_weights_path)
    try:
        # 這裡可能不需要再次加載權重，因為上面已經在構造函數中指定了
        results = yolo.predict(image)  # 对图像进行预测
        # 显示检测结果
        st.subheader("檢測結果")
        
        # 新增這一行，用於在圖片上繪製檢測框
        annotated_image = image.copy()  
        draw = ImageDraw.Draw(annotated_image)
        
        for result in results.xyxy[0]:
            st.write(f"類別: {result[5]}, 置信度: {result[4]*100:.2f}%")
            
            # 在圖片上繪製檢測框
            box = tuple(map(int, result[0:4]))
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=2)
            
            # st.image(result[0:4], caption="檢測結果", use_column_width=True)  # 不再需要這行
            
        # 顯示標註後的圖片
        st.image(annotated_image, caption="標註後的檢測結果", use_column_width=True)
        
    except Exception as e:
        print(f"Error during prediction: {e}")



