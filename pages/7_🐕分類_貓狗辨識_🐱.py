import streamlit as st
from skimage import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

# 定義自定義RMSprop
class CustomRMSprop(RMSprop):
    pass

# 載入模型時使用custom_objects
custom_objects = {'CustomRMSprop': CustomRMSprop}
model = tf.keras.models.load_model('./model/cats_and_dogs_new_2.h5')

st.title("上傳圖片(貓~狗)辨識")
st.info("因訓練模型(VGG-16)輸入圖片為150*150，輸入圖片狗跟貓比例占比需高")

uploaded_file = st.file_uploader("上傳圖片(.png)", type=['png','jpg'])
if uploaded_file is not None:
    # 读取上传的图像并调整大小
    image = io.imread(uploaded_file)
    if image.shape[-1] == 4:  # 如果通道数为4，通常是带有alpha通道的图像
        image = image[:, :, :3]  # 去除alpha通道

    image_resized = resize(image, (150, 150))  # 调整为150x150的大小

    # 预处理图像并进行预测
    input_image = image_resized[np.newaxis, ...]  # 添加批次维度
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])
    print("模型已加载")
    print("模型预测结果：", predictions)
    # 根据模型预测的类别显示结果
    if predictions <= 0.5:
        st.header("這是一隻猫！",divider='rainbow')
    else:
        st.header("這是一隻狗！",divider='rainbow')
    
    # 显示原始上传图像
    st.image(image, caption="上傳的圖像", use_column_width=True)
