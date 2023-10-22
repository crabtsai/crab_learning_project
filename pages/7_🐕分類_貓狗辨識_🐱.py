import streamlit as st
from skimage import io, transform
import numpy as np
from tensorflow.keras.models import load_model
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

opt = RMSprop()
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

lr_track = get_lr_metric(opt)

MIN_LR = 1e-7
MAX_LR = 1e-3
CLR_METHOD = "triangular"



# Load the model with the custom layer, custom function, and custom optimizer

Vgg16 = keras.models.load_model('./model/cats_and_dogs_new_2.h5', compile=False)
Vgg16.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy' 
   , lr_track, keras.metrics.Precision(), keras.metrics.Recall()])

st.title("上傳圖片(貓~狗)辨識")
st.info("因訓練模型(VGG-16)輸入圖片為150*150，輸入圖片狗跟貓比例占比需高")

uploaded_file = st.file_uploader("上傳圖片(.png)", type=['png','jpg'])
if uploaded_file is not None:
    # 讀取上傳的圖像並調整大小
    image = io.imread(uploaded_file)
    if image.shape[-1] == 4:  # 如果通道數為4，通常是帶有alpha通道的圖像
        image = image[:, :, :3]  # 去除alpha通道

    image_resized = transform.resize(image, (150, 150))  # 調整為150x150的大小

    # 預處理圖像並進行預測
    input_image = image_resized[np.newaxis, ...]  # 添加批次維度
    predictions = Vgg16.predict(input_image)
    predicted_class = np.argmax(predictions[0])
    print("模型已加載")
    print("模型預測結果：", predictions)
    # 根據模型預測的類別顯示結果
    if predictions <= 0.5:
        st.header("這是一隻貓！", divider='rainbow')
    else:
        st.header("這是一隻狗！", divider='rainbow')
    
    # 顯示原始上傳圖像
    st.image(image, caption="上傳的圖像", use_column_width=True)
