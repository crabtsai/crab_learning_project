# https://docs.streamlit.io/library/cheatsheet
# streamlit run app.py
import streamlit as st
import numpy as np 
import joblib
import base64

def get_image_html(page_name, file_name):
    with open(file_name, "rb") as f:
        contents = f.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    return f'<a href="{page_name}"><img src="data:image/png;base64,{data_url}" style="width:300px"></a>'

data_url = get_image_html("分類", "./image/iris.png")
data_url_2 = get_image_html("迴歸", "./image/taxi.png")
data_url_3 = get_image_html("分類", "./image/breast.jpg")
data_url_4 = get_image_html("CNN辨識英文字母", "./image/ABC.PNG")

st.set_page_config(
    page_title="我的學習歷程",
    page_icon="👋",
)

st.title('Machine Learning 學習歷程')   

col1, col2 , col3, col4 = st.columns(4,gap="small")
with col1:
    # url must be external url instead of local file
    # st.markdown(f"### [![分類]({url})](分類)")
    st.markdown('### [(分類)企鵝品種辨識](分類)')
    st.markdown('''
    ##### 特徵(X):
        - 島嶼
        - 嘴巴長度
        - 嘴巴寬度
        - 翅膀長度
        - 體重
        - 性別
    ##### 預測類別(Class):
        - Adelie
        - Chinstrap
        - Gentoo
        ''')
    # st.image('iris.png')
    st.markdown(data_url, unsafe_allow_html=True)
with col2:
    st.markdown('### [(迴歸)計程車小費預測](迴歸)')
    st.markdown('''
    ##### 特徵(X):
        - 車費
        - 性別
        - 吸菸
        - 星期
        - 時間
        - 同行人數
    ##### 目標：預測小費金額
        ''')
    # st.image('taxi.png')
    st.markdown(data_url_2, unsafe_allow_html=True)
with col3:
    st.markdown('### [(分類)乳房預測](分類)')
    st.markdown('''
    ##### 特徵(X):
                #Breast cancer wisconsin (diagnostic) dataset#
         - 半徑（從中心到周邊點的距離的平均值）
         - 紋理（灰度值的標準偏差）
         - 周長
         - 區域
         - 平滑度（半徑長度的局部變化）
         - 緊湊性（周長^2 /面積 - 1.0）
         - 凹度（輪廓凹入部分的嚴重程度）
         - 凹點（輪廓凹部的數量）
         - 對稱性
         - 分形維數
    ##### 目標：腫瘤 良性or惡性
        ''')
    # st.image('taxi.png')
    st.markdown(data_url_3, unsafe_allow_html=True)
with col4:
    st.markdown('### [(CNN神經網路_辨識英文字母)')
    st.markdown('''
    ##### 資料集:
         -EMNIST Letters
    ##### 神經網路:
         -  Layer (type)                Output Shape              Param #   
            =================================================================
            conv2d_21 (Conv2D)          (None, 26, 26, 32)        320       
                                                                            
            max_pooling2d_14 (MaxPooli  (None, 13, 13, 32)        0         
            ng2D)                                                           
                                                                            
            conv2d_22 (Conv2D)          (None, 11, 11, 64)        18496     
                                                                            
            max_pooling2d_15 (MaxPooli  (None, 5, 5, 64)          0         
            ng2D)                                                           
                                                                            
            flatten_14 (Flatten)        (None, 1600)              0         
                                                                            
            dense_26 (Dense)            (None, 64)                102464    
                                                                            
            dropout_11 (Dropout)        (None, 64)                0         
                                                                            
            dense_27 (Dense)            (None, 26)                1690  
            =================================================================           
    ##### 目標：辨識A~Z英文字母
        ''')
    # st.image('taxi.png')
    st.markdown(data_url_4, unsafe_allow_html=True)
