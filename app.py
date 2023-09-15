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
data_url_4 = get_image_html("CNN", "./image/ABC.PNG")
data_url_5 = get_image_html("分類", "./image/spaceship-titanic.PNG")


st.set_page_config(
    page_title="我的學習歷程",
    page_icon="👋",
)

st.title('Crab_Machine Learning 學習歷程')   

col1, col2 = st.columns(2)
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
col3, col4 = st.columns(2)
with col3:
    st.markdown('### [(分類)乳癌腫瘤預測](分類)')
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
    st.markdown('### [(CNN)神經網路_辨識英文字母)](CNN)')
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

col5, col6 = st.columns(2)
with col5:
    st.markdown('### [(分類)kaggle競賽(太空船難預測)](分類)')
    st.markdown('''
    ##### 數據說明:
歡迎來到 2912 年，您需要數據科學技能來解決宇宙之謎。我們收到了四光年外的信號，情況看起來不太妙。
太空飛船鐵達尼號是一個月前發射的星際客輪。船上有近 13,000 名乘客，這艘船開始了處女航，
將太陽系的移民運送到圍繞附近恆星運行的三顆新可居住的系外行星。
在繞過半人馬座阿爾法星前往它的第一個目的地——炎熱的巨蟹座 55 E 時，粗心的太空飛船鐵達尼號與隱藏在塵埃雲中的時空異常相撞
。可悲的是，它遭遇了與1000 年前同名的命運相似的命運。雖然船完好無損，但幾乎有一半的乘客被運送到了異次元！
您的任務是預測在太空飛船泰坦尼克號與時空異常相撞期間是否有乘客被運送到另一個維度。
為了幫助你做出這些預測，你會得到一組從船上受損的系統中恢復的個人記錄。        
   #####特徵說明: 
可以發現train data資料欄位為乘客ID、乘客離開的星球、乘客是否選擇在航行期間進入假死狀態、客艙編號、乘客將要去的星球、
年齡、VIP、豪華設施中所支付的金額、姓名與乘客是否被運送到另一個維度。其中是否傳送至異次元為所預測資料(label)，
乘客ID為影響是否傳送，其他項目為預測是否傳送的特徵資料     
Cabin
特徵說明:
假死狀態: 有假死存活率較高
Age:在0-12歲間 存活比例較高(禮讓幼童?)
solo:沒獨行 存活率高一點點
Cabin deck: 在B，C的比例存活率高一點
Cabin Side: 在S(右旋)存活率高一點
Expenditure : 建立一個新指標來區隔奢侈消費總金額，因窮人是完全沒花費
No_spending : 隨行人員幾乎沒消費
Family size : 5~6人間存活率較高       
        ''')
    # st.image('taxi.png')
    st.markdown(data_url_5, unsafe_allow_html=True)
