import streamlit as st
import joblib
import numpy as np

# 載入模型與標準化轉換模型
clf = joblib.load('./model/model_space.pkl')
scaler = joblib.load('./model/scaler_space.pkl')
st.title('太空船難')

#檢測參數


HomePlanet = st.selectbox("HomePlanet(家鄉星球)", ['Europa', 'Earth', 'Mars'])
CryoSleep = st.selectbox("CryoSleep(冷凍睡眠)", ['False', 'True'])
Destination = st.selectbox("Destination(目的地)", ['TRAPPIST-1e', 'PSO J318.5-22','55 Cancri e'])
Age = st.slider('age:', min_value=0, max_value=79, value=18)
VIP = st.selectbox("VIP(是否有付費享受)", ['False', 'True'])
RoomService = st.slider('RoomService(客房支付費用):', min_value=0, max_value=14327, value=10000)
FoodCourt = st.slider('FoodCourt(美食廣場費用):', min_value=0, max_value=29813, value=10000)
ShoppingMall = st.slider('ShoppingMall(支付費用):', min_value=0, max_value=23492, value=10000)
Spa = st.slider('Spa(支付費用):', min_value=0, max_value=23492, value=10000)
VRDeck = st.slider('VRDeck(支付費用):', min_value=0, max_value=24133, value=10000)
Expenditure = st.slider('Expenditure(總花費):', min_value=0, max_value=35987, value=10000)
No_spending = st.slider('No_spending:', min_value=0, max_value=1, value=0)
Solo = st.slider('Solo(獨行):', min_value=0, max_value=1, value=1)
Cabin_deck = st.selectbox("Cabin_deck(A(居住人最少)~G(人最多))", ['B', 'F', 'A', 'G','E','D','C','T'])
Cabin_side = st.selectbox("Cabin_side(P左旋，S右璇，Z甲板)", ['P', 'S', 'Z'])
Cabin_region1 = st.slider('Cabin_region1(最多人):', min_value=0, max_value=1, value=0)
Cabin_region2 = st.slider('Cabin_region2:', min_value=0, max_value=1, value=0)
Cabin_region3 = st.slider('Cabin_region3:', min_value=0, max_value=1, value=0)
Cabin_region4 = st.slider('Cabin_region4:', min_value=0, max_value=1, value=0)
Cabin_region5 = st.slider('Cabin_region5:', min_value=0, max_value=1, value=0)
Cabin_region6 = st.slider('Cabin_region6:', min_value=0, max_value=1, value=0)
Cabin_region7 = st.slider('Cabin_region7(最少人):', min_value=0, max_value=1, value=0)
Family_size = st.slider('Family_size:', min_value=0, max_value=19, value=1)

labels = ['死亡', '倖存',]
# if st.button('預測'):
#     X_new = [[HomePlanet, CryoSleep, Destination, Age, VIP,
#     RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,
#     Expenditure, No_spending,
#     Solo, Cabin_deck, Cabin_side, Cabin_region1,
#     Cabin_region2, Cabin_region3, Cabin_region4, Cabin_region5,
#     Cabin_region6, Cabin_region7, Family_size]]
#     X_new = np.array(X_new)
#     X_new = scaler.transform(X_new)
#     st.write('### 預測結果是：', labels[clf.predict(X_new)[0]])


import pandas as pd

# ... 其他部分代码不变 ...

if st.button('預測'):
    data = [[HomePlanet, CryoSleep, Destination, Age, VIP,
            RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,
            Expenditure, No_spending,
            Solo, Cabin_deck, Cabin_side, Cabin_region1,
            Cabin_region2, Cabin_region3, Cabin_region4, Cabin_region5,
            Cabin_region6, Cabin_region7, Family_size]]
    
    # 将数据转换为 pandas DataFrame
    columns = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
               'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
               'Expenditure', 'No_spending', 'Solo', 'Cabin_deck',
               'Cabin_side', 'Cabin_region1', 'Cabin_region2', 'Cabin_region3',
               'Cabin_region4', 'Cabin_region5', 'Cabin_region6', 'Cabin_region7', 'Family_size']
    
    X_new_df = pd.DataFrame(data, columns=columns)
    
    # 使用 DataFrame 进行标准化
    X_new_scaled = scaler.transform(X_new_df)
    
    st.write('### 預測結果是：', labels[clf.predict(X_new_scaled)[0]])

