import streamlit as st
import joblib
import numpy as np
import pandas as pd
# 載入模型與標準化轉換模型
clf = joblib.load('./model/model_spaceship.joblib')
scaler = joblib.load('./model/scaler_spaceship.joblib')
st.title('太空船難預測')
#太空船難檢測參數
HomePlanet = st.slider('HomePlanet:', min_value=0, max_value=1, value=0)
CryoSleep = st.slider('CryoSleep:', min_value=0, max_value=1, value=0)
Destination = st.slider('Destination:', min_value=0, max_value=1, value=0)
Age = st.slider('age:', min_value=0, max_value=79, value=18)
VIP = st.slider('VIP:', min_value=0, max_value=1, value=0)
RoomService = st.slider('RoomService:', min_value=0, max_value=14327, value=10000)
FoodCourt = st.slider('FoodCourt:', min_value=0, max_value=29813, value=10000)
ShoppingMall = st.slider('ShoppingMall:', min_value=0, max_value=23492, value=10000)
Spa = st.slider('Spa:', min_value=0, max_value=23492, value=10000)
VRDeck = st.slider('VRDeck:', min_value=0, max_value=24133, value=10000)
Expenditure = st.slider('Expenditure:', min_value=0, max_value=35987, value=10000)
No_spending = st.slider('No_spending:', min_value=0, max_value=1, value=1)
# Group = st.slider('Group:', min_value=1, max_value=9280, value=100)
# Group_size = st.slider('Group_size:', min_value=1, max_value=8, value=1)
Solo = st.slider('Solo:', min_value=0, max_value=1, value=1)
Cabin_deck = st.slider('Cabin_deck:', min_value=0, max_value=1, value=0)
Cabin_side = st.slider('Cabin_side:', min_value=0, max_value=1, value=0)
# Cabin_number = st.slider('Cabin_number:', min_value=-2, max_value=1894, value=1)
Cabin_region1 = st.slider('Cabin_region1:', min_value=0, max_value=1, value=0)
Cabin_region2 = st.slider('Cabin_region2:', min_value=0, max_value=1, value=0)
Cabin_region3 = st.slider('Cabin_region3:', min_value=0, max_value=1, value=0)
Cabin_region4 = st.slider('Cabin_region4:', min_value=0, max_value=1, value=0)
Cabin_region5 = st.slider('Cabin_region5:', min_value=0, max_value=1, value=0)
Cabin_region6 = st.slider('Cabin_region6:', min_value=0, max_value=1, value=0)
Cabin_region7 = st.slider('Cabin_region7:', min_value=0, max_value=1, value=0)
Family_size = st.slider('Family_size:', min_value=0, max_value=19, value=1)




# PassengerId', 'Group', 'Group_size', 'Age_group', 'Cabin_number


labels = ['死亡', '倖存',]
if st.button('預測'):
    X_new = [[HomePlanet, CryoSleep, Destination, Age, VIP,
       RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,
        Expenditure, No_spending,
       Solo, Cabin_deck, Cabin_side, Cabin_region1,
       Cabin_region2, Cabin_region3, Cabin_region4, Cabin_region5,
       Cabin_region6, Cabin_region7, Family_size]]

    X_new = scaler.transform(X_new)
    st.write('### 預測結果是：', labels[clf.predict(X_new)[0]])
