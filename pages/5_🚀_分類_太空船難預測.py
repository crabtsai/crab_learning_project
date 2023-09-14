import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

# 載入模型與標準化轉換模型
clf = joblib.load('./model/model_spaceship.joblib')
scaler = joblib.load('./model/scaler_spaceship.joblib')
st.title('太空船難預測')
#太空船難檢測參數
HomePlanet = float(st.slider('HomePlanet:', min_value=0.0, max_value=1.0, value=0.0))
CryoSleep = float(st.slider('CryoSleep:', min_value=0.0, max_value=1.0, value=0.0))
Destination = float(st.slider('Destination:', min_value=0.0, max_value=1.0, value=0.0))
Age = float(st.slider('age:', min_value=0.0, max_value=79.0, value=18.0))
VIP = float(st.slider('VIP:', min_value=0.0, max_value=1.0, value=0.0))
RoomService = float(st.slider('RoomService:', min_value=0.0, max_value=14327.0, value=10000.0))
FoodCourt = float(st.slider('FoodCourt:', min_value=0.0, max_value=29813.0, value=10000.0))
ShoppingMall = float(st.slider('ShoppingMall:', min_value=0.0, max_value=23492.0, value=10000.0))
Spa = float(st.slider('Spa:', min_value=0.0, max_value=23492.0, value=10000.0))
VRDeck = float(st.slider('VRDeck:', min_value=0.0, max_value=24133.0, value=10000.0))
Expenditure = float(st.slider('Expenditure:', min_value=0.0, max_value=35987.0, value=10000.0))
No_spending = float(st.slider('No_spending:', min_value=0.0, max_value=1.0, value=10000.0))
# Group = float(st.slider('Group:', min_value=1, max_value=9280, value=100.0))
# Group_size = float(st.slider('Group_size:', min_value=1, max_value=8, value=1.0))
Solo = float(st.slider('Solo:', min_value=0.0, max_value=1.0, value=1.0))
Cabin_deck = float(st.slider('Cabin_deck:', min_value=0.0, max_value=1.0, value=0.0))
Cabin_side = float(st.slider('Cabin_side:', min_value=0.0, max_value=1.0, value=0.0))
# Cabin_number = float(st.slider('Cabin_number:', min_value=-2, max_value=1894, value=1.0))
Cabin_region1 = float(st.slider('Cabin_region1:', min_value=0.0, max_value=1.0, value=0.0))
Cabin_region2 = float(st.slider('Cabin_region2:', min_value=0.0, max_value=1.0, value=0.0))
Cabin_region3 = float(st.slider('Cabin_region3:', min_value=0.0, max_value=1.0, value=0.0))
Cabin_region4 = float(st.slider('Cabin_region4:', min_value=0.0, max_value=1.0, value=0.0))
Cabin_region5 = float(st.slider('Cabin_region5:', min_value=0.0, max_value=1.0, value=0.0))
Cabin_region6 = float(st.slider('Cabin_region6:', min_value=0.0, max_value=1.0, value=0.0))
Cabin_region7 = float(st.slider('Cabin_region7:', min_value=0.0, max_value=1.0, value=0.0))
Family_size = float(st.slider('Family_size:', min_value=0.0, max_value=19.0, value=1.0))




# PassengerId', 'Group', 'Group_size', 'Age_group', 'Cabin_number


labels = ['死亡', '倖存',]
if st.button('預測'):
    X_new = [[HomePlanet, CryoSleep, Destination, Age, VIP,
       RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,Expenditure, No_spending,
       Solo, Cabin_deck, Cabin_side, Cabin_region1,
       Cabin_region2, Cabin_region3, Cabin_region4, Cabin_region5,
       Cabin_region6, Cabin_region7, Family_size]]

    print(X_new.type())
    print(X_new)
    
    X_new = scaler.fit_transform(X_new)

    st.write('### 預測結果是：', labels[clf.predict(X_new)[0]])

