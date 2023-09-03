import streamlit as st
import joblib

# 載入模型與標準化轉換模型
clf = joblib.load('./model/model.breast')
scaler = joblib.load('./model/scaler.breast')
st.title('乳癌預測')
#癌症檢測參數
radius = st.slider('radius:', min_value=6.981, max_value=28.11, value=8.0)
texture = st.slider('texture:', min_value=9.710, max_value=39.28, value=10.0)
perimeter = st.slider('perimeter:', min_value=43.79, max_value=118.5, value=50.0)
area = st.slider('area:', min_value=143.5, max_value=2501.0, value=200.0)
smoothness = st.slider('smoothness:', min_value=0.053, max_value=0.163, value=0.08)
compactness = st.slider('compactness:', min_value=0.019, max_value=0.345, value=0.02)
concavity = st.slider('concavity:', min_value=0.0, max_value=0.427, value=0.02)
concave_points = st.slider('concave_points:', min_value=0.0, max_value=0.201, value=0.1)
symmetry = st.slider('symmetry:', min_value=0.106, max_value=0.304, value=0.1)
fractal_dimension = st.slider('fractal_dimension:', min_value=0.05, max_value=0.097, value=0.08)
radius_s = st.slider('radius_s:', min_value=0.112, max_value=2.873, value=0.18)
texture_s = st.slider('texture_s:', min_value=0.36, max_value=4.885, value=0.58)
perimeter_s = st.slider('perimeter_s:', min_value=0.757, max_value=21.98, value=0.88)
area_s = st.slider('area_s:', min_value=6.802, max_value=542.2, value=8.88)
smoothness_s = st.slider('smoothness_s:', min_value=0.002, max_value=0.031, value=0.18)
compactness_s = st.slider('compactness_s:', min_value=0.002, max_value=0.135, value=0.18)
concavity_s = st.slider('concavity_s:', min_value=0.0, max_value=0.396, value=0.18)
concave_points_s = st.slider('concave_points_s:', min_value=0.0, max_value=0.053, value=0.08)
symmetry_s = st.slider('symmetry_s:', min_value=0.008, max_value=0.079, value=0.008)
fractal_dimension_s = st.slider('fractal_dimension_s:', min_value=0.001, max_value=0.03, value=0.008)
radius_w = st.slider('radius_w:', min_value=0.001, max_value=0.03, value=0.001)
texture_w = st.slider('texture_w:', min_value=12.02, max_value=49.54, value=12.108)
perimeter_w = st.slider('perimeter_w:', min_value=50.41, max_value=251.2, value=80.008)
area_w = st.slider('area_w:', min_value=185.2, max_value=4254.0, value=200.008)
smoothness_w = st.slider('smoothness_w:', min_value=0.071, max_value=0.223, value=0.001)
compactness_w = st.slider('compactness_w:', min_value=0.027, max_value=1.058, value=0.101)
concavity_w = st.slider('concavity_w:', min_value=0.0, max_value=1.252, value=0.101)
concave_points_w = st.slider('concave_points_w:', min_value=0.0, max_value=0.291, value=0.101)
symmetry_w = st.slider('symmetry_w:', min_value=0.156, max_value=0.664, value=0.301)
fractal_dimension_w = st.slider('fractal_dimension_w:', min_value=0.055, max_value=0.208, value=0.301)



labels = ['WDBC-惡性', 'WDBC-良性',]
if st.button('預測'):
    X_new = [[radius,texture,perimeter,area,smoothness,compactness,concavity,concave_points, 
    symmetry,fractal_dimension,radius_s,texture_s,perimeter_s,area_s,smoothness_s,compactness_s, 
    concavity_s,concave_points_s,symmetry_s,fractal_dimension_s,radius_w,texture_w,perimeter_w,area_w
    ,smoothness_w,compactness_w,concavity_w,concave_points_w,symmetry_w,fractal_dimension_w]]

    X_new = scaler.transform(X_new)
    st.write('### 預測結果是：', labels[clf.predict(X_new)[0]])
