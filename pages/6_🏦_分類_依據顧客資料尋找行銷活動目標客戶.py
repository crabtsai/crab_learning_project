import streamlit as st
import joblib
import numpy as np
# 載入模型與標準化轉換模型
clf = joblib.load('./model/model.custom')

st.title('銀行預測顧客是否購買定存')
#銀行檢測參數
euribor3m = st.slider('euribor 3 個月利率:', min_value=0.634, max_value=5.045, value=1.88)
job = st.selectbox('job工作:',['blue-collar','job_housemaid'])
if job == 'blue-collar':
    job_blue_collar = 1
    job_housemaid = 0
else:
    job_blue_collar = 1
    job_housemaid = 0
marital_unknown = st.checkbox('是否結婚了')
if marital_unknown:
    marital_unknown = 1
else:
    marital_unknown = 0

education_illiterate = st.checkbox('識字')
if education_illiterate:
    education_illiterate = 1
else:
    education_illiterate = 0

# default_no = st.selectbox("信用不良", ['False', 'True','unknown'])
# if default_no == 'False':
#     default_no = 1
#     default_unknown =0
# elif default_no =='True':
#     default_no = 0
#     default_unknown = 0    
# else:
#     default_no = 0
#     default_unknown =1  

# contact_cellular = st.selectbox("是否有留家用電話號碼", ['False', 'True'])
# contact_cellular_list = {'False':0,'True':1}
# contact_cellular = contact_cellular_list[contact_cellular]

# contact_telephone = st.selectbox("是否有留手機號碼", ['False', 'True'])
# contact_telephone_list = {'False':0,'True':1}
# contact_telephone = contact_telephone_list[contact_telephone]

month = st.selectbox('最後一次聯繫月份',['3','4','5','6','7','8','10','11','12'])
month_apr = 0
month_aug = 0
month_dec = 0
month_jul = 0 
month_jun = 0
month_mar = 0
month_may = 0
month_nov = 0
month_oct = 0
# 根據選擇的月份設置相應的變數為1
if month == '3':
    month_mar = 1
elif month == '4':
    month_apr = 1
elif month == '5':
    month_may = 1
elif month == '6':
    month_jun = 1
elif month == '7':
    month_jul = 1
elif month == '8':
    month_aug = 1
elif month == '10':
    month_oct = 1
elif month == '11':
    month_nov = 1
elif month == '12':
    month_dec = 1

poutcome = st.selectbox("先前行銷結果", ['failure', 'success'])
if poutcome == 'failure':
    poutcome_failure = 1
    poutcome_success = 0
else:
    poutcome_failure = 0
    poutcome_success = 1    




labels = ['不買', '買']
if st.button('預測'):
    X_new = [[float(euribor3m), float(job_blue_collar), float(job_housemaid), float(marital_unknown),
       float(education_illiterate), float(month_apr), float(month_aug),
       float(month_dec), float(month_jul), float(month_jun), float(month_mar), float(month_may),
       float(month_nov), float(month_oct), float(poutcome_failure), float(poutcome_success)]]

    # 如果 X_new 是一個列表，轉換為二維數組
    X_new = np.array(X_new)
    X_new = np.array(X_new).reshape(1, -1)
    # print(f'X_new.shape: {X_new.shape}')
    # print(f'X_new: {X_new}')
    # print(f'X_new: {type(X_new)}')
    prediction_result = clf.predict(X_new).tolist()
    prediction_index = int(prediction_result[0])
    st.write('### 預測結果是：', labels[prediction_index])

