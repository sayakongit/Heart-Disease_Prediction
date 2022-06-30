import pickle
import streamlit as st
import numpy as np

# loading the saved model
heart_disease_model = pickle.load(open('Model/heart_disease_model.sav','rb'))

# page title
st.title('Heart Disease Prediction using ML')


# Getting the user inputs        
col1, col2, col3 = st.columns(3)
        
with col1:
    age = st.number_input('Age', step=1,format="%i")
    trestbps = st.number_input('Resting Blood Pressure', step=1,format="%i")
    restecg = st.number_input('Resting Electrocardiographic results', step=1,format="%i")
    oldpeak = st.number_input('ST depression induced by exercise', step=1.,format="%.2f")
    thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', step=1,format="%i")
            
with col2:
    sex = st.number_input('Sex', step=1,format="%i")
    chol = st.number_input('Serum Cholestoral in mg/dl', step=1,format="%i")
    thalach = st.number_input('Maximum Heart Rate achieved', step=1,format="%i")
    slope = st.number_input('Slope of the peak exercise ST segment', step=1,format="%i")
            
with col3:
    cp = st.number_input('Chest Pain types', step=1,format="%i")
    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl', step=1,format="%i")
    exang = st.number_input('Exercise Induced Angina', step=1,format="%i")
    ca = st.number_input('Major vessels colored by flourosopy', step=1,format="%i")
    

# reshape the numpy array
  
array = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
input_data_as_numpy_array= np.asarray(array, dtype=float)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Model Prediction        
heart_prediction = heart_disease_model.predict(input_data_reshaped)
print(heart_prediction)       
                
# creating a button for Prediction

heart_diagnosis = ''   
if st.button('Heart Disease Test Result'):                      
            
    if (heart_prediction[0] == 1):
        heart_diagnosis = 'The person is having heart disease'
    else:
        heart_diagnosis = 'The person does not have any heart disease'
            
st.success(heart_diagnosis)


        
