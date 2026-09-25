import numpy as np              
import joblib 
import streamlit as st         

#--------------------------------------
#load the model
obj=joblib.load('california.joblib')  #is dict
model=obj['model']
cols=obj['columns']

#---------------------------------

st.title('California app')
In=[]
for i in cols:
    v=st.number_input(f'Enter {i} Value:')
    In.append(v)
    
if st.button('click'):
    out=model.predict([In])
    st.success(f'The Median House value is: {out}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
