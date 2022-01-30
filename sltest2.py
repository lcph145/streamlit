import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression


introduction= st.container()
data= st.container()
model= st.container()

@st.cache
def opendf(filename):
    df= pd.read_csv(filename)
    st.write(df)


with introduction:
    st.title("Linear Regression Model")
    st.text("Hello this is my first streamlit project. I will be using the linear regression model from scilearnkit for this project")
    image = Image.open('demo.jpg')
    st.image(image, caption='measuring heights')

with data:
    st.header("Data")
    st.text("measurements of height, age and gender")
    df=pd.DataFrame({ 'height': [163,170,185,190,163],
                        'gender':[0,1,1,1,0],
                            'age':[10,18,20,24,22]
   
    })
    
    #opendf(xxx)
    st.write(df)
    #df1= st.dataframe(df,) 

   

    df1= pd.DataFrame(df['height'].value_counts())
    st.bar_chart(df1)
    
    


with model:
    st.header("Running the model")
    input_col,results_col= st.columns(2)

with input_col:
    st.subheader("Enter age to predict height")
    #user_input=input_col.number_input('Insert age')
    user_input = st.slider('Please pick an age', 10, 100, 10)
    st.write('The current age inserted is',user_input)
  
    gender= st.selectbox('Which gender?',('Male', 'Female'))

    if gender== 'Male':
        genderconv=1
    if gender == 'Female':
        genderconv=0 


    X= df[['age','gender']]
    Y= df[['height']]
    reg = LinearRegression().fit(X, Y)
    regscore= reg.score(X, Y)
    regcoef= reg.coef_
    predict_score=reg.predict(np.array([[user_input,genderconv]]))


    
with results_col:
    st.subheader("Results")
    st.subheader('The R^2 value is ')
    st.write(regscore)
 
    st.subheader('The predicted height is')
    st.write(predict_score)
