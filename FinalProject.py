###Final Project

####packages

import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly
import plotly.graph_objects as go


####creating model 
s= pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x= np.where(x==1, 1, 0)
    return(x)
    
    
ss=pd.DataFrame ({
    "income": np.where (s["income"] > 9, np.nan, s["income"]),
    "education": np.where (s["educ2"] >8, np.nan, s["educ2"]),
    "parent": np.where (s["par"] ==1, 1, 0),
    "married": np.where (s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age" : np.where (s["age"] > 98, np.nan, s["age"]),
    "sm_li": clean_sm(s["web1h"])})

ss= ss.dropna()

y= ss["sm_li"]
x= ss[["income", "education", "parent", "married", "female","age"]]


x_train, x_test, y_train, y_test = train_test_split (x, 
                                                    y,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    random_state = 1498)

lr=LogisticRegression(class_weight= "balanced")

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

###App view 


st.image(image= "data:image/png;base64,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")
st.title("or")
st.image(image= "data:image/jpeg;base64,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")
st.subheader("Using Maching Learning to Determine the Probability of being a Member of LinkedIn")
st.caption("Created by: Stacie Horn")

st.write("Please Choose from the Following Options:")
st.caption("*All responses are annoymous and no information provided is saved")

income= st.selectbox ("Income Level", 
                options = ["Less than $10,000",
                "$10,000 to under $20,000",
                "$20,000 to under $30,000",
                "$30,000 to under $40,000",
                "$40,000 to under $50,000",
                "$50,000 to under $75,000", 
                "$75,000 to under $100,000",
                "$100,000 to under $150,000",
                "More than $150,000"])

if income =="Less than $10,000":
    inc= 1
elif income == "$10,000 to under $20,000":
    inc = 2
elif income == "$20,000 to under $30,000":
    inc = 3
elif income == "$30,000 to under $40,000":
    inc = 4
elif income == "$40,000 to under $50,000":
    inc = 5
elif income == "$50,000 to under $75,000": 
    inc = 6
elif income == "$75,000 to under $100,000":
    inc = 7
elif income == "$100,000 to under $150,000":
    inc = 8
else: inc = 9

education= st.selectbox ("Highest Level of School Completed", 
                options = ["Less than High School",
                "Some High School",
                "High School Graduate",
                "Some College",
                "Two-year Associate Degree",
                "Four-year College Degree", 
                "Some Graduate School",
                "Post Graduate or Professional Degree"])

if education=="Less than High School":
    edc= 1
elif education == "Some High School":
    edc = 2
elif education == "High School Graduate":
    edc = 3
elif education == "Some College":
    edc = 4
elif education == "Two-year Associate Degree":
    edc = 5
elif education == "Four-year College Degree": 
    edc = 6
elif education == "Some Graduate School":
    edc = 7
else: edc = 8

parent= st.selectbox ("Do You Have Children? ", 
                options = ["Yes",
                "No"])
if parent =="Yes":
    par = 1
else: par = 0

married= st.selectbox ("Are You Married? ", 
                options = ["Yes",
                "No"])
if married =="Yes":
    mar = 1
else: mar = 0

female= st.selectbox ("What is Your Gender ", 
                options = ["Female",
                "Male"])
if female =="Female":
    gen = 1
else: gen = 0

age= st.number_input ("Please Select Your Age", min_value=1, max_value=97)

user =[inc, edc, par, mar, gen, age]
predicted_class = lr.predict([user])
prob = lr.predict_proba([user])

if st.button("Are You LinkedIn or LinkedOut?"):
    user =[inc, edc, par, mar, gen, age]
    predicted_class = lr.predict([user])
    prob = lr.predict_proba([user])
    if prob[0][1] >.50:
        print(st.subheader ("LinkedIn"))
    else: 
        print(st.subheader ("LinkedOut"))

chart = go.Figure(go.Indicator(
    mode = "gauge + number",
    value = prob[0][1],
    title = { 'text': f"Probability LinkedIn"},
    gauge = {"axis": {"range": [0, 1]},
    "steps": [
        {"range": [0, .50], "color": "lightpink"},
        {"range": [.50, 1], "color": "lightgreen"}],
        "bar": {"color": "lightblue"}}))

st.plotly_chart(chart)
