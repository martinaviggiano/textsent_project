import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

st.set_page_config(
    page_title="Hate speech detection", page_icon="🔤", layout="centered"
)

st.header("HATE SPEECH DETECTION")
st.subheader("Text mining and sentiment analysis project")
st.write("Martina Viggiano (954603)")

#@st.cache
with open("C:/Users/marti/repos/textsent_project/data.pkl", "rb") as f:
    data = pickle.load(f)
    
st.write("This is the entire dataset")
#@st.cache
st.write(data)


st.write("You can choose which label to display")
selected_sex = st.multiselect("Select Label", data['label'].unique())
st.write("Remember: 0 = No Hate Speech and 1 = Hate Speech")
st.write(f"Selected Option:  {selected_sex!r}", selected_sex)
if 0 in selected_sex:
    st.write(data.loc[data['label'] == 0])
if 1 in selected_sex:
    st.write(data.loc[data['label'] == 1])
    
#if 0 in selected_sex and 1 in selected_sex:
#    st.write(data)




progress_bar = st.progress(0)
progress_text = st.empty()
for i in range(101):
    time.sleep(0.1)
    progress_bar.progress(i)
    progress_text.text(f"Progress: {i}%")