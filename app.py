import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Hate speech detection", page_icon="🎓", layout="centered"
)

st.title("Hate speech detection")
st.write("Text mining and sentiment analysis project")
st.write("Martina Viggiano (954603)")

with open("data.pkl", "rb") as f:
    data = pickle.load(f)

st.write(data)
