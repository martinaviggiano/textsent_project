import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image
import re
import preprocessor as pproc
from cleantext import clean
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import os
from pathlib import Path

os.chdir(os.path.dirname(os.path.realpath(__file__)))


DATA_PATH = 'serialized'


st.set_page_config(
    page_title="Hate speech detection", page_icon="🔤", layout="centered"
)


# My custom functions
@st.cache()
def get_data(path):
    """
    Loads the serialized objects
    
    Parameters
    ----------
    path : str
        The folder where the serialized objects are stored
    
    Returns
    -------
    A tuple with the data pd.DataFrame, the TfidfVectorizer
    and the SVC fitted model.
    """
    
    data = pickle.load(open(Path(path, "data.pkl"), "rb"))
    vect = pickle.load(open(Path(path, "vect.pkl"), "rb"))
    svc_i = pickle.load(open(Path(path, "svc.pkl"), "rb"))
    vect_pos = pickle.load(open(Path(path, "vect_pos.pkl"), "rb"))
    svc_pos = pickle.load(open(Path(path, "svc_pos.pkl"), "rb"))

    return data, vect, svc_i, vect_pos, svc_pos
    

# Classification functions
def expand_contractions(text):
    cList = {
        "n't": " not",
        "/TD": " ",
        " PM ": " personal message ",
        " pm ": " personal message ",
        "PM ": "personal message ",
        " Donot ": " do not ",
        " MB ": " megabytes ",
    }
    
    c_re = re.compile("(%s)" % "|".join(cList.keys()))

    return c_re.sub(lambda match: cList[match.group(0)], text)


def full_text_clean(text):
    aa = expand_contractions(text)
    
    bb = pproc.clean(
        clean(pproc.clean(aa),
              fix_unicode=True,               # fix various unicode errors
              to_ascii=True,                  # transliterate to closest ASCII representation
              lower=True,                     # lowercase text
              no_line_breaks=True,            # fully strip line breaks as opposed to only normalizing them
              no_urls=True,                   # replace all URLs with a special token
              no_emails=True,                 # replace all email addresses with a special token
              no_phone_numbers=False,         # replace all phone numbers with a special token
              no_numbers=False,               # replace all numbers with a special token
              no_digits=False,                # replace all digits with a special token
              no_currency_symbols=False,      # replace all currency symbols with a special token
              no_punct=True,                  # remove punctuations
              replace_with_url=" ",
              replace_with_email=" ",
        )
    )
    
    swords = stopwords.words("english")
    swords.extend(string.punctuation)

    cc = (
        bb.lower()
        .replace(r"(@[a-z0-9]+)\w+", " ")
        .replace(r"http\S+", " ")
        .replace(r"www\S+", " ")
        .replace(r"com/watch", " ")
        .replace(r"\S*[.,:;!?-]\S*[^\s\.,:;!?-]", " ")
        .replace(r" th ", " ")
        .replace(r"\w*\d\w*", " ")
        .replace(r"rlm", " ")
        .replace(r"pttm", " ")
        .replace(r"ghlight", " ")
        .replace(r"[0-9]+(?:st| st|nd| nd|rd| rd|th| th)", " ")
        .replace(r"([^a-z \t])", " ")
        .replace(r" +", " "))
    
    cc = " ".join([i for i in cc.split() if not i in swords])
    
    return cc


def hate_predict(X, vect, clf):
    lista_pulita = [full_text_clean(text) for text in X]
    X_new = vect.transform(lista_pulita)
    classification = clf.predict(X_new)
    
    return classification
       
 
# Pages
def load_homepage(data):
    or_data = data[["label" , "text"]]
    with st.beta_expander("Show original data"):
        st.write(or_data)
    with st.beta_expander("Select columns to display"):
        selected_col = st.multiselect("Select Columns", data.columns)
        st.write(data.loc[:,selected_col])


def load_eda(data):
    # IMAGES Data Analysis
    freq_label = Image.open('images_d/Freq_labels.png')
    words_b_a = Image.open('images_d/Words_before_after.png')

    top20_words = Image.open('images_d/Top20_words.png')
    top20_adjs = Image.open('images_d/Adj_tot_hate.png')
    top20_nouns = Image.open('images_d/ Noun_tot_hate.png')
    top20_propns = Image.open('images_d/Propn_tot_hate.png')


    adj = Image.open('images_d/wc_a_hate.png')
    noun = Image.open('images_d/wc_n_hate.png')
    propn = Image.open('images_d/wc_p_hate.png')
    total_cloud = Image.open('images_d/wc_tot_hate.png')
    
    st.header("Original data")
    st.write("""
    You can choose which columns and label to display. Remember:

    - 0 = No Hate Speech
    - 1 = Hate Speech
    """)
    selected_col = st.multiselect("Select Columns", list(data.columns.values), list(data.columns.values))
    selected_lab = st.selectbox("Select Label", ["Hate Speech", "Non Hate Speech"])
    selected_lab_val = 1 if selected_lab == "Hate Speech" else 0

    st.write(data.loc[data['label'] == selected_lab_val, selected_col])

    st.header("Plot Section")

    st.write("In this section you can use the interactive tools to display plots of explorative analysis of data")

    st.subheader("Plot Section")
    st.write("Select the plot to display, by clicking on one of the following buttons:")


    if st.button('Frequency per Label'):
        st.image(freq_label, caption='Frequency per Label')
        
    if st.button('Word Counts'):
        st.image(words_b_a, caption='Count of Words before and after cleanining')
        
    if st.button('Top 20 Words'):
        st.image(top20_words, caption='Top 20 most common Words in the entire dataset')

    if st.button('Top 20 Adjectives'):
        st.image(top20_adjs, caption='Differences in frequency betweeen common Adjectives in Hate Speech and neutral sentences')

    if st.button('Top 20 Nouns'):
        st.image(top20_nouns, caption='Differences in frequency betweeen common Nouns in Hate Speech and neutral sentences')

    if st.button('Top 20 Proper Nouns'):
        st.image(top20_propns, caption='Differences in frequency betweeen common Proper Nouns in Hate Speech and neutral sentences')

    st.write("You can choose which label to display")
    selected_lab = st.multiselect("Select Label", data['label'].unique())
    st.write("Remember:")
    st.write(" - 0 = No Hate Speech")
    st.write(" - 1 = Hate Speech")
    st.write(f"Selected Option:  {selected_lab!r}")
    if 0 in selected_lab:
        st.write(data.loc[data['label'] == 0])
    if 1 in selected_lab:
        st.write(data.loc[data['label'] == 1])
        


    st.subheader("Top words in WordCloud")
    st.write("You can choose which Part of Speech to display")
    selected_pos = st.multiselect("Select the Part of Speech", ["Adjectives", "Nouns" , "Proper Nouns", "All of them"])
    st.write(f"Selected Option:  {selected_pos!r}")
    if "Adjectives" in selected_pos:
        st.image(adj, caption='Top 20 Adjectives in Hate Speech sentences')
    if "Nouns" in selected_pos:
        st.image(noun, caption='Top 20 Nouns in Hate Speech sentences')
    if "Proper Nouns" in selected_pos:
        st.image(propn, caption='Top 20 Proper Nouns in Hate Speech sentences')
    if "All of them" in selected_pos:
        st.image(total_cloud, caption='Top 20 of all the parts of speech in Hate Speech sentences')

####
def load_classif_under(data, vect, svc_i):
    written_under = st.text_input('Write your sentence here', 'I hate nobody' , key = '1')
        
    if not written_under:
        warn_lbl = st.warning('Please write the sentence you want to test')

    if written_under:
        warn_lbl = st.empty()
        pred = hate_predict([written_under], vect, svc_i)

        if pred == 0:
            prediction = 'NOT HATE SPEECH'
            color = 'green'
        else:
            prediction = 'HATE SPEECH'
            color = 'red'
     
        st.markdown(f'The sentence has been classified as: <span style="color:{color}">**{prediction}**</span>', unsafe_allow_html=True)


def load_classif_pos(data, vect_pos, svc_pos):
    written_pos = st.text_input('Write your sentence here', 'I hate nobody' , key = '2')
        
    if not written_pos:
        warn_lbl = st.warning('Please write the sentence you want to test')

    if written_pos:
        warn_lbl = st.empty()
        pred = hate_predict([written_pos], vect_pos, svc_pos)

        if pred == 0:
            prediction = 'NOT HATE SPEECH'
            color = 'green'
        else:
            prediction = 'HATE SPEECH'
            color = 'red'
     
        st.markdown(f'The sentence has been classified as: <span style="color:{color}">**{prediction}**</span>', unsafe_allow_html=True)



def main():
    """Main routine of the app"""
    
    st.header("HATE SPEECH DETECTION")
    st.subheader("Text mining and sentiment analysis project")
    st.write("Martina Viggiano (954603)")
    
    app_mode = st.sidebar.radio(
        "Go to:", ["Homepage", "Data Exploration", "Classification"]
    )
    
    data, vect, svc_i, vect_pos, svc_pos = get_data(DATA_PATH)
    
    if app_mode == "Homepage":
        load_homepage(data)
    elif app_mode == "Data Exploration":
        load_eda(data)
    elif app_mode == "Classification":
        st.subheader("Model with Undersampled data")
        load_classif_under(data, vect, svc_i)
        st.subheader("Model containing only some parts of speech")
        load_classif_pos(data, vect_pos, svc_pos)


if __name__ == "__main__":
    main()
