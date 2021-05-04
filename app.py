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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import statistics
from statistics import mode


st.set_page_config(
    page_title="Hate speech detection", page_icon="🔤", layout="centered"
)


progress_bar = st.progress(0)
progress_text = st.empty()
for i in range(101):
    time.sleep(0.1)
    progress_bar.progress(i)
    progress_text.text(f"Progress: {i}%")


st.header("HATE SPEECH DETECTION")
st.subheader("Text mining and sentiment analysis project")
st.write("Martina Viggiano (954603)")

#@st.cache
with open("C:/Users/marti/repos/textsent_project/data.pkl", "rb") as f:
    data = pickle.load(f)

#@st.cache
#def get_data(path):
#    data = pickle.load(open(path, "rb"))
#    return data(path)


   
st.write("Entire dataset")
#@st.cache
st.write(data)

#IMAGES Data Analysis
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
    

    
#if 0 in selected_sex and 1 in selected_sex:
#    st.write(data)


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


####################################################
## CLASSIFICATION


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
              no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
              no_urls=True,                  # replace all URLs with a special token
              no_emails=True,                # replace all email addresses with a special token
              no_phone_numbers=False,         # replace all phone numbers with a special token
              no_numbers=False,               # replace all numbers with a special token
              no_digits=False,                # replace all digits with a special token
              no_currency_symbols=False,      # replace all currency symbols with a special token
              no_punct=True,                 # remove punctuations
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
        .replace(r" +", " ")
        )
    
    bb = " ".join([i for i in bb.split() if not i in swords])
    
    return bb

def hate_predict(X, vect, clf):
    lista_pulita = [full_text_clean(X) for text in X]
    X_new = vect.transform(lista_pulita)
    classification = clf.predict(X_new)
    return classification


tfidf_vectorizer = TfidfVectorizer()
X_bal = tfidf_vectorizer.fit_transform(data['lemmatized'])
y_bal = data["label"]
undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_under, y_under = undersample.fit_resample(X_bal, y_bal)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_under, y_under, test_size=0.3, random_state=42, stratify=y_under)
svc_i = SVC(verbose=10, random_state=42)
svc_i.fit(X_train_i, y_train_i)


written = st.text_input('Write your sentence here')

pred = hate_predict(written, tfidf_vectorizer, svc_i)
  
def most_common(List):
    return(mode(List))
    
prediction = most_common(pred)

if not written:
    st.warning('Please write the sentence you want to test')

#if written == []:
#    st.write( 'Please write the sentence you want to test' )
#if written is None:
#    st.write( 'Please write the sentence you want to test' )

if written:
    if prediction == 0:
        prediction = 'NOT HATE SPEECH'
    else:
        prediction = 'HATE SPEECH'
    st.write( 'The sentence has been classified as:', prediction )
    




progress_bar = st.progress(0)
progress_text = st.empty()
for i in range(101):
    time.sleep(0.1)
    progress_bar.progress(i)
    progress_text.text(f"Progress: {i}%")
    
with st.spinner('Wait for it...'):
    time.sleep(3)
st.success('Done!')