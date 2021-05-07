from plots import (
    plot_freq_labels,
    plot_most_common_words,
    plot_top_20_pos,
    plot_word_hist,
)
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


st.set_page_config( page_title="Hate speech detection", page_icon="🔤", layout="centered")


# My custom functions
@st.cache(allow_output_mutation=True)
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
    and the SVC fitted model.log_fit_n 
    """
    
    data = pickle.load(open(Path(path, "data.pkl"), "rb"))
    vect = pickle.load(open(Path(path, "vect.pkl"), "rb"))
    svc_i = pickle.load(open(Path(path, "svc.pkl"), "rb"))
    vect_pos = pickle.load(open(Path(path, "vect_pos.pkl"), "rb"))
    log_pos = pickle.load(open(Path(path, "log_pos.pkl"), "rb"))
    
    mcw = pickle.load(open(Path(path, "mcw.pkl"), "rb"))
    top20adj = pickle.load(open(Path(path, "top20adj.pkl"), "rb"))
    top20noun = pickle.load(open(Path(path, "top20noun.pkl"), "rb"))
    top20propn = pickle.load(open(Path(path, "top20propn.pkl"), "rb"))
    top20verb = pickle.load(open(Path(path, "top20verb.pkl"), "rb"))
    top_pos = pickle.load(open(Path(path, "top_pos.pkl"), "rb"))

    return data, vect, svc_i, vect_pos, log_pos, mcw, top20adj, top20noun, top20propn, top20verb, top_pos
    

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
        "I'm" : "I am",
        " 've " : " have ",
        " 're " : " are ",
        " 'll " : " will ",
    }
    
    c_re = re.compile("(%s)" % "|".join(cList.keys()))

    return c_re.sub(lambda match: cList[match.group(0)], text)


def full_text_clean(text):
    aa = expand_contractions(text)
    
    bb = pproc.clean(
        clean(
            pproc.clean(aa),
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
            replace_with_email=" ")
    )
    
    swords = string.punctuation

    cc = (
        bb.lower()
        .replace(r"(@[a-z0-9]+)\w+", " ")
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
        .replace(r"http", " "))
    
    cc = " ".join([i for i in cc.split() if i not in swords]) ###############
    
    return cc


def hate_predict(X, vect, clf):
    lista_pulita = [full_text_clean(text) for text in X]
    X_new = vect.transform(lista_pulita)
    classification = clf.predict(X_new)
    
    return classification


# Pages
def load_homepage(data):
    st.markdown('''
    This dashboard is used to display the preliminary analysis and the results of Sentiment Analysis module project: the goal of the study is to <strong>classify text contents as hate speech</strong> or not.
    
    The dataset considered in the study contains textual data extracted from <em>Stormfront</em>, a white supremacist forum.
    
    In particular, data were taken from the Github repository <em>hate-speech-dataset</em> by Vicomtech, in which a list of sentences from a random set of forums posts is provided.
    
    A few number of variables are assigned to each of the sentences, including the label of interest: for every sentence, the dataset provides the column <em>label</em> which shows whether it has been classified as hate speech (1) or not hate speech (0).
    
    The first step was to clean deeply textual data; we faced some difficulties working with this dataset, since it turned out to be full of contractions and _ that created some bias, so had to be removed.
    
    After that, we perfomed several types of analysis on the data: we derived the most common words and also with respect to some characteristics of the words themselves, such as part of speech or the fact of being used in a sentence calssified as hate speech.
    
    Last thing was to employ various supervised statistical methods, training them of a set of data. Among them, we selected two models that seem to better perform classification in terms of precision and f1 score.
    In particular, they are a <strong>support vector machine</strong> model over a sample of the data balanced with respect to labels and the second one is a <strong>logistic regression</strong> over balanced data but considering only words beloning to specific parts of speech (nouns, proper nouns, verbs, adjectives, pronouns, subordinating conjunction, coordinating conjunction, "other" defined by spacy).
    
    ---
    
    In this dashboard you can find three pages: Homepage, Data Exploration, Classification.
    
    You are currently on the <strong>Homepage</strong>. :smile:
    
    On <strong>Data Exploration</strong> page you can use interactive tools to display plots and visualize data.
    
    On <strong>Classification</strong> page you can take advantage of the two trained model described above by writing a sentence on which you want to test the classification accuracy.
    ''', unsafe_allow_html = True)
    
   
def load_eda(data, mcw, top20adj, top20noun, top20propn, top20verb, top_pos):
    # IMAGES Data Analysis
    freq_label = Image.open('images_d/Freq_labels.png')
    words_b_a = Image.open('images_d/Words_before_after.png')

    top20_words = Image.open('images_d/Top20_words.png')
    top20_adjs = Image.open('images_d/Adj_tot_hate.png')
    top20_nouns = Image.open('images_d/ Noun_tot_hate.png')
    top20_propns = Image.open('images_d/Propn_tot_hate.png')
    top20_verbs = Image.open('images_d/Verb_tot_hate.png')


    adj = Image.open('images_d/wc_a_hate.png')
    noun = Image.open('images_d/wc_n_hate.png')
    propn = Image.open('images_d/wc_p_hate.png')
    verb = Image.open('images_d/wc_v_hate.png')
    total_cloud = Image.open('images_d/wc_tot_hate.png')
    
    st.write("On this page you can use the interactive tools to explore data and gather intormation about them.") 
    
    st.markdown("<h3><strong>Data Table Visualization</strong></h3>", unsafe_allow_html=True)

    st.write("Here you can build the data table by selecting columns and class of the label you want to display below.")
    st.write("At the bottom of this page you can find the legend explaining what each column contains.")
    
    selected_col = st.multiselect( "Select columns:", list(data.columns.values), list(data.columns.values))
    selected_lab = st.selectbox("Select the label:", ["Hate Speech", "Not Hate Speech"])
    selected_lab_val = 1 if selected_lab == "Hate Speech" else 0

    st.write(data.loc[data['label'] == selected_lab_val, selected_col])
    
    st.markdown("---", unsafe_allow_html=True)

    st.markdown("<h3><strong>Frequency Plots Section</h3></strong>", unsafe_allow_html=True)

    st.write("You can click on buttons below to display plots and explore data.")

    st.write("Select the plots to display by clicking on the corresponding buttons:")
    
    with st.beta_expander("Labels frequency"):
        # Choose a template among:
        #   "plotly", "plotly_white", "plotly_dark",
        #   "ggplot2", "seaborn", "simple_white", "none"
        st.plotly_chart(
            plot_freq_labels(data, template="plotly_white"), use_container_width=True
        )

    with st.beta_expander("Word Counts"):
        st.plotly_chart(
            plot_word_hist(data, template="plotly_white"), use_container_width=True
        )

    with st.beta_expander("Top 20 Words"):
        st.plotly_chart(
            plot_most_common_words(mcw, template="plotly_white"),
            use_container_width=True,
        )

    with st.beta_expander("Top 20 Adjectives"):
        st.plotly_chart(
            plot_top_20_pos(
                top20adj,
                x_col="Adj",
                title="Frequency of Top 20 Adjectives in Hate Speech and total sentences",
                template="plotly_white",
            ),
            use_container_width=True,
        )

    with st.beta_expander("Top 20 Nouns"):
        st.plotly_chart(
            plot_top_20_pos(
                top20noun,
                x_col="Nouns",
                title="Frequency of Top 20 Nouns in Hate Speech and total sentences",
                template="plotly_white",
            ),
            use_container_width=True,
        )

    with st.beta_expander("Top 20 Proper Nouns"):
        st.plotly_chart(
            plot_top_20_pos(
                top20propn,
                x_col="Proper Nouns",
                title="Frequency of Top 20 Proper Nouns in Hate Speech and total sentences",
                template="plotly_white",
            ),
            use_container_width=True,
        )
    
    with st.beta_expander("Top 20 Verbs"):
        st.plotly_chart(
            plot_top_20_pos(
                top20verb,
                x_col="Verb",
                title="Frequency of Top 20 Verbs in Hate Speech and total sentences",
                template="plotly_white",
            ),
            use_container_width=True,
        )
        
    with st.beta_expander("Top POS"):
        st.dataframe(top_pos)
        
        

    st.markdown("---", unsafe_allow_html=True)

    st.markdown("<h3><strong>Top words in WordCloud</h3></strong>", unsafe_allow_html=True)
    
    st.write("In this section you can choose to display one or more cloud of words plots.")
    st.write("You can choose between the cloud containing the 20 most common words for four parts of speech (Nouns, Proper Nouns, Adjectives and Verbs) or the plot that displays all of them together.")
    selected_pos = st.multiselect("Select the part of speech:", ["Adjectives", "Nouns" , "Proper Nouns", "Verbs", "All of them"])
    st.write(f"Selected Option:  {selected_pos!r}")
    if "Adjectives" in selected_pos:
        st.image(adj, caption='Top 20 Adjectives in Hate Speech sentences')
    if "Nouns" in selected_pos:
        st.image(noun, caption='Top 20 Nouns in Hate Speech sentences')
    if "Proper Nouns" in selected_pos:
        st.image(propn, caption='Top 20 Proper Nouns in Hate Speech sentences')
    if "Verbs" in selected_pos:
        st.image(verb, caption='Top 20 Verbs in Hate Speech sentences')
    if "All of them" in selected_pos:
        st.image(total_cloud, caption='Top 20 of all the parts of speech in Hate Speech sentences')
        
    st.markdown("---")
    st.markdown("<h3><strong>Legend</h3></strong>", unsafe_allow_html=True)
    st.markdown('''
    - <em>label</em> = Categorical variable for which 0 is Not Hate Speech and 1 is Hate Speech;
    - <em>text</em> = String containing the original sentence;
    - <em>text_clean</em> = String cleaned;
    - <em>POS_spacy</em> = List of tokens associated to the part of speech;
    - <em>lemmatized</em> = String of tokens lemmatized by spacy;
    - <em>tokens</em> = List of tokens retrieved by spacy;
    - <em>language</em> = Language of the sentence defined by spacy;
    - <em>word_count_before</em> = Number of tokens in original sentence, before cleaning;
    - <em>word_count</em> = Number of words after cleaning;
    - <em>word_cleaning</em> = Difference in number of words before and after cleaning;
    - <em>NOUN</em> = List of nouns in the sentence;
    - <em>NOUN_count</em> = Number of nouns in the sentence;
    - <em>PROPN</em> = List of proper nouns in the sentence;
    - <em>PROPN_count</em> = Number of proper nouns in the sentence;    
    - <em>VERB</em> = List of verbs in the sentence;
    - <em>VERB_count</em> = Number of verbs in the sentence;
    - <em>ADJ</em> = List of adjectives in the sentence;
    - <em>ADJ_count</em> = Number of advectives in the sentence;
    - <em>ADV</em> = List of adverbs in the sentence;
    - <em>ADV_count</em> = Number of adverbs in the sentence;
    - <em>PRON</em> = List of pronouns in the sentence;
    - <em>PRON_count</em> = Number of pronouns in the sentence;
    - <em>SCONJ</em> = List of subordinating conjunction in the sentence;
    - <em>SCONJ_count</em> = Number of subordinating conjunction in the sentence;
    - <em>INTJ</em> = List of interjection in the sentence;
    - <em>INTJ_count</em> = Number of interjection in the sentence;    
    
    ''', unsafe_allow_html=True)
        

def load_classif(data, vect, svc_i, vect_pos, log_pos):
    st.write('''
    On this page you can test two of the models that have been trained for the project:
    
    1. Support vector machine modelled on data balanced with RandomUnderSampler;
    2. Logistic regression over data including only some parts of speech.
    ''')
    
    st.write("To do so, you need to write down the sentence you want to test in the board below.")
    
    st.write("As you cas see, there is a sentence displayed by default. It was choosen since it clearly shows that the two models work differently: in this case, the second method performs better the classification between Hate Speech and Not Hate Speech.")
    
    
    written_sent = st.text_input('Write your sentence here:', "I don't think I am racist, but I hate blacks!")

    warn_lbl = st.empty()
    if not written_sent:
        warn_lbl = st.warning('Please write the sentence you want to test')

    if written_sent:
        if warn_lbl is not None:
            warn_lbl = st.empty()

        pred_under = hate_predict([written_sent], vect, svc_i)
        pred_pos = hate_predict([written_sent], vect_pos, log_pos)

        prediction_under, color_under = get_text_color(pred_under)
        prediction_pos, color_pos = get_text_color(pred_pos)

        st.markdown("<h3><strong>Model with Undersampled data</strong></h3>", unsafe_allow_html = True)
        st.markdown(f'The sentence has been classified as: <span style="color:{color_under}">**{prediction_under}**</span>', unsafe_allow_html=True)

        st.markdown("<h3><strong>Model containing only some Parts of speech</h3></strong>", unsafe_allow_html= True)
        st.markdown(f'The sentence has been classified as: <span style="color:{color_pos}">**{prediction_pos}**</span>', unsafe_allow_html=True)


def get_text_color(pred):
    if pred == 0:
        prediction = 'NOT HATE SPEECH'
        color = 'green'
    else:
        prediction = 'HATE SPEECH'
        color = 'red'
    
    return prediction, color


def main():
    """Main routine of the app"""
    
    st.markdown("<h1>HATE SPEECH DETECTION</h1>", unsafe_allow_html=True)
    st.markdown("<h4><strong>Text mining and sentiment analysis project</strong></h4>", unsafe_allow_html=True)
    st.write("Martina Viggiano (954603)")
    st.markdown("---", unsafe_allow_html=True)
    
    app_mode = st.sidebar.radio(
        "Go to:", ["Homepage", "Data Exploration", "Classification"]
    )
    
    data, vect, svc_i, vect_pos, log_pos, mcw, top20adj, top20noun, top20propn, top20verb, top_pos = get_data(DATA_PATH)
    
    if app_mode == "Homepage":
        load_homepage(data)
    elif app_mode == "Data Exploration":
        load_eda(data, mcw, top20adj, top20noun, top20propn, top20verb, top_pos)
    elif app_mode == "Classification":
        load_classif(data, vect, svc_i, vect_pos, log_pos)

if __name__ == "__main__":
    main()
