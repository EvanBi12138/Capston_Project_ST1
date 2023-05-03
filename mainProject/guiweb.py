import streamlit as st
from joblib import dump, load
import pandas as pd


rf = load('rfcFinal.joblib')


st.title('Prediction of the popularity of business news articles')
st.subheader('Input of constant feature values')
num_hrefs = st.text_input("Number of links:")
kw_avg_min = st.text_input("Worst keyword (avg. shares)")
kw_min_avg = st.text_input("Avg. keyword (min. shares)")
kw_max_min = st.text_input("Worst keyword (max. shares)")
kw_max_avg = st.text_input("Avg. keyword (max. shares)")
n_tokens_content = st.text_input("Number of words in the content")
st.subheader('Input of ratio features')
global_sentiment_polarity = st.text_input("Text sentiment polarity(0-1)")
n_non_stop_unique_tokens = st.text_input("Rate of unique non-stop words in the content(0-1)")
global_rate_positive_words = st.text_input("Rate of positive words in the content(0-1)")
global_subjectivity = st.text_input("Text subjectivity(0-1)")
n_unique_tokens = st.text_input("Rate of unique words in the content(0-1)")


button = st.button("Submit")




keylist = ['global_sentiment_polarity', 'num_hrefs', 'n_non_stop_unique_tokens',
       'kw_avg_min', 'kw_min_avg', 'global_rate_positive_words',
       'n_unique_tokens', 'global_subjectivity', 'kw_max_min',
       'n_tokens_content', 'kw_max_avg']

keydic = {}



if button:
    for i in keylist:
        key = i
        val =[float(eval(i))]
        keydic.update({key: val})
    print(keydic)

    df = pd.DataFrame(keydic)
    pro = rf.predict_proba(df)[0, 1]
    res = round(pro*100)
    st.subheader('The likelihood of this news article being popular is {}%'.format(res))

    pass