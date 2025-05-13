import streamlit as st
import string
import nltk
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import pickle

ps = PorterStemmer()
def text_transformation(text):
    text = text.lower()
    text = word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    x = []
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(ps.stem(i))
    return " ".join(x)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')
input_sms = st.text_area('Enter the massage')

if st.button('Predict'):

    # 1. Preprocess
    transformed_text = text_transformation(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_text])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

# from nltk.tokenize import word_tokenize
# text = "Hello! How are you?"
# print(word_tokenize(text))