import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.special import softmax

st.set_page_config(page_title='Sentiment Analysis',layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'> Sentiment Analysis </h1>", unsafe_allow_html=True)





text = st.text_area("Write a review here:", height=100)


filename='model1.pkl'
model1 = pickle.load(open(filename,'rb')) 

filename='model2.pkl'
model2 = pickle.load(open(filename,'rb')) 

filename='tokenizer.pkl'
tokenizer = pickle.load(open(filename,'rb')) 

if text:
    if st.button("Analysis by bag of word technique"):
        result=model1.polarity_scores(text)
        compound=result['compound']
        if (compound<0.1 and compound<-0.1):
            review='Neutral'
        if (compound>0.1 and compound<0.5):
            review='Fairly Positive'
        if (compound>0.5):
            review='Positive'
        if (compound< -0.1 and compound>-0.5):
            review='Fairly Negative'
        if (compound<-0.5):
            review='Negative'
        st.success(f"Your review is {review}")
    
    if st.button("Analysis using transformers(hugging face model)"):
        encoded_text=tokenizer(text,return_tensors='pt')
        output=model2(**encoded_text)
        scores=output[0][0].detach().numpy()
        scores_new=softmax(scores)
        scores_dict={'Negative':scores_new[0],'Neutral':scores_new[1],'Positive':scores_new[2]}
        result = max(scores_dict.values(), default=None)
        for key, value in scores_dict.items():
           if value == result:
               review=key
                
        st.success(f"Your review is {review}")



