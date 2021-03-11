import random
import matplotlib.pyplot as plt
import warnings
import numpy 

# Bibliotecas para pre-processamento:
import re
import nltk
import spacy
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from unidecode import unidecode
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# # Bibliotecas para avaliação
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report 
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score

#%matplotlib inline
#warnings.filterwarnings('ignore')


class preprocess_nlp(object):
    
    def __init__(self, texts, stopwords=True, lemma=False, stem=False, wordcloud=True, numeric='tfidf', ngram=1):
        
        self.texts = texts
        self.stopwords = stopwords
        self.lemma = lemma
        self.stem = stem
        self.wordcloud = wordcloud
        self.numeric = numeric
        self.new_texts = None
        self.stopwords_list = list()
        self.ngram = ngram
        
        
    def clean_text(self):

        new_texts = list()

        for text in self.texts:

            text = text.lower()
            text = re.sub('@[^\s]+', '', text)
            text = unidecode(text)
            text = re.sub('<[^<]+?>','', text)
            text = text.replace('{', '').replace('}', '')
            text = ''.join(c for c in text if not c.isdigit())
            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
            text = ''.join(c for c in text if c not in punctuation)
            # Checar se é necessário tirar todas hashtag no texto
            new_texts.append(text)
        
        self.new_texts = new_texts

    def create_stopwords(self):
        
        stop_words = list(set(stopwords.words('portuguese') + list(STOP_WORDS)))
        
        for word in stop_words:

            self.stopwords_list.append(unidecode(word))
       
    
    def add_stopword(self, word):
        
        self.stopwords_list += [word]
        

    def remove_stopwords(self):

        new_texts = list()

        for text in self.new_texts:

            new_text = ''

            for word in word_tokenize(text):

                if word.lower() not in self.stopwords_list:

                    new_text += ' ' + word

            new_texts.append(new_text)

        self.new_texts = new_texts


    def extract_lemma(self):
        
        nlp = spacy.load("pt_core_news_sm")
        new_texts = list()

        for text in self.texts:

            new_text = ''

            for word in nlp(text):

                new_text += ' ' + word.lemma_

            new_texts.append(new_text)
        
        self.new_texts = new_texts
    

    def extract_stem(self):

        stemmer = nltk.stem.SnowballStemmer('english')
        new_texts = list()

        for text in self.texts:

            new_text = ''

            for word in word_tokenize(text):

                new_text += ' ' + stemmer.stem(word)

            new_texts.append(new_text)

        self.new_texts = new_texts
    

    def word_cloud(self):

        all_words = ' '.join([text for text in self.new_texts])
        word_cloud = WordCloud(width= 800, height= 500,
                               max_font_size = 110,
                               collocations = False).generate(all_words)
        plt.figure(figsize=(12,6))
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        

    def countvectorizer(self):

        vect = CountVectorizer()
        text_vect = vect.fit_transform(self.new_texts)

        return text_vect
    

    def tfidfvectorizer(self):

        vect = TfidfVectorizer(max_features=50, ngram_range=(1,self.ngram))
        text_vect = vect.fit_transform(self.new_texts)

        return text_vect
    
    
    def preprocess(self):

        self.clean_text()
        
        if self.stopwords == True:
            self.create_stopwords()
            self.remove_stopwords()
            
        if self.lemma == True:
            self.extract_lemma()
        
        if self.stem == True:
            self.extract_stem() 

        if self.wordcloud == True:
            self.word_cloud()
        
        if self.numeric == 'tfidf':
            text_vect = self.tfidfvectorizer()
        elif self.numeric == 'count':
            text_vect = self.countvectorizer()
        else:
            print('metodo nao mapeado!')
            exit()
            
        return text_vect, self.new_texts