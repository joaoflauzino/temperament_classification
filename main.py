import pandas as pd

# Pré-Processamento
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Fluxo de Pré-Processamento + Extração de Features
from processing.pre_processing import preprocess_nlp

# Pipeline de modelos a serem executados
from modeling.classification import classification_models

# Modelos que serão executados
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# Leitura da base
def leitura_base():
    return pd.read_csv('dataset/NoThemeTweets.csv', sep = ",")

# Adição de stopwords no dicionário
def add_stop_words(prepro_imdb):
    stops = ['mim', 'eh', 'vamo', 'deu', 'tb', 'pro', 'oi', 'oq']
    for i in stops:
        prepro_imdb.add_stopword(i)

# Chamando o pré-processamento de texto
def pre_processing(prepro_imdb):
    return prepro_imdb.preprocess()

# Encoder
def encoder(df):
    le = preprocessing.LabelEncoder()
    le.fit(df['sentiment'].unique())
    df['sentiment'] = le.transform(df['sentiment'])
    return df['sentiment']

# Chamando classificadores
def apply_classifiers(mod):
    mod.k_fold()
    mod.reports()
    
# Main
def main(models):
    df = leitura_base()
    df = df.sample(frac=0.001, replace=True, random_state=20)
    prepro_imdb = preprocess_nlp(df['tweet_text'], lemma=False, wordcloud=True, numeric='tfidf', ngram=3)
    add_stop_words(prepro_imdb)
    matrix = pre_processing(prepro_imdb)
    mod = classification_models(matrix.todense(), encoder(df), models)
    apply_classifiers(mod)
    
if __name__ == '__main__':
    
    models = [
    ("RandomForest", RandomForestClassifier()),
    ("LogisticRegression", LogisticRegression(max_iter = 10000)),
    ("SVC", SVC()),
    ("KNeighborsClassifier", KNeighborsClassifier()),
    ("MultinomialNB", MultinomialNB())
    ]
    
    main(models)