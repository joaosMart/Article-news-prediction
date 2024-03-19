import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
# train_test
from sklearn.model_selection import train_test_split

# CV
from sklearn.feature_extraction.text import CountVectorizer

# TF-idF
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('stopwords')

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import scipy.sparse as sparse

from keras.models import load_model
# LSD

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
#!pip install scikit-multilearn==0.2.0
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import multilabel_confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from skmultilearn.problem_transform import ClassifierChain
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential

import unicodedata

# Importing the pickle library

import pickle

from flask import Flask,render_template,request,send_file,send_from_directory,jsonify

import numpy as np

import zipfile
from zipfile import ZipFile


def strip_accents(text):
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|„|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def stemming(sentence):
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        if (len(stem) > 2): # small edit
          stemSentence += stem
          stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


def removeStopWords(sentence):
    global re_stop_words

    stop_words = set(stopwords.words('english'))
    stop_words.update(['zero','one','two',
                      'three','four','five',
                      'six','seven','eight',
                      'nine','ten','may',
                      'also','across','among',
                      'beside','however','yet',
                      'within','since'])

    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    
    return re_stop_words.sub(" ", sentence)

def preprocessing(text):
  # just do everything in one function
  text = strip_accents(text)
  text = cleanPunc(text)
  text = removeStopWords(text)
  text = stemming(text)
  return text

df = pd.read_csv('assm_4.csv')
df = df.dropna(axis = 0).drop('Unnamed: 0', axis = 1)

# One hot Encoding
df['topics'] = df['topics'].str.replace('[', '')
df['topics'] = df['topics'].str.replace(']', '')
df['topics'] = df['topics'].str.replace("' ", '')
df['topics'] = df['topics'].str.replace("'", '')

df_dummy = (df['topics'].str.replace(", ", ',')   # remove all spaces
    .str.get_dummies(',')            # get the dummies
)

df = pd.concat([df,df_dummy], axis = 1)

# Remove not necessary
df = df.drop('Archive', axis = 1)
df_dummy = df_dummy.drop('Archive',axis=1)

# so we don't run into any trouble
df["headline"] = df["headline"].astype(str)
df["body"] = df["body"].astype(str)

df["processed"] = df["body"].apply(lambda x : preprocessing(x))

tc = TfidfVectorizer( stop_words='english',
                      max_features= 1500, # found with experimentation
                      max_df = 0.75,
                      smooth_idf=True)
X = tc.fit_transform(df["processed"])

svd_model = TruncatedSVD(n_components = 500,
                         algorithm='randomized',
                         n_iter=100, 
                         random_state=122)
X = svd_model.fit_transform(X)

model = Sequential()
model.add(layers.Dense(250, input_dim=X.shape[1], kernel_initializer='he_uniform',activation='relu'))
model.add(layers.Dense(150,activation='relu'))
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(9,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

y = df_dummy.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)

history = model.fit(X_train, y_train, batch_size = 16, epochs = 10, validation_data = (X_test, y_test))

y_pred = (model.predict(X_test)).round()

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

confusion = multilabel_confusion_matrix(y_test.astype(float).argmax(axis=1), 
                                        y_pred.astype(float).argmax(axis=1))
print('Confusion Matrixes: \n')

for i, j in zip(df_dummy.columns, confusion):
  print('\n' + i + ':')
  print(j)

  
# Dumping the model object to save it as model.pkl file

pickle.dump(tc, open('tfidf.pkl', 'wb+'))
pickle.dump(svd_model,open('model_svd.pkl','wb+'))
model.save('model.h5')

