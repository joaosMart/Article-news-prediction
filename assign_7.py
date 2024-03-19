# Importing the pickle library

import pickle
from flask import Flask,render_template,request,send_file,send_from_directory,jsonify
import numpy as np
import zipfile
from zipfile import ZipFile
import nltk

nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('stopwords')

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import scipy.sparse as sparse
from nltk.stem.snowball import SnowballStemmer
import unicodedata
from nltk.corpus import stopwords
from keras.models import load_model
   

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

# html and style templates
ZipFile('templates.zip','r').extractall()
ZipFile('static.zip','r').extractall()

# init flask
app = Flask(__name__)

# load models
svd_model=pickle.load(open('model_svd.pkl','rb+'))
model = load_model('model.h5')
tfidf =pickle.load(open('tfidf.pkl','rb+'))

# home page
@app.route('/')
def home():
	return render_template('home.html')

# predict
@app.route('/predict',methods=['POST'])
def predict():
  # Categries gor a good clean get
  columns = np.array(['Business', 'Culture', 'Nature', 'Podcast', 'Politics', 'Sci&Tech','Society', 'Sport', 'Travel'])
  
  if request.method == 'POST':
    # get user input
    s = request.form['message']

    # process and predict
    sample = [preprocessing(s)]
    X = tfidf.transform(sample)
    inp = svd_model.transform(X)
    pred = model.predict(inp)
    b = pred.round().astype(bool)[0]

    # get output ready 
    if sum(b) > 0:
        output = ""
        for i in columns[b]:
    	    output = output + " " + i
    else:
    	output = "This has no category"
  return render_template('result.html', prediction = output)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)



