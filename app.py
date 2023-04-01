pip install flask
from flask import  Flask , render_template , request
import pandas as pd
import streamlit as st
app = Flask(__name__)

import pickle

mnb = pickle.load(open('multimodel.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))
dict1 = pickle.load(open('dictionary.pkl','rb'))
df = pd.DataFrame(dict1)
similarity = pickle.load(open('similarity.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

@app.route('/recommendation')
def recommend():
    return render_template('recommendation.html')






import string
exclude = string.punctuation
def remove_punc(text):
    return text.translate(str.maketrans('', '', exclude))

from nltk.corpus import stopwords
sw = stopwords.words('english')

def remove_stopwords(text):
    x = []
    for word in text.split():
        if word in sw:
            x.append('')
        else:
            x.append(word)
    return ' '.join(x)



@app.route('/predict' ,methods=['POST'])
def predict():
    text = request.form.get('query')
    text = text.lower()
    text = remove_punc(text)
    text = remove_stopwords(text)
    vector_input = tfidf.transform([text])
    output = mnb.predict(vector_input)[0]

    if output == 1:
        return render_template('sentiment.html' , label=1)
    else:
        return render_template('sentiment.html' , label=-1)



def recommend(title):
    movie_index = df[df['title'] == title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendation_list = []
    for i in movie_list:
        recommendation_list.append(df['title'][i[0]])
        print(df['title'][i[0]])
    print(df['title'].unique()[1:10])
    return recommendation_list


@app.route('/list')
def movieslist():
    titles = sorted(dict1['title'].values())
    return render_template('recommendation.html' , titles=titles)

@app.route('/predict_movie' , methods=['POST'])
def predict_movie():
    #titles = sorted(dict1['title'].values())
    movie = request.form.get('title')
    movie_list = recommend(movie)
    return render_template('recommendation.html', movie_list = movie_list)

if __name__ == '__main__':
    app.run(use_reloader = True , debug=True)

