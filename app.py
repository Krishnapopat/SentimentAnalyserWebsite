from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from collections import Counter

df=pd.read_csv("C:\\Users\\DELL\\Downloads\\full-corpus.csv")

df = df[df["Sentiment"] != "irrelevant"]

df_copy = df.copy()
df_copy.loc[:, "TweetText"] = df["TweetText"].astype(str)
df_copy.loc[:, "TweetText"] = df["TweetText"].str.replace('[^a-zA-Z0-9\s]', '', regex=True)
df=df_copy

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

df["TweetText"] = df["TweetText"].apply(tokenize_and_lemmatize)

x=df["TweetText"]
y=df["Sentiment"]

X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y ,test_size=0.2, random_state=42,shuffle=True)

X_train1 = X_train1.apply(' '.join)
X_test1 = X_test1.apply(' '.join)

X_train1 = X_train1.tolist()
X_test1 = X_test1.tolist()


vectorizer = TfidfVectorizer(min_df=1, stop_words=None, lowercase=True)
X_train1 = vectorizer.fit_transform(X_train1)
X_test1 = vectorizer.transform(X_test1)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train1, y_train1)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    inp = request.form['sentence']
    inp=tokenize_and_lemmatize(inp)
    inp=np.array(inp)
    print(inp)
    inp = vectorizer.transform(inp)

    pred=svm_classifier.predict(inp)

    print(pred)
    sentiment_counts = Counter(pred)

    ANS, count = sentiment_counts.most_common(1)[0]

    if count >= 1 and "neutral" in sentiment_counts and sentiment_counts["neutral"] == count:
        if "positive" in sentiment_counts:
            ANS = "Positive"
        elif "negative" in sentiment_counts:
            ANS= "Negative" 
    if ANS=="neutral":
        ANS="Neutral"
    elif ANS=="negative":
        ANS="Negative"
    elif ANS=="positive":
        ANS="Positive"
    return jsonify({'sentiment': ANS})

def mock_sentiment_analysis(sentence):
    # Replace this with your actual sentiment analysis logic.
    sentiments = ['Positive', 'Neutral', 'Negative']
    return sentiments[hash(sentence) % len(sentiments)]

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
