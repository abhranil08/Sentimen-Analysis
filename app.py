from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home1.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("Restaurant_Reviews.tsv",sep="\t")
	#df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	#df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    """X = df['Review']  
    y = df['Liked']
	# Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)"""
	#Alternative Usage of Saved Model
    #joblib.dump(clf, 'NB_spam_model.pkl')
    X = df['Review']
    y = df["Liked"]
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    NB_spam_model = open('NB_spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)
    pos = 0
    neg = 2
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction,pos = pos , neg = data)

@app.route('/homenew.html',methods=['GET','POST'])
def homenew():
    df = pd.read_csv("Restaurant_Reviews.tsv",sep="\t")
    X = df['Review']
    y = df["Liked"]
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    NB_spam_model = open('NB_spam_model.pkl','rb')
    clf = joblib.load(NB_spam_model)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        pos = 0
        neg = 2
        return render_template('result.html',prediction = my_prediction,pos = pos , neg = neg )
    else:
        return render_template("homenew.html")
        
    

if __name__ == '__main__':
	app.run(debug=True)