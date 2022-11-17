import streamlit as st
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
import extra_streamlit_components as stx

# Function of loading model
@st.cache(allow_output_mutation=True)
def load_model(model):
    algo_model = pickle.load(open(model,'rb'))
    return algo_model

# Function of loading vecorizer
@st.cache(allow_output_mutation=True)
def load_model2(model):
    vectorizer_model = pickle.load(open(model,'rb'))
    return vectorizer_model

#----------------------------------------------------
# Data Preprocessing :-
# Lower case
# Tokenization
# Removing special characters
# Removing stopwords and puctuation
# Stemming
ps = PorterStemmer()
# let's define a function that will be do all the above operation at a time
def text_transform(text):
    text = text.lower()  # lower case of all letters
    text = nltk.word_tokenize(text)   # separate the words
    
    y = []
    for i in text:
        if i.isalnum():       
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:   # stopwords and punctuation remove
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))    # stem the all words 
        
    return " ".join(y)   # at last join all words in a string

#-------------------------------------------------
selected = option_menu(menu_title =None,
                           options=['Home','Project Workflow'],
                           icons = ['',''],
                           orientation='horizontal',
                           styles={ 
                            "container": {"padding": "0!important", "background-color": "#000000"},
                            "icon": {"color": "white", "font-size": "15px"}, 
                            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#4a4a4a"},
                            "nav-link-selected": {"background-color": '#FF4B4B'},
                                }
                    )
if selected == 'Home':

    st.markdown("<h1 style='text-align: left; color: #ffffff;'> Email/SMS Spam Classification</h1>", unsafe_allow_html=True)
    st.caption('''Email communication has become crucial in today’s world. Email spam is a kind of unsolicited messages sent in bulk by email. A common terminology to describe an email as not spam is “Ham”, meaning an email is either Ham or Spam.
    Classify here if a message is Spam or not.            
    ''')

    st.write('\n')
    st.write('\n')

    text = st.text_area('Email Text')
    pred_button = st.button('Classify')

    if pred_button:
        if text == '':
            st.warning('Invalid input')
        else:
            transformed_sms = text_transform(text)   # Text sms preprocessing
        
            load_vectorize_model = load_model2('vectorizer.pkl')  # Load vectorize model
            vectorize_sms = load_vectorize_model.transform([transformed_sms])
            model_loading = load_model('model.pkl')   # Load model
            prediction = model_loading.predict(vectorize_sms)[0]
        
            if prediction == 1:
            
                st.warning('Spam')
            else:
                st.success('Not Spam')
            
            
                
if selected == 'Project Workflow':
    st.write('\n')
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'> Project Workflow</h1>", unsafe_allow_html=True)
    st.write('\n')
    # columns for shields 
    _ ,col2,_= st.columns([0.1,2,0.1])

    with col2:
        st.write('''[![Data kaggle](https://img.shields.io/badge/Data-Kaggle-blueviolet)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) 
                [![scikitlearn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)](https://scikit-learn.org/stable/tutorial/index.html) 
                [![Python 3.10.0](https://img.shields.io/badge/Python-3.10.0-brightgreen)](https://www.python.org/downloads/release/python-3100/) 
                [![Github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/Email-SMS-Spam-Classifier) 
                [![Streamlit 1.14.0](https://img.shields.io/badge/Streamlit%20-1.14.0-Ff0000)](https://docs.streamlit.io/) 
                [![Cloud Platform](https://img.shields.io/badge/CloudPlatform-Heroku-9cf)](https://www.heroku.com/managed-data-services)''')

    st.write('-----')

    # animated image loading function 
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    # columns for animated image and description of page
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='text-align: center; color: #fffff;'> Description</h3>", unsafe_allow_html=True)
        st.caption(f'''Email Spam and Ham Classification System is a simple Machine Learning Project.
                By collecting the data from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
                I started the project and then I 
                did some preprocessing on the dataset and build the final model. 
                It takes an Wmail or any other information as a input and classify that message is Spam or 
                Ham. For Classifying the EmailI used MultinomialNB algorithm. 
                For more information scroll down 
                and check out the Model Build page. Output of the code is not available 
                write all code in your notebook and run to see the output!
                ''')

    # url of animated image
    lotti_ur = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_w51pcehl.json')
    # second grid for animated image   
    with col2:
        st_lottie(lotti_ur,
            speed = 1,
            key=None,
            height=280,
            width=350,
            quality='high',
            reverse=False
            )

            
    st.write('-----')
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>  Steps </h2>", unsafe_allow_html=True)

    val = stx.stepper_bar(steps=["DataCollection🗂️", "Preprocessing👨‍💻", "Model Build🤖",'Website Build🌐','Deployment🎯'])

    if val == 0:
        st.write('----')
        st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Data Collection Processs</h2>", unsafe_allow_html=True)
        # columns create for align the text in middle
        col1, col2, col3 = st.columns([0.35,1,0.1])
        with col1:
            pass
        with col2:
            st.write('''[![Data Kaggle](https://img.shields.io/badge/Data-Kaggle-blueviolet)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
                    [![Size](https://img.shields.io/badge/Size-45.74mb-br)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
                    [![File Format](https://img.shields.io/badge/FileFormat-.csv-blue)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
                    [![Github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/Email-SMS-Spam-Classifier)''')
        
        st.write('\n')
        st.markdown(f'''<h4 style = 'text-align: left;'> Dependencies :</h4>''',
                        unsafe_allow_html=True)
        st.markdown(f'''* Jupyter Notebook''') 
        st.markdown(f'''* Python 3.10.0''')
        st.markdown(f'''* Pandas''')
        st.markdown(f'''* Numpy''')
        st.caption(f'''Install dependencies using [conda](https://docs.conda.io/en/latest/)''')
        
        st.markdown(f'''<h4 style = 'text-align: left;'> ⚙️Setup :</h4>''',
                        unsafe_allow_html=True)
        st.code('''import numpy as np
import pandas as pd''')
        st.markdown(f'''<h4 style = 'text-align: left;'> 🗂️Dataset import :</h4>''',
                        unsafe_allow_html=True)
        st.code('''df = pd.read_csv('spam.csv', encoding='latin1')  # import dataset
df.sample(5)   # random sample of 5 data''')
        st.code('df.info()')
        st.code('''# in unnamed 2,3,4 column almost all values are missing reove those three column
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True) 
df.head(2)''')
        st.code('''# let's rename the column name for better understanding
df = df.rename(columns={'v1':'target','v2':'text'})
df.head(2)''')
        st.code('''# label should be in numeric format so let me apply labelencoder to convert it in numeric
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df.head(5)''')
        st.code('''print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())''')
        st.code('''# in this dataset don't have any missing value but 403 rows are duplicate so, first remove those rows
df = df.drop_duplicates(keep='first')
df.shape''')
        st.code('print(df.duplicated().sum())')
    if val == 1:
        st.markdown(f'''<h4 style = 'text-align: left;'> 📈Exploratory Data Analysis :-</h4>''',
                        unsafe_allow_html=True)
        st.code('''import matplotlib.pyplot as plt
import seaborn as sns

df['target'].value_counts()''')
        st.code('''# pie plot of ham and spam msgs
plt.pie(df['target'].value_counts(),labels=('ham','spam'),autopct='%0.2f')
plt.show()''')
        st.code('# from this pie chat clearly visisble it is an imbalance datset, spam data is very less.')
        st.code('''import nltk
nltk.download('punkt')''')
        st.code('''# let's calculate number of chercters present in per rows..
df['num_cheracters'] = df['text'].apply(len)
df.tail(5)''')
        st.code('''# now lets calculat number of words present in per words....
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df.sample(5)''')
        st.code('''# now lets calculate the number of sentence per rows...
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))  # number of sentence per rows
df.sample(5)''')
        st.code('''df[df['target']==0][['num_cheracters','num_words','num_sentences']]''')
        st.code('''df[df['target']==0][['num_cheracters','num_words','num_sentences']].describe()''')
        st.code('''df[df['target']==1][['num_cheracters','num_words','num_sentences']]''')
        st.code('''df[df['target']==1][['num_cheracters','num_words','num_sentences']].describe()''')
        st.code('''sns.histplot(df[df['target']==0]['num_cheracters'])
sns.histplot(df[df['target']==1]['num_cheracters'],color='red')''')
        st.code('''sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')''')
        st.code('''sns.histplot(df[df['target']==0]['num_sentences'])
sns.histplot(df[df['target']==1]['num_sentences'],color='red')''')
        st.code('''sns.pairplot(df,hue='target')''')
        st.code('''# lets check the correlation 
df.corr()''')
        st.code('''# plot correlation 
sns.heatmap(df.corr(),annot=True)''')
        st.code('''# note : In time of model building if i added one manually created columns the i will go for num_cheracters column
# bcz it's having a strong correlation rather than others two.''')
    
        st.markdown(f'''<h7 style = 'text-align: left;'> **Data Preprocessing :-**

- Lower case
- Tokenization
- Removing special characters
- Removing stopwords and puctuation
- Stemming :-</h7>''',
                        unsafe_allow_html=True)
        st.markdown(f'''<h7 style = 'text-align: left;'> **Tokenization**
>Tokenization is basically splitting the sentences into words known as tokens. This is mainly one of the first steps to do when it comes to text classification.</h7>''',
                        unsafe_allow_html=True)
        st.image('https://i.ibb.co/mzQt0w3/blob.jpg')
        st.markdown(f'''<h7 style = 'text-align: left;'>**Stemming :**
>Stemming is a natural language processing technique that lowers inflection in words to their root forms, hence aiding in the preprocessing of text, words, and documents for text normalization.</h7>''',
                        unsafe_allow_html=True)
        st.image('https://i.ibb.co/b7h2qcf/stemminglemmatization-n8bmou.webp')
        st.write('\n')
        st.code('''from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()''')
        st.code('''# let's define a function that will be do all the above operation at a time
def text_transform(text):
    text = text.lower()  # lower case of all letters
    text = nltk.word_tokenize(text)   # separate the words
    
    y = []
    for i in text:
        if i.isalnum():       
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:   # stopwords and punctuation remove
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))    # stem the all words 
        
    return " ".join(y)   # at last join all words in a string''')
        st.code('''text_transform(df['text'][133])  # test the function''')
        st.code('''from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')''')
        st.code('''import string
string.punctuation   # punctuation list''')
        st.code('''from nltk.corpus import stopwords
stopwords.words('english') # stopwords list''')
        st.code('''df['transformed_text'] = df['text'].apply(text_transform)
df.head(5)''')
        st.code('#!pip install wordcloud')
        st.code('''# ham msgs most frequent words
from wordcloud import WordCloud
plt.figure(figsize=(8,6))
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
ham_wc = wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))
plt.imshow(ham_wc)''')
        st.code('''# spam msgs most frequent words
plt.figure(figsize=(8,6))
wc = WordCloud(width=400, height=400, min_font_size=10, background_color='white')
spam_wc = wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))
plt.imshow(spam_wc)''')
        st.code('''spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)''')
        st.code('''len(spam_corpus)''')
        st.code('''# most common 20 words in spam msgs
from collections import Counter
Counter(spam_corpus).most_common(20)
common_words_in_spam_msg = pd.DataFrame(Counter(spam_corpus).most_common(20))
common_words_in_spam_msg = common_words_in_spam_msg.rename(columns={0:'Words',1:'Counts'})
common_words_in_spam_msg''')
        st.code('''# barplot of frequent words
sns.barplot(common_words_in_spam_msg['Words'],common_words_in_spam_msg['Counts'])
plt.xticks(rotation=90)
plt.title('Most common 20 words barplot for Spam msgs')
plt.show()''')
        st.code('''ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for words in msg.split():
        ham_corpus.append(words)
len(ham_corpus)''')
        st.code('''from collections import Counter
Counter(ham_corpus).most_common(30)
# create a dataframe of ham_corpus
common_words_in_ham_msgs = pd.DataFrame(Counter(ham_corpus).most_common(30))
common_words_in_ham_msgs = common_words_in_ham_msgs.rename(columns={0:'Words',1:'Counts'})
common_words_in_ham_msgs''')
        st.code('''# plot this most common words in bar plot
sns.barplot(common_words_in_ham_msgs['Words'],common_words_in_ham_msgs['Counts'])
plt.xticks(rotation=90)
plt.title('Most common 30 words barplot for Ham msgs')
plt.show()''')
    
    if val ==2:
        st.markdown(f'''<h3 style = 'text-align: left;'> Model Building </h3>''',
                        unsafe_allow_html=True)
        st.write('\n')
        st.code('''# text vectorization using BoW technique
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)''')
        st.markdown(f'''<h7 style = 'text-align: left;'> **Tfidf  Vectorizer**
>TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. This is very common algorithm to transform text into a meaningful representation of numbers which is used to fit machine algorithm for prediction. For more information click [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). </h7>''',
                        unsafe_allow_html=True)
        st.write('\n')
        st.write('\n')
        st.markdown(f'''<h7 style = 'text-align: left;'>
                     Accuracy after using CountVectorizer 

>Accuracy score of the GaussianNB algorithm is  0.8684719535783365

>Confusion matrix of the GaussianNB algorithm is  [[772 117]
[ 19 126]]
                                                 
>Precision score of the GaussianNB algorithm is  0.5185185185185185

---
>Accuracy score of the MultinomialNB algorithm is  0.9738878143133463

>Confusion matrix of the MultinomialNB algorithm is  [[872  17]
 [ 10 135]]
 
>Precision score of the MultinomialNB algorithm is  0.8881578947368421

---
>Accuracy score of the BernoulliNB algorithm is  0.9661508704061895

>Confusion matrix of the BernoulliNB algorithm is  [[885   4]
 [ 31 114]]
 
>Precision score of the BernoulliNB algorithm is  0.9661016949152542

---
 Accuracy after using TfidfVectorizer

>Accuracy score of the GaussianNB algorithm is  0.8636363636363636

>Confusion matrix of the GaussianNB algorithm is  [[772 117]
 [ 24 121]]
 
>Precision score of the GaussianNB algorithm is  0.5084033613445378

---
>Accuracy score of the MultinomialNB algorithm is  0.9613152804642167

>Confusion matrix of the MultinomialNB algorithm is  [[888   1]
 [ 39 106]]
 
>Precision score of the MultinomialNB algorithm is  0.9906542056074766

---
>Accuracy score of the BernoulliNB algorithm is  0.9661508704061895

>Confusion matrix of the BernoulliNB algorithm is  [[885   4]
 [ 31 114]]
 
>Precision score of the BernoulliNB algorithm is  0.9661016949152542 </h7>''',
                        unsafe_allow_html=True)
        st.code('''x = tfidf.fit_transform(df['transformed_text']).toarray()
x.shape''')
        st.code('''# MinMaxScaler 
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)''')
        st.code('''y = df['target'].values
y.shape''')
        st.code('''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('Shape of x_train',x_train.shape)
print('Shape of x_test',x_test.shape)
print('Shape of y_train',y_train.shape)
print('Shape of y_test',y_test.shape)''')
        st.code('''from sklearn.naive_bayes import *

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()''')
        st.code('''from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score''')
        st.code('''gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print('Accuracy score of the GaussianNB algorithm is ',accuracy_score(y_test,y_pred1))
print('Confusion matrix of the GaussianNB algorithm is ',confusion_matrix(y_test,y_pred1))
print('Precision score of the GaussianNB algorithm is ',precision_score(y_test,y_pred1))''')
        st.code('''mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print('Accuracy score of the MultinomialNB algorithm is ',accuracy_score(y_test,y_pred2))
print('Confusion matrix of the MultinomialNB algorithm is ',confusion_matrix(y_test,y_pred2))
print('Precision score of the MultinomialNB algorithm is ',precision_score(y_test,y_pred2))''')
        st.code('''bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print('Accuracy score of the BernoulliNB algorithm is ',accuracy_score(y_test,y_pred3))
print('Confusion matrix of the BernoulliNB algorithm is ',confusion_matrix(y_test,y_pred3))
print('Precision score of the BernoulliNB algorithm is ',precision_score(y_test,y_pred3))''')
        st.code('#!pip install xgboost')
        st.code('''# Let's explore more algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier''')
        st.code('''lrc = LogisticRegression()
svc = SVC()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
bc = BaggingClassifier()
etc = ExtraTreesClassifier()
gbc = GradientBoostingClassifier()
xgc = XGBClassifier()''')
        st.code('''clfs = {
    'LR' : lrc,
    'SVC' : svc,
    'NB' : mnb,
    'DT' : dtc,
    'KN' : knc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC' : bc,
    'ETC' : etc,
    'GBDT' : gbc,
    'xgb' : xgc
}''')
        st.code('''def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision''')
        st.code('''def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision''')
        st.code('''etc.fit(x_train,y_train)
y_predicted  = etc.predict(x_test)
print('Accuracy - ',accuracy_score(y_test,y_predicted))
print('Confusion matrix - ',confusion_matrix(y_test,y_predicted))
print('Precision_score - ',precision_score(y_test,y_predicted))''')
        st.code('''accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, x_train,y_train,x_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)''')
        st.code('''# accuracy and precision score with max_features
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,
                               'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df''')
        st.write('\n')
        st.markdown(f'''<h3 style = 'text-align: left;'> Model Improve </h3>''',
                        unsafe_allow_html=True)
        st.write('\n')
        st.code('''# Model improve
# 1. change the max_features parameters to 1000 of Tfidf 
test_df1 = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_1000':accuracy_scores,
                        'Precision_max_ft_1000':precision_scores}).sort_values('Precision_max_ft_1000',ascending=False)
test_df1 = performance_df.merge(test_df1,on='Algorithm')
test_df1''')
        st.code('''# change the max_features parameters to 2000 of Tfidf 
test_df2 = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_2000':accuracy_scores,
                        'Precision_max_ft_2000':precision_scores}).sort_values('Precision_max_ft_2000',ascending=False)
test_df2 = test_df1.merge(test_df2,on='Algorithm')
test_df2''')
        st.code('''# change the max_features parameters to 2000 of Tfidf 
test_df3 = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,
                        'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
test_df3 = test_df2.merge(test_df3,on='Algorithm')
test_df3''')
        st.code('''# after apply minmaxscaler function
test_df4 = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_Scaling':accuracy_scores,
                        'Precision_Scaling':precision_scores}).sort_values('Precision_Scaling',ascending=False)
test_df4 = test_df3.merge(test_df4,on='Algorithm')
test_df4''')
        st.code('''# After check different algorithm , MinMaxScaler, and max features of tfidf I came an conclusion that MultinomialNB algorithm performing good in precision score and accuracy as well as, here precision score is imporatnt beacuse dataset is imbalance. So, after consider all the factors I decide to use MultinomialNB algorithm for the further process.''')
        st.markdown(f'''<h7 style = 'text-align: left;'>  **Save Model**
>By using pickle library saved the model in .pkl format so, i can load this .pkl file again in vs code to use it in Web Application   </h7>''',
                        unsafe_allow_html=True)
        st.code('''import pickle
pickle.dump(mnb,open('model.pkl','wb'))
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(text_transform,open('text_transform.pkl','wb'))''')
    if val == 3:
        st.write('----')
        st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Web Application Build</h2>", unsafe_allow_html=True)
        # columns create for align the text in middle
        col1, col2, col3 = st.columns([0.48,1,0.1])
        with col1:
            pass
        with col2:
            st.write('''[![Python](https://img.shields.io/badge/Python-3.10.0-brightgreen)](https://www.python.org/downloads/release/python-3100/)
                    [![Streamlit](https://img.shields.io/badge/Streamlit%20-1.14.0-Ff0000)](https://docs.streamlit.io/library/get-started)
                    [![github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/Email-SMS-Spam-Classifier)''')
        st.write('\n')
        st.markdown(f'''<h4 style = 'text-align: left;'> Dependencies :</h4>''',
                        unsafe_allow_html=True)
        st.markdown(f'''* Python 3.10.0''')
        st.markdown(f'''* Streamlit 1.14.0''')
        st.markdown(f'''* Pandas''')
        st.markdown(f'''* Numpy''')
      
        st.caption('''This website Application build by using Python, steamlit library. It's an secured🔐 safe Application.''')
        
        st.write('')
        st.subheader('Pages :')
        st.caption('''
                
                >web Application build with total 2page's which are shown below for individual page code is diiferent
                >. If you wanna check how I build
                >the full website and used the machine learning model(.pkl) file in it.
                >Check my [GitHub](https://github.com/Rafikul10/Email-SMS-Spam-Classifier) Repositories. Thank You!''')
        st.markdown(f'''* Classification Page''')
        st.markdown(f'''* Project Workflow page''') 
        st.write('---')
        st.write('\n')
        st.subheader('📬 Contact Details')
        st.caption('_If you need any help related website build or ML model build feel free to contact me!!!_')
        st.write(f'📧 ' ,' rafikul.official10@gmail.com',type='mail')
        st.write("👾  [GitHub](https://github.com/Rafikul10?tab=repositories)")
        
    if val == 4:
        st.write('---')    
        st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Deployment</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.39,1,0.1])
        with col1:
            pass
        with col2:
            st.write('''[![Cloud](https://img.shields.io/badge/CloudPlatform-Heroku-blue)](https://devcenter.heroku.com/categories/reference)
                    [![Streamlit](https://img.shields.io/badge/Streamlit%20-1.14.0-Ff0000)](https://streamlit.io/)
                    [![Github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/Email-SMS-Spam-Classifier)
                    ''')
        with col3:
            pass  
            
        st.caption('''Now Web Application is ready for deploy in any cloud server. In my case i choose Heroku cloud server for deploy the Application. In
                order to deploy the Application on Heroku first create an account and login to your account. After sucessfully logged in.
                You can deploy your application directly by downloading the CLI from official website of
                Heroku for all processs check [Heroku](https://devcenter.heroku.com/categories/reference) official website.''')
        
        st.caption('''NOTE : For all code check my [GitHub](https://github.com/Rafikul10/Email-SMS-Spam-Classifier) repositories.''')   
            
        
        