# In[]
import numpy as np
from collections import defaultdict
import operator
from collections import OrderedDict, Counter
from operator import itemgetter
import pandas
from sklearn.linear_model import Ridge
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
from nltk.sentiment.vader import allcap_differential
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import os
os.chdir('/home/kolassc/Desktop/ucsd_course_materials/CSE258/datasets/yelp/')

# In[]
yelp_business_path = 'yelp_academic_dataset_business.json'
yelp_review_path = 'yelp_academic_dataset_review.json'
yelp_user_path = 'yelp_academic_dataset_user.json'
# In[]
# read data
def parseDataB(file):
    null=None
    with open(file, errors='ignore') as f:
        for l in f:
            yield eval(l)

def loadDataB(f, read_limit=1000000):
    return  list(parseDataB(f))


# In[]
business_data = loadDataB(yelp_business_path)
# In[]
business_data_id = defaultdict(int)
i = 0
for b in business_data:
    business_data_id[b['business_id']] = i
    i += 1


# In[]
train = np.load("train.npy")
valid = np.load("hold.npy")
test = np.load("test.npy")

punctuation = set(string.punctuation)
stopwordList = stopwords.words('english')

def get_year(d):
    return int(d.split('-')[0])

def getMonth(s):
    return int(s.split("-")[1])

def year_one_hot(d):
    year = [0]*6
    y = get_year(d)
    if y>min_year:
        year[y-min_year-1]=1
    return year
def getCityOneHot(city):
    index = top_cities_map[city]
    vec = [0]*4
    if index > 0:
        vec[index-1] = 1
    return vec
def reviewCounts(s):
    words = s.split()
    nWords = len(words)
    nSentences = len(s.split("."))
    nChars = len(s)
    nPunctuations  = len([w for w in words if w in punctuation])
    nExclamations = len([w for w in words if w=='!'])
    nAllCaps = allcap_differential(words)
    nTitleWords = len([w for w in words if w[0].isupper()])
    return {"nWords":nWords, "nSentences":nSentences,
            "nChars":nChars, "nPunctuations":nPunctuations,
            "nExclamations":nExclamations,"nAllCaps":nAllCaps,
            "nTitleWords":nTitleWords}
# In[]
all_train_text = []
for d in train:
    b = business_data[business_data_id[d['business_id']]]
    if(b['review_count']>10):
        all_train_text.append(d['text'])

unigram = CountVectorizer(max_features=200,stop_words='english')
X_train_unigram = unigram.fit_transform(all_train_text).toarray()

bigram = CountVectorizer(max_features=200,stop_words='english',ngram_range=(2,2))
X_train_bigram = bigram.fit_transform(all_train_text).toarray()

trigram = CountVectorizer(max_features=200,stop_words='english',ngram_range=(3,3))
X_train_trigram = trigram.fit_transform(all_train_text).toarray()

mixed_gram = CountVectorizer(max_features=500,stop_words='english',ngram_range=(1,3))
mixed_gram.fit(all_train_text)

X_train = np.concatenate((X_train_unigram,X_train_bigram,X_train_trigram),axis=1)
X_train = np.insert(X_train,X_train.shape[1],1,axis=1)
del all_train_text


# In[]
def feature(data):
    review_text_all=[]
    for d in data:
        review_text_all.append(d['text'])
    feature_unigram = unigram.transform(review_text_all).toarray()
    feature_bigram = bigram.transform(review_text_all).toarray()
    feature_trigram = trigram.transform(review_text_all).toarray()
    feature_all = np.concatenate((feature_unigram,feature_bigram,feature_trigram),axis=1)
    feature_all = np.insert(feature_all,feature_all.shape[1],1,axis=1)
    del review_text_all
    return feature_all

def label(r):
    return r['stars']
# In[]
y_train = []
for d in train:
    b = business_data[business_data_id[d['business_id']]]
    if(b['review_count']>10):
        y_train.append([label(d)])

X_valid = feature(valid)
y_valid = []
for d in valid:
    y_valid.append([label(d)])
# In[]
X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

# In[]
clf_l = Ridge(alpha=1, fit_intercept = False, solver='lsqr')
clf_l.fit(X_train,y_train)
predict = clf_l.predict(X_valid)
theta = clf_l.coef_
print(theta)
rmse = np.sqrt(np.average(np.square(y_valid - predict)))
print("RMSE = ", rmse)
# In[]

X_train_tf = tf.constant(X_train, shape=X_train.shape, dtype=tf.float32)
y_train_tf = tf.constant(y_train, shape=y_train.shape, dtype=tf.float32)
X_valid_tf = tf.constant(X_valid, shape=X_valid.shape, dtype=tf.float32)
y_valid_tf = tf.constant(y_valid, shape=y_valid.shape, dtype=tf.float32)
# In[]
def RMSE_regularized(X, y, theta, lamb):
  return np.sqrt(tf.reduce_mean((tf.matmul(X,theta) - y)**2) + lamb*tf.reduce_sum(theta**2))

# In[]
t = np.zeros((len(X_train[0]),1))
theta = tf.Variable(tf.constant(t, shape=[len(X_train[0]),1], dtype=tf.float32))
# In[]

# Stochastic gradient descent
optimizer = tf.train.AdamOptimizer(0.01)
# The objective we'll optimize is the MSE
objective = RMSE_regularized(X_train_tf,y_train_tf,theta, 0.0)

# Our goal is to minimize it
train = optimizer.minimize(objective)

# Initialize our variables
init = tf.global_variables_initializer()

# Create a new optimization session
sess = tf.Session()
sess.run(init)
# Run 20 iterations of gradient descent

prev_valid_RMSE = np.inf
early_stop = 3
for iteration in range(2000):
    
    cvalues = sess.run([train, objective])
    print("objective = " + str(cvalues[1]))
  
    with sess.as_default():
        cur_valid_RMSE = RMSE_regularized(X_valid_tf, y_valid_tf, theta, 0.0).eval()
        print(cur_valid_RMSE)
        if iteration>100:
            if prev_valid_RMSE>cur_valid_RMSE:
                cur_valid_RMSE = prev_valid_RMSE
                early_stop = 3
            else:
                early_stop -= 1
        
        if early_stop == 0:
            break


# Print the outputs
with sess.as_default():
    print(RMSE_regularized(X_train_tf, y_train_tf, theta, 0.0).eval())
    print(theta.eval())
# In[]
