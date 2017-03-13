# In[]
import numpy as np
from collections import defaultdict
import operator
from collections import OrderedDict, Counter
from operator import itemgetter
import pandas
from sklearn.linear_model import Ridge
from nltk.corpus import stopwords
import string
from nltk.sentiment.vader import allcap_differential
import tensorflow as tf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# In[]
yelp_business_path = 'yelp_academic_dataset_business.json'
yelp_review_path = 'yelp_academic_dataset_review.json'
yelp_user_path = 'yelp_academic_dataset_user.json'
sid = SentimentIntensityAnalyzer()
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
top_cities = ['Pittsburgh','Las Vegas','Phoenix','Charlotte','Toronto']
top_cities_map = {top_cities[0]:0, top_cities[1]:1, top_cities[2]:2, top_cities[3]:3, top_cities[4]:4}
top_cities_review = defaultdict(list)
min_year = 2010
max_year = 2016
# In[]
def parseData(file):
    global top_cities_review
    null = None
    with open(file, errors='ignore') as f:
        for l in f:
            r = eval(l)

            b = business_data[business_data_id[r['business_id']]]

            if b['categories']!=None and b['city'] in top_cities and 'Restaurants' in b['categories'] and (min_year<=int(r['date'].split('-')[0])<=max_year):
               top_cities_review[b['city']].append(r)

def loadData(f):
    parseData(f)
# In[]

# In[]
reviews, train, valid, test = [], [], [], []
for x in top_cities_review.keys():
    reviews += top_cities_review[x]

np.random.shuffle(reviews)

# In[]
np.save('reviews',reviews)
# In[]
train = reviews[:50000]
valid = reviews[50000:100000]
test = reviews[100000:150000]
# In[]
train = np.load("train.npy")
valid = np.load("hold.npy")
# In[]
train_users = set()
for d in train:
    #print(d)
    train_users.add(d['user_id'])

# In[]
def parseDataU(file):
    null=None
    with open(file, errors='ignore') as f:
        for l in f:
            u = eval(l)
            if u['user_id'] in train_users:
                yield u

def loadDataU(f):
    return  list(parseDataB(f))


# In[]
users = list(parseDataU(yelp_user_path))
# In[]
users_dict = defaultdict()
for u in users:
    users_dict[u['user_id']] = u
# In[]
#np.save('train',train)
#np.save('valid',valid)
#np.save('test',test)
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

def getSenti(d):
    senti = sid.polarity_scores(d['text'])
    return senti['compound']

# In[]
def feature(r, b):
    f = []
    f.append(1)
    f += year_one_hot(r['date'])
    rew = reviewCounts(r['text'])
    f.append(rew['nWords'])
    #f.append(rew['nExclamations'])
    #f.append(rew['nAllCaps'])
    #f.append(rew['nPunctuations'])
    f.append(1 if getMonth(r['date'])==12 else 0)
    #b = business_data[business_data_id[r['business_id']]]

    f += getCityOneHot(b['city'])
    f.append(b['stars'])
    f.append(b['review_count'])
    userid = r['user_id']
    
    if userid in train_users:
        f.append(users_dict[userid]['average_stars'])
    else:
        f.append(3.6)
    
    #f.append(getSenti(r))
    return f

def label(r):
    return r['stars']
# In[]
X_train = []
y_train = []
for d in train:
    b = business_data[business_data_id[d['business_id']]]
    #if(b['review_count']>10):
    X_train.append(feature(d, b))
    y_train.append([label(d)])
X_valid = []
y_valid = []

# In[]
for d in valid:
    b = business_data[business_data_id[d['business_id']]]
    X_valid.append(feature(d,b))
    y_valid.append([label(d)])
# In[]
X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

# In[]
clf_l = Ridge(alpha=0.1, fit_intercept = False, solver='auto')
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
  return tf.reduce_mean((tf.matmul(X,theta) - y)**2) + lamb*tf.reduce_sum(theta**2)

# In[]
#t = np.random.random((len(X_train[0]),1)) - 0.5
t = np.zeros((len(X_train[0]),1))
theta = tf.Variable(tf.constant(t, shape=[len(X_train[0]),1], dtype=tf.float32))


# Stochastic gradient descent
optimizer = tf.train.AdamOptimizer(0.01)
# The objective we'll optimize is the MSE
objective = RMSE_regularized(X_train_tf,y_train_tf,theta, 0.1)

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
    if(iteration%10==0):
        print("objective = " + str(cvalues[1]))
  
    with sess.as_default():
        cur_valid_RMSE = RMSE_regularized(X_valid_tf, y_valid_tf, theta, 0.0).eval()
        cur_valid_RMSE = np.sqrt(cur_valid_RMSE)
        if(iteration%10==0):
            print(cur_valid_RMSE)
        if iteration>100:
            if prev_valid_RMSE>cur_valid_RMSE:
                cur_valid_RMSE = prev_valid_RMSE
                early_stop = 3
            else:
                early_stop -= 1
        
        if early_stop == 0:
            print('DONE')
            break


# Print the outputs
with sess.as_default():
    print(RMSE_regularized(X_train_tf, y_train_tf, theta, 0.0).eval())
    print(theta.eval())
# In[]
