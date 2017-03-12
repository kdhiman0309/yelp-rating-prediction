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
top_cities = ['Pittsburgh','Las Vegas','Phoenix','Charlotte','Toronto']
top_cities_map = {top_cities[0]:0, top_cities[1]:1, top_cities[2]:2, top_cities[3]:3, top_cities[4]:4}
top_cities_review = defaultdict(list)
min_year = 2010
max_year = 2016
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

loadData(yelp_review_path)

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

def feature(r):
    f = []
    f.append(1)
    f += year_one_hot(r['date'])
    rew = reviewCounts(r['text'])
    f.append(rew['nWords'])
    f.append(rew['nExclamations'])
    f.append(rew['nAllCaps'])
    f.append(rew['nPunctuations'])
    f.append(1 if getMonth(r['date'])==12 else 0)
    b = business_data[business_data_id[r['business_id']]]

    f += getCityOneHot(b['city'])
    f.append(b['stars'])
    f.append(b['stars'])
    f.append(b['review_count'])
    return f
    
def label(r):
    return r['stars']
# In[]
train = np.load("train.npy")
valid = np.load("hold.npy")
# In[]
X_train = [feature(d) for d in train]
y_train = [label(d) for d in train]
X_valid = [feature(d) for d in valid]
y_valid = [label(d) for d in valid]

# In[]
clf_l = Ridge(alpha=10, fit_intercept = False, solver='lsqr')
clf_l.fit(X_train,y_train)
predict = clf_l.predict(X_valid)
theta_l = clf_l.coef_
rmse = np.sqrt(np.average(np.square(y_valid - predict)))
print("RMSE = ", rmse)

    
    
    
    