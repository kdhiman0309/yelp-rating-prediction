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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
os.chdir('/home/kolassc/Desktop/ucsd_course_materials/CSE258/datasets/yelp')

# In[]
yelp_business_path = 'yelp_academic_dataset_business.json'
yelp_review_path = 'yelp_academic_dataset_review.json'
yelp_user_path = 'yelp_academic_dataset_user.json'
sid = SentimentIntensityAnalyzer()
# In[]
# read data
def parseDataB(file,filter_set=set(),filter_parameter="",filter_value=0):
    null=None
    with open(file, errors='ignore') as f:
        if len(filter_parameter) == 0:
            for l in f:
                yield eval(l)
        else:
            for l in f:
                l = eval(l)
                if l[filter_parameter] in filter_set:
                    yield l
                    
            

def loadDataB(f, read_limit=1000000):
    return  list(parseDataB(f))

def loadDataBFiltered(f, filter_set=set(),filter_parameter="",filter_value=0,read_limit=1000000):
    return  list(parseDataB(f,filter_set,filter_parameter,filter_value))

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
valid = reviews[:50000]
test = reviews[50000:100000]

users_set =  set([r['user_id'] for r in train])

# In[]
user_data = loadDataBFiltered(yelp_user_path,users_set,'user_id',10)

# In[]
user_data_id = defaultdict(int)
i = 0
for u in user_data:
    user_data_id[u['user_id']] = i
    i+= 1
# In[]
train_tmp=[]
for d in train:
    if user_data[user_data_id[d['user_id']]]['review_count'] > 10:
        train_tmp.append(d)
train = train_tmp
# In[]
np.save('train_filtered',train)
np.save('valid_filtered',valid)
np.save('test_filtered',test)
