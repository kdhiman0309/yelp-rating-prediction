# In[]
import numpy as np
from collections import defaultdict
import operator
from collections import OrderedDict, Counter
from operator import itemgetter
import pandas
from sklearn.feature_extraction.text import CountVectorizer
import string
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

os.chdir('/home/kolassc/Desktop/ucsd_course_materials/CSE258/datasets/yelp/')

# In[]
yelp_business_path = 'yelp_academic_dataset_business.json'
yelp_review_path = 'yelp_academic_dataset_review.json'
yelp_user_path = 'yelp_academic_dataset_user.json'

# In[]
min_year = 2010

# read data
def parseData(file, city, read_limit):
    null = None
    with open(file, errors='ignore') as f:
        i=0
        for l in f:
            if i<read_limit:
                x = eval(l)
                b = business_data[business_data_id[x['business_id']]]
                if b['city']==city and int(x['date'].split('-')[0])>=min_year:
                    if b['categories']!=None and 'Restaurants' in b['categories']:
                        i+=1
                        yield x
            else :
                break
            
def loadData(f, city, read_limit=3000):
    return  list(parseData(f, city, read_limit))

# In[]
# read data
def parseDataB(file):
    null=None
    with open(file, errors='ignore') as f:
        for l in f:
            yield eval(l)
            
def loadDataB(f, read_limit=3000):
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
review_data_lasvegas = loadData(yelp_review_path, city='Las Vegas')
review_data_toronto = loadData(yelp_review_path, city='Toronto')

# In[]
median_lasvegas = np.median([x['stars'] for x in review_data_lasvegas])
median_toronto = np.median([x['stars'] for x in review_data_toronto])

# In[]
c_lasvegas = Counter([x['stars'] for x in review_data_lasvegas])
c_toronto = Counter([x['stars'] for x in review_data_toronto])

for key in c_lasvegas:
    c_lasvegas[key] /= len(review_data_lasvegas)
    
for key in c_toronto:
    c_toronto[key] /= len(review_data_toronto)

df = pandas.DataFrame.from_dict(c_lasvegas, orient='index')
df.plot(kind='bar')
df = pandas.DataFrame.from_dict(c_toronto, orient='index')
df.plot(kind='bar')
# In[]
punctuation = set(string.punctuation)
review_text_positive_lasvegas = []
review_text_negative_lasvegas = []
for d in review_data_lasvegas:
    r = ''.join([c for c in d['text'].lower() if not c in punctuation])
    if d['stars'] == 5:
        review_text_positive_lasvegas.append(r)
    if d['stars'] == 1:
        review_text_negative_lasvegas.append(r)

# In[]
cv = CountVectorizer(max_features=50,stop_words='english')
cv.fit(review_text_positive_lasvegas)

# Generate a word cloud image
wordcloud = WordCloud().fit_words([(c,cv.vocabulary_[c]) for c in cv.vocabulary_])
plt.imshow(wordcloud)
plt.axis("off")

# In[]
cv = CountVectorizer(max_features=50,stop_words='english')
cv.fit(review_text_negative_lasvegas)

# Generate a word cloud image
wordcloud = WordCloud().fit_words([(c,cv.vocabulary_[c]) for c in cv.vocabulary_])
plt.imshow(wordcloud)
plt.axis("off")


# In[]
review_text_positive_toronto = []
review_text_negative_toronto = []
for d in review_data_toronto:
    r = ''.join([c for c in d['text'].lower() if not c in punctuation])
    if d['stars'] == 5:
        review_text_positive_toronto.append(r)
    if d['stars'] == 1:
        review_text_negative_toronto.append(r)

# In[]
cv = CountVectorizer(max_features=50,stop_words='english')
cv.fit(review_text_positive_toronto)

# Generate a word cloud image
wordcloud = WordCloud().fit_words([(c,cv.vocabulary_[c]) for c in cv.vocabulary_])
plt.imshow(wordcloud)
plt.axis("off")

# In[]
cv = CountVectorizer(max_features=50,stop_words='english')
cv.fit(review_text_negative_toronto)

# Generate a word cloud image
wordcloud = WordCloud().fit_words([(c,cv.vocabulary_[c]) for c in cv.vocabulary_])
plt.imshow(wordcloud)
plt.axis("off")


