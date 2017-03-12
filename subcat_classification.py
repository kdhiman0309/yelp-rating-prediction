
# coding: utf-8

# In[20]:

import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import NMF, LatentDirichletAllocation


# In[2]:

def parseData(file, read_limit = 100000):
    null = None
    with open(file, errors='ignore') as f:
        i=0
        for l in f:
            if i<read_limit:
                i+=1;
                yield eval(l)
            else :
                break
            
def LoadData(f):
    return list(parseData(f))

review_data = LoadData('yelp_academic_dataset_review.json')
tip_data = LoadData('yelp_academic_dataset_tip.json')
business_data = LoadData('yelp_academic_dataset_business.json')


# In[3]:

cat_dict = defaultdict(int)
biz_cat_map = defaultdict(list)
for data in business_data:
    if data['categories'] != None:
        biz_cat_map[data['business_id']] = data['categories']
        for cat  in data['categories']:
            cat_dict[cat]+=1


# In[4]:

#years = [d['date'].split('-')[0] for d in data]
#data[0]['date'].split('-')[0]
sort_cat =  sorted(cat_dict.items(), key=lambda x:x[1])[-50:]
sort_cat.reverse()
sorted_cat = [d[0] for d in sort_cat]


# In[5]:

review_data[0]


# In[6]:

business_data[78]


# In[7]:

Y_cat=[]
for data in review_data:
    biz_id = data['business_id']
    biz_cat = biz_cat_map[biz_id]
    done = False
    if len(biz_cat) > 0: 
        for i in range(len(sort_cat)):
            if sorted_cat[i] in biz_cat:
                Y_cat.append(i)
                done = True
                break
        if done == False:
            Y_cat.append(len(sort_cat))
            


# In[8]:

Counter(Y_cat)


# In[21]:

n_top_words = 20
n_feat = 1000
n_topics = 16
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


corpus = [d['text'] for d in review_data if len(biz_cat_map[d['business_id']])> 0]
tf_vectorizer = CountVectorizer(max_features=1000, stop_words='english', max_df=0.95, min_df=2)
tf = tf_vectorizer.fit_transform(corpus)


tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', max_df=0.95, min_df=2)
tfidf = tfidf_vectorizer.fit_transform(corpus)

nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0).fit(tf)
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

# In[]
def addUnigramTfIdf(s):
    e = []
    tf = vectorizer.transform([s]).toarray()
    tfidf = tf * idf
    tfidf = normalize(tfidf)
    for i in tfidf[0]:
         e.append(i)   
    return e


# In[14]:

X = [addUnigramTfIdf(d['text']) for d in review_data if len(biz_cat_map[d['business_id']])>0]


# In[15]:

clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,multi_class='multinomial').fit(X, Y_cat)


# In[16]:

print("training score : %.3f (%s)" % (clf.score(X, Y_cat), 'multinomial'))


# In[17]:

clf.n_iter_ 


# In[18]:

len(X), len(Y_cat)


# In[ ]:




# In[ ]:



