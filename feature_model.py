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
import scipy
import copy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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
def parseData(file):
    null = None
    with open(file, errors='ignore') as f:
        for l in f:
            yield eval(l)
def loadData(f):
    parseData(f)
# In[]
reviews = list(parseData(yelp_review_path))
# In[]
for r in reviews:
    b = business_data[business_data_id[r['business_id']]]
    if b['categories']!=None and b['city'] in top_cities and 'Restaurants' in b['categories'] and (min_year<=int(r['date'].split('-')[0])<=max_year):
       top_cities_review[b['city']].append(r)

# In[]
np.random.shuffle(reviews)
np.save('reviews',reviews)
# In[]
reviews, train, valid, test = [], [], [], []
for x in top_cities_review.keys():
    reviews += top_cities_review[x]
# In[]
b = business_data[business_data_id[r['business_id']]]

if b['categories']!=None and b['city'] in top_cities and 'Restaurants' in b['categories'] and (min_year<=int(r['date'].split('-')[0])<=max_year):
   top_cities_review[b['city']].append(r)
# In[]
reviews = []
for u, i in top_cities_review.items():
    reviews.append(i)
# In[]
train = reviews[:50000]
valid = reviews[50000:100000]
test = reviews[100000:150000]
# In[]
train = np.load("train.npy")
valid = np.load("hold.npy")
# In[]
train_ = []
for x in train:
    if business_data[business_data_id[x['business_id']]]['city']=='Las Vegas':
        train_.append(x)
train = copy.deepcopy(train_)
valid_ = []
for x in valid:
    if business_data[business_data_id[x['business_id']]]['city']=='Las Vegas':
        valid_.append(x)
valid = copy.deepcopy(valid_)
del train_, valid_
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

def parseData(file):
    null = None
    with open(file, errors='ignore') as f:
        for l in f:
            yield eval(l)
def loadData(f):
    parseData(f)


# In[]
lis = []
for u in users:
    lis.append({"average_stars":u['average_stars'], 'cool':u['cool'], 'fans':u['fans'], 'elite':u['elite'], 'funny':u['funny'], 'num_friends':len(u['friends']), 'review_count':u['review_count'],'useful':u['useful'], 'user_id':u['user_id'], 'yelping_since':u['yelping_since']})

np.save("users_all",lis)
# In[]
users = np.load("users_all.npy")
# In[]
users_dict = defaultdict()
for u in users:
    users_dict[u['user_id']] = u
# In[]
users_avg = defaultdict(float)
for uid in train_users:
    u = users_dict[uid]
    users_avg['average_stars'] += u['average_stars'] 
    users_avg['review_count'] += u['review_count']
    users_avg['useful'] += u['useful']
    users_avg['funny'] += u['funny']
    users_avg['cool'] += u['cool']
    users_avg['elite'] += len(u['elite'])
t = len(train_users)
for k, v in users_avg.items():
    users_avg[k] = v / t

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
    f.append(rew['nPunctuations'])
    f.append(1 if getMonth(r['date'])==12 else 0)
    #b = business_data[business_data_id[r['business_id']]]

    #f += getCityOneHot(b['city'])
    f.append(b['stars'])
    f.append(b['review_count'])
    userid = r['user_id']
    user = users_dict[userid]
    if userid in train_users:
        f.append(user['average_stars'])
        f.append(user['review_count'])
    else:
        f.append(users_avg['average_stars'])
        f.append(users_avg['review_count']) 
    
    #f.append(getSenti(r))
    return f

def label(r):
    return r['stars']
#In[]
X_train = []
y_train = []
for d in train[:20000]:
    b = business_data[business_data_id[d['business_id']]]
    if(b['review_count']>6 and users_dict[d['user_id']]['review_count']>5):
        X_train.append(feature(d, b))
        y_train.append([label(d)])
X_valid = []
y_valid = []

#In[]
for d in valid:
    b = business_data[business_data_id[d['business_id']]]
    X_valid.append(feature(d,b))
    y_valid.append([label(d)])
#In[]
X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

# In[]
clf_l = Ridge(alpha=.01, fit_intercept = False, solver='auto')
clf_l.fit(X_train,y_train)
predict = clf_l.predict(X_valid)
theta = clf_l.coef_
print(theta)
rmse = np.sqrt(np.average(np.square(y_valid - predict)))
print("RMSE = ", rmse)
# In[]

# In[]
model = RandomForestRegressor(5, max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
predictions = predictions.reshape(len(predictions), 1)
mse = np.average(np.square(y_valid - predictions))
print(np.sqrt(mse))

# In[]
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
predictions = predictions.reshape(len(predictions), 1)
mse = np.average(np.square(y_valid - predictions))
print(np.sqrt(mse))

# In[]
def f_mse(theta, X, y, lam):
    global iters, min_theta, min_rmse_v
    theta = theta.reshape((len(theta),1))
    error = np.sum(np.square(y - np.dot(X,theta))); #MSE
    error =  error/len(y) + lam * np.sum(np.square(theta)) #L2
    iters +=1
    if lam!=0 and iters%10==0:
        rmse_v = np.sqrt(f_mse(theta, X_valid, y_valid, 0))
        if(rmse_v < min_rmse_v):
            min_rmse_v  = rmse_v
            min_theta = copy.deepcopy(theta)
        print(error, " Train:",np.sqrt(f_mse(theta, X, y, 0)),' Valid:', rmse_v)
    return error
# NEGATIVE Derivative of log-likelihood
def fprime_mse(theta, X, y, lam):
    theta = theta.reshape((len(theta),1))
    dl = 2 * np.dot((np.dot(X, theta) - y).T, X).T #MSE
    #print(len(y))
    dl = dl/len(y)
    dl += 2* lam * (theta); #L2
    return dl
# In[]
# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, x_v, y_v, theta, lam, alpha,mu, m, numIterations):
    xTrans = x.transpose()
    grad_acc = 0
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (m)
        print("Iteration %d | Cost: %f" % (i, cost))
        print("RMSE valid = ", np.sqrt(np.average(np.square(np.dot(x_v, theta) - y_v))))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        g = (gradient+2*theta*lam)
        del_theta = mu * grad_acc - alpha * g
        theta = theta + del_theta
        grad_acc = del_theta
    return theta

# In[Gradient Descent]
theta_min = []
prev_mse = np.inf
theta = np.random.random((len(X_train[0]),1))
theta[0] = 3.7

def gd():
    global prev_mse, theta, theta_min
    eta = 0.000001
    lam = 0.0
    print(theta.shape)
    for i in range(10):
        #print(theta.shape, fprime_mse(theta, X_train, y_train, lam))
        theta = theta - eta*fprime_mse(theta, X_train, y_train, lam)
        mse_on_valid = np.sqrt(f_mse(theta, X_valid, y_valid, 0))
        print(mse_on_valid)
        if mse_on_valid < prev_mse:
            prev_mse = mse_on_valid
            theta_min = copy.deepcopy(theta)
        
gd()
# In[]
theta = np.zeros((len(X_train[0]),1))
theta[0] = 3.7
gradientDescent(X_train, y_train, X_valid, y_valid, theta, 0.1, 0.000001, 0.999,len(X_train), 100000)
# In[]
iters = 0
min_theta = 0
min_rmse_v = np.inf
scipy.optimize.fmin_l_bfgs_b(f_mse, [0]*len(X_train[0]), fprime_mse, factr=1e7,args = (X_train, y_train, 0.01))

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
optimizer = tf.train.AdamOptimizer(0.05)
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
