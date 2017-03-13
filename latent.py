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
import copy
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
user_data = loadDataB(yelp_user_path)

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
valid = reviews[50000:100000]
test = reviews[100000:150000]
# In[]
train = np.load("train.npy")
valid = np.load("hold.npy")
test = np.load("test.npy")
N_train = 50000
# In[]
# In[]
userDict = defaultdict(list)
itemDict = defaultdict(list)
#list  = []

for l in train:
    userID,itemID = l['user_id'],l['business_id']
    userDict[userID].append({'rating':l['stars'], 'itemID':itemID})
    itemDict[itemID].append({'rating':l['stars'], 'userID':userID});
'''
userDict_vl = defaultdict(list)
itemDict_vl = defaultdict(list)
for l in valid:
    userID,itemID = l['user_id'],l['business_id']
    userDict_vl[userID].append({'rating':l['stars'], 'itemID':itemID})
    itemDict_vl[itemID].append({'rating':l['stars'], 'userID':userID});

userDict_tt = defaultdict(list)
itemDict_tt = defaultdict(list)
for l in valid:
    userID,itemID = l['user_id'],l['business_id']
    userDict_tt[userID].append({'rating':l['stars'], 'itemID':itemID})
    itemDict_tt[itemID].append({'rating':l['stars'], 'userID':userID});
'''

# In[]
beta_user = defaultdict(float)
beta_item = defaultdict(float)
lamda = 7.0;
alpha = 0

for itemID, users in itemDict.items():
    beta_item[itemID] = 1.0;
for userID, items in userDict.items():
    beta_user[userID] = 1.0

# In[]
# Naive model
def getAlpha(beta_user, beta_item):
    sum_ = 0;
    for itemID, users in itemDict.items():
        for u in users:
            sum_ += u['rating'] - (beta_user[u['userID']] + beta_item[itemID])
    sum_ = sum_ / N_train
    return sum_;
def getBetaUser(userID, alpha, beta_item, lamda):
    sum_ = 0;
    items = userDict[userID]
    #print(userID)
    #print(items)
    for item in items:
        sum_ += item['rating'] - (alpha + beta_item[item['itemID']])
    sum_ = sum_ / (lamda + len(items))
    return sum_
def getBetaItem(itemID, alpha, beta_user, lamda):
    sum_ = 0;
    users = itemDict[itemID]
    for user in users:
        sum_ += user['rating'] - (alpha + beta_user[user['userID']])
    sum_ = sum_ / (lamda + len(users))
    return sum_
# In[]
iters = 150
lamda = 7.0;
alpha = 0
avg_beta_user = 0
avg_beta_item = 0;
    
for i in range(iters):
    alpha = getAlpha(beta_user, beta_item)
    for uid, beta in beta_user.items():
        beta_user[uid] = getBetaUser(uid, alpha, beta_item, lamda)
    for iid, beta in beta_item.items():
        beta_item[iid] = getBetaItem(iid, alpha, beta_user, lamda)
    getMSE()

for itemID, users in itemDict.items():
    avg_beta_item += beta_item[itemID]

avg_beta_item = avg_beta_item / len(beta_item)

for userID, items in userDict.items():
    avg_beta_user += beta_user[userID]
avg_beta_user = avg_beta_user / len(beta_user)
print("avg_beta_item=",avg_beta_item)
print("avg_beta_user",avg_beta_user)
getMSE()
# In[]
def predict(userID, itemID):
    #global alpha, beta_user, beta_item, avg_beta_item, avg_beta_user
    return alpha + beta_user.get(userID, avg_beta_user) + beta_item.get(itemID, avg_beta_item)
# In[]
def getMSE():
    sum_ = 0;
    for l in valid:
        sum_ += np.square(l['stars'] - predict(l['user_id'], l['business_id']))
    mse_valid = sum_/len(valid)
    print("RMSE Valid=",np.sqrt(mse_valid))
# In[]
alpha_org = copy.deepcopy(alpha)
beta_user_org = copy.deepcopy(beta_user)
beta_item_org = copy.deepcopy(beta_item)
# In[]
'''
def getObjFunction():
    sum_ = 0
    for itemID, users in itemDict.items():
        for u in users:
            userID = u['userID']
            sum_ += np.square(alpha + beta_user[userID] + beta_item[itemID] - u['rating']) 

    sum_ += lamda * (np.sum(np.sqaure(beta_user))
                + np.sum(np.sqaure(beta_item))
                    + np.sum(np.sqaure(gamma_item))
                       + np.sum(np.sqaure(gamma_user)))
'''
def getAlphaDerv():
    sum_ = 0;
    N = 0
    for itemID, users in itemDict.items():
        for u in users:
            N += 1
            userID = u['userID']
            sum_ +=  (alpha+beta_user[userID] + beta_item[itemID] 
                        + np.dot(gamma_user[userID],gamma_item[itemID]) \
                    - u['rating'])
    sum_ = 2 * sum_ / N
    return  sum_;
def getBetaUserDerv(userID):
    sum_ = 0;
    items = userDict[userID]
    #print(userID)
    #print(items)
    for item in items:
        sum_ += ((alpha + beta_item[item['itemID']] + beta_user[userID] 
                    + np.dot(gamma_user[userID],gamma_item[itemID])) \
                 - item['rating'])
    sum_ = 2 * (sum_/len(items)) + 2 * lamda * beta_user[userID] 
    #sum_ = sum_ / (lamda + len(items)) 
    return sum_
def getBetaItemDerv(itemID):
    sum_ = 0;
    users = itemDict[itemID]
    for user in users:
        sum_ += (alpha + beta_user[user['userID']] + beta_item[itemID]
                    + np.dot(gamma_user[userID],gamma_item[itemID]) \
                  - user['rating'])
    sum_ = 2 * (sum_/len(users)) + 2 * lamda * beta_item[itemID] 
    return sum_

def getGammaUserDerv(userID, k):
    sum_ = 0;
    items = userDict[userID]
    for item in items:
        sum_ += gamma_item[item['itemID']][k] * ((alpha 
                    + beta_item[item['itemID']]
                    +  beta_user[userID]
                    + np.dot(gamma_user[userID],gamma_item[itemID])) \
                 - item['rating'])
    sum_ = 2 * (sum_/len(items)) + 2 * lamda * gamma_user[userID][k]
    return sum_

def getGammaItemDerv(itemID, k):
    sum_ = 0;
    users = itemDict[itemID]
    for user in users:
        sum_ += gamma_user[user['userID']][k] * ((alpha 
                    + beta_user[user['userID']] 
                    + beta_item[itemID]
                    + np.dot(gamma_user[userID],gamma_item[itemID])) \
                  - user['rating'])
    sum_ = 2 * (sum_/len(users)) + 2 * lamda * gamma_item[itemID][k] 
    return sum_
    
def getAvgBeta(beta):
    sum_ = 0
    for k,v in beta.items():
        sum_ += v
    sum_ = sum_ / len(beta)
    return sum_
def getAvgGamma(gamma):
    sum_ = [0]*K
    for u,v in gamma.items():
        for i in range(len(sum_)):
            sum_[i] += v[i]
    for i in range(len(sum_)):
        sum_[i] = sum_[i]/len(gamma)
    return sum_
    
# In[]
def evaulate(data_):
    mse = 0;
    for d in data_:
        mse += np.square(d['stars'] - predict(d['user_id'], d['business_id']))
    return np.sqrt(mse/len(data_))
def anealing(i, eta):
    return eta / (1+i/T)
    #return eta
def momentum(var, grad, accu_grad, eta_, mu_):
    new_del = accu_grad * mu_ - eta_ * grad
    var = var + new_del 
    accu_grad = new_del
    return var, accu_grad

    
def updateAlpha(eta):
    global alpha, delta_alpha
    alpha_ = alpha
    alpha += mu * delta_alpha;
    grad = getAlphaDerv()
    delta_alpha = mu * delta_alpha  - eta * grad
    alpha = alpha_ + delta_alpha

def updateBetaUser(userID, eta):
    global beta_user, delta_beta_user
    beta_ = beta_user[userID];
    beta_user[userID] += mu * delta_beta_user[userID]
    grad = getBetaUserDerv(userID)
    delta_beta_user[userID] = mu * delta_beta_user[userID] - eta * grad
    beta_user[userID] = beta_ + delta_beta_user[userID]

def updateBetaItem(itemID, eta):
    global beta_item, delta_beta_item
    beta_ = beta_item[itemID]
    beta_item[itemID] += mu * delta_beta_item[itemID]
    grad = getBetaItemDerv(itemID)
    delta_beta_item[itemID] = mu * delta_beta_item[itemID] - eta * grad
    beta_item[itemID] = beta_ + delta_beta_item[itemID]

def updateGammaUser(userID, k, eta):
    global gamma_user, delta_gamma_user
    gamma_ = gamma_user[userID][k]
    gamma_user[userID][k] += mu * delta_gamma_user[userID][k]
    grad = getGammaUserDerv(userID, k)
    delta_gamma_user[userID][k] = mu * delta_gamma_user[userID][k] - eta * grad
    gamma_user[userID][k] = gamma_ + delta_gamma_user[userID][k]

def updateGammaItem(itemID, k, eta):
    global gamma_item, delta_gamma_item
    gamma_ = gamma_item[itemID][k]
    gamma_item[itemID][k] += mu * delta_gamma_item[itemID][k]
    grad = getGammaItemDerv(itemID, k)
    delta_gamma_item[itemID][k] = mu * delta_gamma_item[itemID][k] - eta * grad
    gamma_item[itemID][k] = gamma_ + delta_gamma_item[itemID][k]

def predict(userID, itemID):
    global alpha, beta_user, beta_item, gamma_user, gamma_item, \
                avg_beta_item, avg_beta_user, avg_gamma_item, avg_gamma_user
    rating = alpha + beta_user.get(userID, avg_beta_user) \
                + beta_item.get(itemID, avg_beta_item) \
                    + np.asscalar(np.dot(gamma_item.get(itemID, avg_gamma_item),
                                         gamma_user.get(userID, avg_gamma_user)))
    '''
    rating =  alpha + beta_user.get(userID, 0) \
                + beta_item.get(itemID, 0) \
                    + np.asscalar(np.dot(gamma_item.get(itemID, [0]*K),
                                         gamma_user.get(userID, [0]*K)))
    '''
    rating = max(1, rating)
    rating = min(5, rating)
    return rating             
 
def gradientDecent(eta_a, eta_b, eta_g):
    global alpha, beta_user, beta_item, gamma_user, gamma_item, \
                avg_beta_item, avg_beta_user, avg_gamma_item, avg_gamma_user,\
                delta_gamma_user,delta_gamma_item,delta_beta_item,delta_beta_user,delta_alpha,\
                alpha_min, beta_user_min,beta_item_min,gamma_user_min, gamma_item_min
            
    min_rmse = 1000
    count = 0
    
    for i in range(iters):
        
        # alpha
        #new_del = delta_alpha * mu - anealing(i,eta_a) * getAlphaDerv()
        #alpha = alpha + new_del 
        #delta_alpha = new_del
        
        #alpha, delta_alpha = momentum(alpha, getAlphaDerv(), delta_alpha, anealing(i,eta_a), mu)
        updateAlpha(anealing(i,eta_a))
        for userID in beta_user:
            updateBetaUser(userID,anealing(i,eta_b))
            #beta_user[userID], delta_beta_user[userID] = momentum(beta_user[userID], 
            #                getBetaUserDerv(userID), delta_beta_user[userID], anealing(i,eta_b), mu)
            
            #new_del = delta_beta_user[userID] * mu - anealing(i,eta_b) * getBetaUserDerv(userID)
            #beta_user[userID] = beta_user[userID] + new_del
            #delta_beta_user[userID] = new_del
        for itemID in beta_item:
            updateBetaItem(itemID, anealing(i,eta_b))
            #beta_item[itemID], delta_beta_item[itemID] = momentum(beta_item[itemID], 
            #                getBetaItemDerv(itemID), delta_beta_item[itemID], anealing(i,eta_b), mu)
            #new_del = delta_beta_item[itemID] * mu - anealing(i,eta_b) * getBetaItemDerv(itemID)
            #beta_item[itemID] = beta_item[itemID] + new_del
            #delta_beta_item[itemID] = new_del
                
        for userID in gamma_user:
            for k in range(K):
                updateGammaUser(userID, k, anealing(i,eta_g))
                #gamma_user[userID][k], delta_gamma_user[userID][k] = momentum(gamma_user[userID][k], 
                #        getGammaUserDerv(userID, k), delta_gamma_user[userID][k], anealing(i,eta_g), mu)
                #new_del = delta_gamma_user[userID][k] * mu - anealing(i,eta_g) * getGammaUserDerv(userID, k)
                #gamma_user[userID][k] = gamma_user[userID][k] + new_del
                #delta_gamma_user[userID][k] = new_del
                    
        for itemID in gamma_item:
            for k in range(K):
                updateGammaItem(itemID, k, anealing(i,eta_g))
                #gamma_item[itemID][k], delta_gamma_item[itemID][k] = momentum(gamma_item[itemID][k], 
                #        getGammaItemDerv(itemID,k), delta_gamma_item[itemID][k], anealing(i,eta_g), mu)
                #new_del = delta_gamma_item[itemID][k] * mu - anealing(i,eta_g) * getGammaItemDerv(itemID,k)
                #gamma_item[itemID][k] = gamma_item[itemID][k] + new_del
                #delta_gamma_item[itemID][k] = new_del
        avg_beta_item = getAvgBeta(beta_item)
        avg_beta_user = getAvgBeta(beta_user)
        avg_gamma_item = getAvgGamma(gamma_item)
        avg_gamma_user = getAvgGamma(gamma_user)
        rmse = evaulate(valid)
        if(rmse < min_rmse):
            min_rmse = rmse
            count = 0
            alpha_min = copy.deepcopy(alpha)
            beta_user_min = copy.deepcopy(beta_user)
            beta_item_min = copy.deepcopy(beta_item)
            gamma_user_min = copy.deepcopy(gamma_user)
            gamma_item_min = copy.deepcopy(gamma_item)
        else:
            count += 1 
        if(count ==4):
            break
        print(i," RMSE=",rmse)
# In[]
'''
beta_user = defaultdict(float)
beta_item = defaultdict(float)
avg_beta_user = 0
avg_beta_item = 0;
alpha = avgRating
'''
# In[]
# In[]
'''
for u in itemDict:
    beta_item[u] = random.uniform(-0.05, 0.05)
    
for u in userDict:
    beta_user[u] = random.uniform(-0.05, 0.05)
'''
# In[]
alpha = copy.deepcopy(alpha_org)
beta_item = copy.deepcopy(beta_item_org)
beta_user = copy.deepcopy(beta_user_org)
K=5
gamma_user = defaultdict(list)
gamma_item = defaultdict(list)

delta_gamma_user = defaultdict(list)
delta_gamma_item = defaultdict(list)
delta_beta_item = defaultdict(list)
delta_beta_user = defaultdict(list)
delta_alpha = 0

for itemID, users in itemDict.items():
    delta_beta_item[itemID] = 0.0;
for userID, items in userDict.items():
    delta_beta_user[userID] = 0.0

for userID in userDict:
    gamma_user[userID] = [0]*K  #(np.random.random((K, 1)) - 0.5) * 0.001
    delta_gamma_user[userID] = [0]*K
for itemID in itemDict:
    gamma_item[itemID] = [0]*K #(np.random.random((K,1)) - 0.5) * 0.001
    delta_gamma_item[itemID] = [0]*K

avg_beta_item = getAvgBeta(beta_item)
avg_beta_user = getAvgBeta(beta_user)
avg_gamma_user = getAvgGamma(gamma_user)
avg_gamma_item = getAvgGamma(gamma_item)

alpha_min = copy.deepcopy(alpha)
beta_user_min = copy.deepcopy(beta_user)
beta_item_min = copy.deepcopy(beta_item)
gamma_user_min = copy.deepcopy(gamma_user)
gamma_item_min = copy.deepcopy(gamma_item)

print("avg_beta_item=",avg_beta_item)
print("avg_beta_user",avg_beta_user)

rmse = evaulate(valid)
print("RMSE=",rmse)    

for k,v in gamma_item.items():
    for i in range(len(v)):
        v[i] = random.uniform(-0.001, 0.001)

for k,v in gamma_user.items():
    for i in range(len(v)):
        v[i] = random.uniform(-0.001, 0.001)

print("Gradient Decent")
lamda = 6.0;
eta_a = 1.0e-15
eta_b = 1.0e-15
eta_g = 1.0e-4
T = 100
mu = 0.8
iters = 150
gradientDecent(eta_a, eta_b, eta_g)
# not present
# user items
# cal avg from those items
# ceil

alpha = copy.deepcopy(alpha_min)
beta_user = copy.deepcopy(beta_user_min)
beta_item = copy.deepcopy(beta_item_min)
gamma_user = copy.deepcopy(gamma_user_min)
gamma_item = copy.deepcopy(gamma_item_min)

# In[]
predictions = open("predictions_Rating.txt", 'w')
for l in open("pairs_Rating.txt"):
  if l.startswith("userID"):
    #header
    predictions.write(l)
    continue
  u,i = l.strip().split('-')
  predictions.write(u + '-' + i +","+ str(predict(u,i))+'\n')
  
predictions.close()
# In[]
'''
# In[]
from time import gmtime, strftime

import time
import sys
import os
from pylab import *
from scipy import sparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
# In[]
# Given a set of ratings, 2 matrix factors that include one or more
# trainable variables, and a regularizer, uses gradient descent to
# learn the best values of the trainable variables.
def mf(ratings_train, ratings_val, W, H, regularizer, mean_rating, max_iter, lr = 0.01, decay_lr = False, log_summaries = False):
    # Extract info from training and validation data
    rating_values_tr, num_ratings_tr, user_indices_tr, item_indices_tr = extract_rating_info(ratings_train)
    rating_values_val, num_ratings_val, user_indices_val, item_indices_val = extract_rating_info(ratings_val)

    # Multiply the factors to get our result as a dense matrix
    result = tf.matmul(W, H)

    # Now we just want the values represented by the pairs of user and item
    # indices for which we had known ratings.
    result_values_tr = tf.gather(tf.reshape(result, [-1]), user_indices_tr * tf.shape(result)[1] + item_indices_tr, name="extract_training_ratings")
    result_values_val = tf.gather(tf.reshape(result, [-1]), user_indices_val * tf.shape(result)[1] + item_indices_val, name="extract_validation_ratings")

    # Calculate the difference between the predicted ratings and the actual
    # ratings. The predicted ratings are the values obtained form the matrix
    # multiplication with the mean rating added on.
    diff_op = tf.subtract(tf.add(result_values_tr, mean_rating, name="add_mean"), rating_values_tr, name="raw_training_error")
    diff_op_val = tf.subtract(tf.add(result_values_val, mean_rating, name="add_mean_val"), rating_values_val, name="raw_validation_error")

    with tf.name_scope("training_cost") as scope:
        base_cost = tf.reduce_sum(tf.square(diff_op, name="squared_difference"), name="sum_squared_error")

        cost = tf.div(tf.add(base_cost, regularizer), num_ratings_tr * 2, name="average_error")

    with tf.name_scope("validation_cost") as scope:
        cost_val = tf.div(tf.reduce_sum(tf.square(diff_op_val, name="squared_difference_val"), name="sum_squared_error_val"), num_ratings_val * 2, name="average_error")

    with tf.name_scope("train") as scope:
        if decay_lr:
            # Use an exponentially decaying learning rate.
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # Passing global_step to minimize() will increment it at each step 
            # so that the learning rate will be decayed at the specified 
            # intervals.
            train_step = optimizer.minimize(cost, global_step=global_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(lr)
            train_step = optimizer.minimize(cost)

    with tf.name_scope("training_rmse") as scope:
      rmse_tr = tf.sqrt(tf.reduce_sum(tf.square(diff_op)) / num_ratings_tr)

    with tf.name_scope("validation_rmse") as scope:
      # Validation set rmse:
      rmse_val = tf.sqrt(tf.reduce_sum(tf.square(diff_op_val)) / num_ratings_val)

    # Create a TensorFlow session and initialize variables.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if log_summaries:
        # Make sure summaries get written to the logs.
        accuracy_val_summary = tf.scalar_summary("accuracy_val", accuracy_val)
        accuracy_tr_summary = tf.scalar_summary("accuracy_tr", accuracy_tr)
        summary_op = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/recommender_logs", sess.graph_def)
    # Keep track of cost difference.
    last_cost = 0
    diff = 1
    # Run the graph and see how we're doing on every 1000th iteration.
    for i in range(max_iter):
        
        if i > 0 and i % 10 == 0:
            if diff < 0.000001:
                print("Converged at iteration %s" % (i))
                break;
            if log_summaries:
                res = sess.run([rmse_tr, rmse_val, cost, summary_op])
                summary_str = res[3]
                writer.add_summary(summary_str, i)
            else:
                res = sess.run([rmse_tr, rmse_val, cost])
            acc_tr = res[0]
            acc_val = res[1]
            cost_ev = res[2]
            print("Training RMSE at step %s: %s" % (i, acc_tr))
            print("Validation RMSE at step %s: %s" % (i, acc_val))
            diff = abs(cost_ev - last_cost)
            last_cost = cost_ev
        else:
            sess.run(train_step)

    finalTrain = rmse_tr.eval(session=sess)
    finalVal = rmse_val.eval(session=sess)
    finalW = W.eval(session=sess)
    finalH = H.eval(session=sess)
    sess.close()
    return finalTrain, finalVal, finalW, finalH

# Extracts user indices, item indices, rating values and number
# of ratings from the ratings triplets.
def extract_rating_info(ratings):
    rating_values = np.array(ratings[:,2], dtype=float32)
    user_indices = ratings[:,0]
    item_indices = ratings[:,1]
    num_ratings = len(item_indices)
    return rating_values, num_ratings, user_indices, item_indices

# Creates a trainable tensor representing either user or item bias,
# and a corresponding tensor of 1's for the other.
def create_factors_for_bias(num_users, num_items, lda, user_bias = True):
    if user_bias:
        # Random normal intialized column for users
        W = tf.Variable(tf.truncated_normal([num_users, 1], stddev=0.02, mean=0), name="users")
        # Row of 1's for items
        H = tf.ones((1, num_items), name="items")
        # Add regularization.
        regularizer = tf.multiply(tf.reduce_sum(tf.square(W)), lda, name="regularize")
    else:
        # Column of 1's for users
        W = tf.ones((num_users, 1), name="users")
        # Random normal intialized row for items
        H = tf.Variable(tf.truncated_normal([1, num_items], stddev=0.02, mean=0), name="items")
        # Add regularization.
        regularizer = tf.mul(tf.reduce_sum(tf.square(H)), lda, name="regularize")
    return W, H, regularizer


# Runs the factorizer for the given number of iterations and with the given
# regularization parameter to learn item bias on top of provided user bias.
def learn_item_bias_from_fixed_user_bias(ratings_tr, ratings_val, user_bias, num_items, lda, global_mean, max_iter):
    W = tf.concat(1, [tf.convert_to_tensor(user_bias, dtype=float32, name="user_bias"), tf.ones((user_bias.shape[0],1), dtype=float32, name="item_bias_ones")])
    H = tf.Variable(tf.truncated_normal([1, num_items], stddev=0.02, mean=0), name="items")
    H_with_user_bias = tf.concat(0, [tf.ones((1, num_items), name="user_bias_ones", dtype=float32), H])
    regularizer = tf.mul(tf.reduce_sum(tf.square(H)), lda, name="regularize")
    return mf(ratings_tr, ratings_val, W, H_with_user_bias, regularizer, global_mean, max_iter, 0.8)

# Learns factors of the given rank with specified regularization parameter.
def create_factors_without_biases(num_users, num_items, rank, lda):
    # Initialize the matrix factors from random normals with mean 0. W will
    # represent users and H will represent items.
    W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.02, mean=0), name="users")
    H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.02, mean=0), name="items")
    regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
    return W, H, regularizer

# Given previously learned user bias and item bias vectors, creates
# tensors to learn factors of the given rank (excluding the bias vectors)
# and a regularizer.
def create_factors_with_biases(user_bias, item_bias, rank, lda):
    num_users = user_bias.shape[0]
    num_items = item_bias.shape[1]
    # Initialize the matrix factors from random normals with mean 0. W will
    # represent users and H will represent items.
    W = tf.Variable(tf.truncated_normal([num_users, rank], stddev=0.02, mean=0), name="users")
    H = tf.Variable(tf.truncated_normal([rank, num_items], stddev=0.02, mean=0), name="items")

    # To the user matrix we add a bias column holding the bias of each user,
    # and another column of 1s to multiply the item bias by.
    W_plus_bias = tf.concat(1, [W, tf.convert_to_tensor(user_bias, dtype=float32, name="user_bias"), tf.ones((num_users,1), dtype=float32, name="item_bias_ones")])
    # To the item matrix we add a row of 1s to multiply the user bias by, and
    # a bias row holding the bias of each item.
    H_plus_bias = tf.concat(0, [H, tf.ones((1, num_items), name="user_bias_ones", dtype=float32), tf.convert_to_tensor(item_bias, dtype=float32, name="item_bias")])
    regularizer = tf.mul(tf.add(tf.reduce_sum(tf.square(W)), tf.reduce_sum(tf.square(H))), lda, name="regularize")
    return W_plus_bias, H_plus_bias, regularizer


# Uses k-fold cross-validation to learn the best regularization
# parameter to use for either user or item bias.
def learn_bias_lda(ratings, num_folds, ldas, num_users, num_items, global_mean, max_iter, user_bias = True):
    labels = ratings[:,2]
    skf = StratifiedKFold(labels, num_folds)
    min_lda = None
    min_rmse = 0
    for lda in ldas:
        sum_rmses = 0
        W, H, reg = create_factors_for_bias(num_users, num_items, lda, user_bias)
        for train, test in skf:
            tr, val, finalw, finalh = mf(ratings[train,:], ratings[test,:], W, H, reg, global_mean, max_iter, 0.8)
            sum_rmses += val
            print("Training rmse: %s, val rmse: %s, lda: %s" % (tr, val, lda))
        avg_rmse = sum_rmses / num_folds
        if min_lda == None:
            # This is our first lambda.
            min_lda = lda
            min_rmse = avg_rmse
        elif avg_rmse < min_rmse:
            # We did better than the last lambda.
            min_rmse = avg_rmse
            min_lda = lda
        else:
            # It's not going to get any better with the next lambda.
            break
    return min_lda

# Runs the factorizer for the given number of iterations and with the given
# regularization parameter to learn user bias from the training set.
def get_user_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter):
    W, H, reg = create_factors_for_bias(num_users, num_items, lda, True)
    tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 0.8)
    return finalw

# Runs the factorizer for the given number of iterations and with the given
# regularization parameter to learn item bias from the training set.
def get_item_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter):
    W, H, reg = create_factors_for_bias(num_users, num_items, lda, False)
    tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 0.8)
    return finalh
# In[]
count = 0

userIdMap = defaultdict(int);
for u in userDict_tr.keys():
    if u not in userIdMap:
        userIdMap[u] = count
        count += 1
for u in userDict_vl.keys():
    if u not in userIdMap:
        userIdMap[u] = count
        count += 1
for u in userDict_tt.keys():
    if u not in userIdMap:
        userIdMap[u] = count
        count += 1


count = 0

itemIdMap = defaultdict(int);
for u in itemDict_tr.keys():
    if u not in itemIdMap:
        itemIdMap[u] = count
        count += 1
for u in itemDict_vl.keys():
    if u not in itemIdMap:
        itemIdMap[u] = count
        count += 1
for u in itemDict_tt.keys():
    if u not in itemIdMap:
        itemIdMap[u] = count
        count += 1
del count
# In[]
ratings_tr = []
for userID, items in userDict_tr.items():
    for item in items:
        ratings_tr.append([userIdMap[userID], itemIdMap[item['itemID']], item['rating']])
ratings_tr = np.array(ratings_tr)


ratings_val = []
for userID, items in userDict_vl.items():
    for item in items:
        ratings_val.append([userIdMap[userID], itemIdMap[item['itemID']], item['rating']])
ratings_val = np.array(ratings_val)

ratings_tt = []
for userID, items in userDict_tt.items():
    for item in items:
        ratings_tt.append([userIdMap[userID], itemIdMap[item['itemID']], item['rating']])
ratings_tt = np.array(ratings_tt)

# In[]

global_mean = mean(ratings_tr[:,2])
np.random.seed(12)
num_users = np.unique(ratings_tr[:,0]).shape[0]
num_items = np.unique(ratings_tr[:,1]).shape[0]
    
# In[]
max_iter = 100
lda = 1.0
to_learn_arr = ["user_bias","item_bias","features"]
for l in to_learn_arr:
    train_mf(l)
    print(l," Done")
# In[]
def train_mf(to_learn):
    global lda
    if to_learn == "user_bias_lda":
        lda = learn_bias_lda(ratings_tr, 4, [2,4,6,8,10], num_users, num_items, global_mean, max_iter)
        print("Best lambda for user bias is %s" %(lda))
    elif to_learn == "item_bias_lda":
        lda = learn_bias_lda(ratings_tr, 4, [2,4,6,8,10], num_users, num_items, global_mean, max_iter, False)
        print("Best lambda for item bias is %s" %(lda))
    elif to_learn == "user_bias":
        user_bias = get_user_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter)
        np.save("user_bias", user_bias)
    elif to_learn == "item_bias":
        item_bias = get_item_bias(ratings_tr, ratings_val, lda, num_users, num_items, global_mean, max_iter)
        np.save("item_bias", item_bias)
    elif to_learn == "item_bias_fixed_user":
        user_bias = np.load("user_bias.npy")
        tr, val, finalw, finalh = learn_item_bias_from_fixed_user_bias(ratings_tr, ratings_val, np.load("user_bias.npy"), num_items, lda, global_mean, max_iter)
        print("Final training RMSE %s" % (tr))
        print("Final validation RMSE %s" % (val))
        np.save("item_bias_fixed_user", finalh[1,:].reshape(num_items,))
    elif to_learn == "features":
        rank = int(sys.argv[5])
        user_bias = np.load("user_bias.npy").reshape(num_users, 1)
        item_bias = np.load("item_bias.npy").reshape(1, num_items)
        W, H, reg = create_factors_with_biases(user_bias, item_bias, rank, lda)
        tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 1.0, True)
        print("Final training RMSE %s" % (tr))
        print("Final validation RMSE %s" % (val))
        np.save("final_w", finalw)
        np.save("final_h", finalh)
    elif to_learn == "features-only":
        rank = int(sys.argv[5])
        W, H, reg = create_factors_without_biases(num_users, num_items, rank, lda)
        tr, val, finalw, finalh = mf(ratings_tr, ratings_val, W, H, reg, global_mean, max_iter, 1.0, True)
        print("Final training RMSE %s" % (tr))
        print("Final validation RMSE %s" % (val))
        np.save("final_w", finalw)
        np.save("final_h", finalh)
'''