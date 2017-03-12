# In[]
import numpy as np
from collections import defaultdict
import operator
from collections import OrderedDict
from operator import itemgetter

# In[]
yelp_business_path = 'yelp_academic_dataset_business.json'
yelp_review_path = 'yelp_academic_dataset_review.json'
yelp_user_path = 'yelp_academic_dataset_user.json'
# In[]
# read data
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
def loadData(f):
    return list(parseData(f))
            
# In[]
review_data = loadData(yelp_review_path)
business_data = loadData(yelp_business_path)

# In[]
city_wise_review_counts = defaultdict(int)
city_wise_business_counts = defaultdict(int)
city_wise_avg_rating = defaultdict(int)

for d in business_data:
    city = d['city']
    city_wise_business_counts[city] += 1
    city_wise_avg_rating[city] += d['stars']
    city_wise_review_counts[city] += d['review_count']

del d, city
# In[]
for city, rating in city_wise_avg_rating.items():
    city_wise_avg_rating[city] /= city_wise_business_counts[city]

# In[]
# sort
# city_wise_avg_rating = OrderedDict(sorted(city_wise_avg_rating.items(), key=itemgetter(1)), reverse=True)
city_wise_business_counts = OrderedDict(sorted(city_wise_business_counts.items(), key=itemgetter(1)))
city_wise_review_counts = OrderedDict(sorted(city_wise_review_counts.items(), key=itemgetter(1)))

# In[]
# top cities


