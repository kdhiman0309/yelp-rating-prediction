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
def parseData(file, read_limit):
    null = None
    with open(file, errors='ignore') as f:
        i=0
        for l in f:
            if i<read_limit:
                i+=1;
                yield eval(l)
            else :
                break
def loadData(f, read_limit=1000000):
    return  list(parseData(f, read_limit))
            
# In[]
business_data = loadData(yelp_business_path)
review_data = loadData(yelp_review_path,10000000)
# In[]
business_data_id = defaultdict(int)
i = 0
for b in business_data:
    business_data_id[b['business_id']] = i
    i += 1
# In[]
def getYear(s):
    return int(s.split("-")[0])
# In[]
min_year = 2010
# In[]
city_wise_review_counts = defaultdict(int)
city_wise_avg_rating = defaultdict(int)

for d in review_data:
    b = business_data[business_data_id[d['business_id']]]
    year = getYear(d['date'])
    
    if year >= min_year:
        city = b['city']
        city_wise_avg_rating[city] += d['stars']
        city_wise_review_counts[city] += 1

del d, b, city, year
for city, rating in city_wise_avg_rating.items():
    city_wise_avg_rating[city] /= city_wise_review_counts[city]

# In[]
# sort
# city_wise_avg_rating = OrderedDict(sorted(city_wise_avg_rating.items(), key=itemgetter(1)), reverse=True)
#city_wise_review_counts_ = OrderedDict(sorted(city_wise_review_counts.items(), key=itemgetter(1), reverse=True))
city_wise_review_counts_ = [(city_wise_review_counts[c], c) for c in city_wise_review_counts]
city_wise_review_counts_.sort()
city_wise_review_counts_.reverse()

# In[]
# top cities
top_city_wise_review_counts = {}

for l in city_wise_review_counts_[:5]:
    top_city_wise_review_counts[l[1]] = l[0]
print(top_city_wise_review_counts)