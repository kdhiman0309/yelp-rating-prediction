# In[]
import numpy as np
from collections import defaultdict
import operator
from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt;
import calendar
# In[]
yelp_business_path = 'yelp_academic_dataset_business.json'
yelp_review_path = 'yelp_academic_dataset_review.json'
yelp_user_path = 'yelp_academic_dataset_user.json' 
# In[]
# read data
def parseData(file):
    null = None
    with open(file, errors='ignore') as f:
        for l in f:
            yield eval(l)
def loadData(f):
    return  list(parseData(f))
            
# In[]
business_data = loadData(yelp_business_path)
# In[]
business_data_id = defaultdict(int)
i = 0
for b in business_data:
    business_data_id[b['business_id']] = i
    i += 1
# In[]
# read resturant data 
#only among top 10 cities
#top10cites = ['Pittsburgh', 'Las Vegas', 'Phoenix',
#'Scottsdale', 'Mesa', 'Charlotte', 'Tempe', 'Henderson',
#'Toronto', 'Montreal'
#]
top_cites = set(['Pittsburgh','Las Vegas','Phoenix','Charlotte','Toronto'])

def parseData(file):
    null = None
    with open(file, errors='ignore') as f:
        for l in f:
            r = eval(l)
            
            b = business_data[business_data_id[r['business_id']]]
            if(b['categories']!=None and b['city'] in top_cites and 'Restaurants' in b['categories']):
                yield r
def loadData(f):
    return  list(parseData(f))
review_data = loadData(yelp_review_path)
# In[]
def getYear(s):
    return int(s.split("-")[0])
def getMonth(s):
    return int(s.split("-")[1])
# In[]
# Temporal analysis
rating_time_ = defaultdict(list)
rating_month_ = defaultdict(list)
num_reviews_month = defaultdict(int)
num_reviews_time = defaultdict(int)
rating_dist = defaultdict(int)
for r in review_data:
    year = getYear(r['date'])
    rating_time_[year].append(r['stars'])
    if  year>=2010 and year <=2016:
        month = getMonth(r['date'])
        rating_month_[(year-2010)*12+month].append(r['stars'])
        num_reviews_time[year] += 1
        rating_dist[r['stars']] += 1
        num_reviews_month[(year-2010)*12+month] += 1
rating_time = defaultdict(float)
for year, ratings in rating_time_.items():
    rating_time[year] = np.average(ratings)

rating_month = defaultdict(float)
for month, ratings in rating_month_.items():
    rating_month[month] = np.average(ratings)
del rating_time_, rating_month_
# In[]
keys = []
values = []
for k,v in rating_time.items():
    keys.append(k)
    values.append(v)
keys, values = (list(x) for x in zip(*sorted(zip(keys, values), key=lambda pair: pair[0])))
plt.plot(keys,values)
plt.xlabel("year")
plt.ylabel("average rating")
plt.savefig("avg_rating_year.png")
# In[]
keys = []
values = []
months = []
for k,v in rating_month.items():
    keys.append(k)
    values.append(v)
keys, values = (list(x) for x in zip(*sorted(zip(keys, values), key=lambda pair: pair[0])))
#for k in keys:
    #months.append(calendar.month_abbr[k])
plt.plot(keys,values,color='black')
#plt.xticks(keys, months)
for i in range(12, 12*8, 12):
    plt.axvline(i, color='black')
plt.ylabel("average rating")
plt.savefig("avg_rating_month.png")
# In[]
keys = []
values = []
for k,v in num_reviews_time.items():
    keys.append(k)
    values.append(v/1000)
plt.bar(keys, values, color='black', align='center')
plt.xlabel("year")
plt.ylabel("# reviews (in K)")
plt.savefig("num_reviews_year.png")
# In[]
keys = []
values = []
for k,v in num_reviews_month.items():
    keys.append(k)
    values.append(v/1000)
#plt.bar(keys, values, color='black', align='center')
plt.plot(values,'.')
for i in range(12, 12*8, 12):
    plt.axvline(i, color='black')
plt.xlabel("month")
plt.ylabel("# reviews (in K)")
plt.savefig("num_reviews_month.png")
# In[]
keys = []
values = []
for k,v in rating_dist.items():
    keys.append(k)
    values.append(v/1000)

plt.bar(keys, values, color='black', align='center')
plt.xlabel("star rating")
plt.ylabel("# reviews (in K)")
plt.savefig("num_reviews_rating.png")

# In[]
min_year = 0
# In[]
city_wise_review_counts = defaultdict(int)
city_wise_avg_rating = defaultdict(float)

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
keys = []
values = []
for k,v in city_wise_review_counts.items():
    keys.append(k)
    values.append(v/1000)
k = list(range(0,len(keys),1))
plt.bar(k, values, color='black', align='center')
plt.xticks(k,keys)
plt.ylabel("# reviews (in K)")
plt.savefig("num_reviews_city.png")
# In[]
keys = []
values = []
for k,v in city_wise_avg_rating.items():
    keys.append(k)
    values.append(v)
k = list(range(0,len(keys),1))
plt.plot(k, values, 'ro', color='black')
plt.xticks(k,keys)
plt.margins(0.05, 0.1)
plt.ylabel("avg rating")
plt.savefig("rating_city.png")


# In[]
# top cities
top_city_wise_review_counts = {}

for l in city_wise_review_counts_[:10]:
    top_city_wise_review_counts[l[1]] = l[0]
print(top_city_wise_review_counts)
# In[]