import tweepy
import jsonpickle
import datetime

# init
consumer_key = 'WmZARauKFFdICxQPaJud7kgT4'
consumer_secret = 'nxWrMhd7Zd60kYjD9BwwzLgXOraHXLHSR0RxUrc1YQGsaGwGAK'
access_token = '22621924-D2eHLrYvo7pxb7IJptXkIGB6i3wM4RnyL2inkw8WP'
access_token_secret = 'kwvuhzCEHwoGQQlnJPkjprEfCY4BL8vqerINpvH8FOmXv'
search_query = ''

# authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# specify the search query
search_query = 'politik'
max_tweets = 10000000
tweets_per_query = 100
fName = 'tweets-' + str(datetime.date.today()) + '.txt'

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
since_id = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1

# do the search
tweet_count = 0
print("Downloading max {0} tweets".format(max_tweets))
with open(fName, 'w') as f :
    while tweet_count < max_tweets:
        try:
            if (max_id <= 0):
                if (not since_id):
                    new_tweets = api.search(q=search_query, count=tweets_per_query)
                else:
                    new_tweets = api.search(q=search_query, count=tweets_per_query, since_id=since_id)
    
            else:
                if (not since_id):
                    new_tweets = api.search(q=search_query, count=tweets_per_query, max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=search_query, count=tweets_per_query, max_id=str(max_id - 1), since_id=since_id)
            
            if not new_tweets:
                print("No more tweets found")
                break
            
            for raw_tweet in new_tweets :
                json_pickle = jsonpickle.encode(raw_tweet._json, unpicklable=False)
                #print (json_pickle)
                f.write(json_pickle + '\n')
            
            '''
            print (type(new_tweets[0]))
            status_json = new_tweets[0]._json
            #print (status_json)
            tweet_count += len(new_tweets)
            print (tweet_count)
            
            tweets = json.loads(status_json)
            for tweet in tweets :
                print (tweet)
            '''
            print (tweet_count)
            tweet_count = tweet_count + len(new_tweets)
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break
    
    f.write('==============\nLast ID : ' + str(max_id))