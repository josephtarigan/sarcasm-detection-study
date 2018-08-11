import tweepy
import json
import pymysql
import time
import sys
from pprint import pprint
from inspect import getmembers

consumer_key = 'WmZARauKFFdICxQPaJud7kgT4'
consumer_secret = 'nxWrMhd7Zd60kYjD9BwwzLgXOraHXLHSR0RxUrc1YQGsaGwGAK'
access_token = '22621924-D2eHLrYvo7pxb7IJptXkIGB6i3wM4RnyL2inkw8WP'
access_token_secret = 'kwvuhzCEHwoGQQlnJPkjprEfCY4BL8vqerINpvH8FOmXv'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

result = api.search(lang='id', q='#bego', tweet_mode='extended')
print (str(len(result)) + ' tweet(s) are returned\n\n')

#pprint(getmembers(result[0]))

# Open database connection
db = pymysql.connect(host="localhost", user="root",passwd="",db="corpus_vault", charset='utf8')
# prepare a cursor object using cursor() method
cursor = db.cursor()

for tweet in result:
    print ("========================\n")
    print (tweet._json['full_text'])
    print ("\n========================")
    sql = "INSERT INTO twitter_corpus (keyword, tweet_id, tweet_text, tweet_timestamp_string) VALUES ('sarcasm', '" + str(tweet._json['id']) + "', %s, '" + tweet._json['created_at'] + "')"
    print (sql)
    try:
        # Execute the SQL command
        cursor.execute(sql, (tweet._json['full_text']))
    except:
        # Rollback in case there is any error
        print("Unexpected error:", sys.exc_info())

# Commit your changes in the database
db.commit()
# disconnect from server
db.close()
#print (tweet_data)