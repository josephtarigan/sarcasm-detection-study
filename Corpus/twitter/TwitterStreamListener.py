import tweepy
import json
import pymysql
import re
import sys
import time
from inspect import getmembers

#override tweepy.StreamListener to add logic to on_status
class TwitterStreamListener(tweepy.StreamListener):

    def on_data(self, data):
        # Open database connection
        db = pymysql.connect(host="localhost", user="root",passwd="",db="corpus_vault", charset='utf8')
        # prepare a cursor object using cursor() method
        cursor = db.cursor()

        tweet_data = json.loads(data)
        sql = "INSERT INTO twitter_corpus (keyword, tweet_id, tweet_text, tweet_timestamp_long, tweet_timestamp_string, tweet_timestamp_date, raw_tweet) VALUES ('sarcasm', '" + str(tweet_data['id']) + "', %s, '" + str(tweet_data['timestamp_ms']) + "', '" + tweet_data['created_at'] + "', %s, %s)"
        print (sql)
        try:
            # Execute the SQL command
            cursor.execute(sql, (tweet_data['text'], time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(tweet_data['timestamp_ms'])/1000)), data))
            # Commit your changes in the database
            db.commit()
        except:
            # Rollback in case there is any error
            print("Unexpected error:", sys.exc_info())
            db.rollback()

        # disconnect from server
        db.close()
        #print (tweet_data)

    def on_error(self, status):
        print ("error " + str(status))
        if status == 420:
            #returning False in on_data disconnects the stream
            return False