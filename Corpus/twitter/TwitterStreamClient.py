from TwitterStreamListener import TwitterStreamListener
import tweepy

consumer_key = 'WmZARauKFFdICxQPaJud7kgT4'
consumer_secret = 'nxWrMhd7Zd60kYjD9BwwzLgXOraHXLHSR0RxUrc1YQGsaGwGAK'
access_token = '22621924-D2eHLrYvo7pxb7IJptXkIGB6i3wM4RnyL2inkw8WP'
access_token_secret = 'kwvuhzCEHwoGQQlnJPkjprEfCY4BL8vqerINpvH8FOmXv'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

twitterStreamListener = TwitterStreamListener()
tweetStream = tweepy.Stream(auth=auth, listener=twitterStreamListener, tweet_mode='extended')

#tweetStream.filter(track=['sarkasme'])
#tweetStream.filter(track=['sarkastik'])
#tweetStream.filter(track=['#eh'])
tweetStream.filter(track=['#lah'])