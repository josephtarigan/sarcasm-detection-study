import feedparser
import json

# get the feed
feed = feedparser.parse('http://rss.detik.com/index.php/detikcom_nasional')

# loop the feed
# get the feed title and content
# save to DB
print (feed['entries'])