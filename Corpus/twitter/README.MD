# Specify the DDL

#CREATE TABLE twitter_corpus (
#  id                      int   primary key AUTO_INCREMENT,
#  keyword                 text,
#  tweet_id                bigint  unique,
#  tweet_text              text,
#  tweet_timestamp_long    long,
#  tweet_timestamp_string  text,
#  tweet_timestamp_date    datetime,
#  raw_tweet               text,
#  is_sarcasm              int  default -1
#);

#ALTER TABLE twitter_corpus CONVERT TO CHARACTER SET UTF8 COLLATE utf8_general_ci;