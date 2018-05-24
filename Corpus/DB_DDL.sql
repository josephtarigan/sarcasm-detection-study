use corpus_vault;

#DROP TABLE twitter_corpus;

#CREATE TABLE twitter_corpus (
#  id                      int   primary key AUTO_INCREMENT,
#  keyword                 text,
#  tweet_id                bigint  unique,
#  tweet_text              text,
#  tweet_timestamp_long    long,
#  tweet_timestamp_string  text,
#  tweet_timestamp_date    datetime,
#  raw_tweet               text
#);

#ALTER TABLE twitter_corpus CONVERT TO CHARACTER SET UTF8 COLLATE utf8_general_ci;

SELECT * FROM twitter_corpus;

select * from twitter_corpus where tweet_id = 989177695471681536;

#truncate table twitter_corpus;