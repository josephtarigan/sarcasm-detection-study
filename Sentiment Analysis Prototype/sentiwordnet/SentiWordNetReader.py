import os

file_path = 'resource/SentiWordNet_3.0.0_20130122.txt'

def get_sentiment_score (word):
    f = open(os.path.dirname(__file__) + '/' + file_path, 'r')
    found = False
    for line in f:
        if not line.startswith('#'):
            cols = line.split('\t')
            words_ids = cols[4].split(' ')
            words = [w.split('#')[0] for w in words_ids]

            if word in words:
                print ('Word %s - %s' % (words[0], cols[5]))
                print ('Positive Sentiment : %s' % (cols[2]))
                print ('Negative Sentiment : %s' % (cols[3]))
                print ('Objectivity : %f' % (1 - (float(cols[2]) + float(cols[3]))))
                print ('Sentiment : %f' % (float(cols[2]) - float(cols[3])))
                found = True
            
            if (found):
                break
    
    if (not found):
        print ('Word %s is not found' % (word))

def get_weighted_average_sentiment_score ():
    f = open(os.path.dirname(__file__) + '/' + file_path, 'r')
    temp_dict = {}
    word_dict = {}
    i = 1
    for line in f:
        if not line.startswith('#'):
            cols = line.split('\t')
            
            if (len(cols) == 6 and cols[0] is not ''):
                words_ids = cols[4].split(' ')
                wordsAndRank = [w.split('#') for w in words_ids]
                #print ('%s - %s - %s - %s - %s - %s' % (cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]))
                sentiment_score = (float(cols[2]) - float(cols[3]))

                for synset in wordsAndRank:
                    syn_word = synset[0]
                    score = synset[1]
                    synsetKey = syn_word + '#' + cols[0]

                    if (temp_dict.get(synsetKey) is None):
                        temp_dict[synsetKey] = {}
                    
                    temp_dict[synsetKey][score] = sentiment_score
        i += 1
    
    for key, value in temp_dict.items():
        score = 0
        sum = 0
        for itemKey in value.keys():
            score += float(value.get(itemKey))/float(itemKey)
            sum += 1 / float(itemKey)
        score /= sum
        word_dict[key] = score

    return word_dict


#get_sentiment_score('good')

for key, value in get_weighted_average_sentiment_score().items():
    print ('%s - %s' %(key, value))