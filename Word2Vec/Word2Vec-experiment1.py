import sys
import collections

raw_text_path = 'D:/Workspaces/python/Sarcasm Detector Study/Corpus/text/AA/wiki_00'
raw_text = ''
vocabulary = ''
vocabulary_size = 50000

with open(raw_text_path, mode='r', encoding='utf-8') as file:
    for line in file:
        raw_text = raw_text + line

vocabulary = raw_text.split() # read raw data, split into list

print (len(vocabulary))
#print (sys.maxsize)

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1)) # tokenize the most common words with integer
  dictionary = dict()
  for word, _ in count: # we don't use the int, the '_', from the Counter process
    #print (word + ' ' + str(_))
    dictionary[word] = len(dictionary) # this, will put unique int for each word
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, vocabulary_size)

#print (data)
#print (reversed_dictionary[1] + reversed_dictionary[6036])