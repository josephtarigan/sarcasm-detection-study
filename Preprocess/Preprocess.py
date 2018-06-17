import re
import nltk
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# paths
root_path = os.path.dirname(os.path.abspath(__file__)) + os.sep

# Stopwords
stopwords_file = open(root_path + 'id.stopwords.02.01.2016.txt', encoding='utf-8')
stopwords_list = stopwords_file.readlines()
stopwords = []

for line in stopwords_list :
    stopwords.append(line.rstrip('\n'))

print (stopwords)

# Abbreviation Expanding Dictionary
abbreviationDictionary = {
    'gk'        : 'tidak',
    'gak'       : 'tidak',
    'tdk'       : 'tidak',
    'nggak'     : 'tidak',
    'iy'        : 'iya',
    'yg'        : 'yang',
    'jg'        : 'juga',
    'dst'       : 'dan seterusnya',
    'dsb'       : 'dan sebagainya',
    'sy'        : 'saya',
    'km'        : 'kamu',
    'mrk'       : 'mereka',
    'indo'      : 'indonesia',
    'jpn'       : 'japan',
    'sila'      : 'silakan',
    'jgn'       : 'jangan',
    'dpt'       : 'dapat',
    'y'         : 'ya',
    'ato'       : 'atau',
    'gt'        : 'begitu',
    'drpd'      : 'daripada',
    'tp'        : 'tapi',
    'kk'        : 'kakak',
    'makasih'   : 'terima kasih',
    'ty'        : 'thank you',
    '&'         : 'dan',
    '%'         : 'persen',
    '$'         : 'dolar',
    'rp'        : 'rupiah'
}

reversedWordDictionary = {
    'kuy'       : 'yuk'
}

# Steps:
# 1. Remove tweets start with RT token
# 2. Remove twitter special tags, such as @ and #
# 3. Remove URL
# 4. Tokenize
# 5. Abbreviation expander
# 6. Duplicated letter removal
# 7. Reversed word normalisation
# 8. Reduplication word normalization
# 9. Remove unused punctuation
# 10. Stemming / Lemmatize
# 11. Stopwords removal

# to consider:
# 1. Emoticons handling

# Remove tweets start with RT token
# takes string of tweet
# return boolean
def scanRtTag (tweet) :
    rtTagIsPresent = False
    p = re.compile("RT")
    if (p.match(tweet)) :
        rtTagIsPresent = True
    return rtTagIsPresent

# Remove twitter special tags
# @ # ; :
# return cleaned text
def removeSpecialTags (tweet) :
    # tags that will be removed:
    # @#:;
    return re.sub(r'#*@*:*;*', '', tweet)

# Remove URL
# return cleaned text
def removeUrl (tweet) :
    return re.sub(r'http\S+', '', tweet, flags=re.MULTILINE)

# Tokenize 
def tokenizeTweet (text) :
    tokenizer = nltk.tokenize.TweetTokenizer()
    return tokenizer.tokenize(text)

# Abbreviation Expander
# Takes word token and a dictionary
# Will check for token occurence in the dictionary with word order and matching weigthing algorithm
def abbreviationExpander (word, dict) :
    if (word in dict) :
        tokenizer = nltk.tokenize.TweetTokenizer()
        return tokenizer.tokenize(dict[word])
    else :
        # Special case for Bahasa Indonesia. If the last 2 letters is 'ny', expand it to 'nya'
        p = re.compile(r"ny$")
        if (p.search(word)) :
            return (word + 'a')
        else :
            return word

# Reversed word normalisation
# For example like kuy --> yuk
# Takes the world and the dictionary
def reversedWordNormalisation (word, dict) :
    if (word in dict) :
        return dict[word]
    else :
        return word

# Remove reduplication
# Finds occurence of 2
# Find occurence of hypen '-'
def reduplicationNormalization (word) :
    # scan for number 2 in the middle or in the end of the string
    word = re.sub(r'2$', '', word)
    word = re.sub(r'2', '', word)

    # find cooccurrence chars
    '''
    chars = list(word)
    outputChars = []
    tempChars = []
    i = 1
    for char in chars :
        tempChars.append(char)
        outputChars.append(char)

        if (i%2 = 0) :
            
        i = i+1
    '''

    # Find occurence of hypen '-'
    words = word.split('-')
    if ((len(words) > 1) and (words[1:] == words[:-1])) :
        return words[0]
    else :
        return word

# Normalise duplicated letter
# Takes word and will try to normalise the word
def duplicateLetterRemoval (word) :
    letters = list(word)
    letterBin = []
    tempLetter = ''
    for letter in letters :
        if tempLetter != letter :
            letterBin.append(letter)
        tempLetter = letter
    return ''.join(letterBin)

# Remove unused punctuation
# Takes tokens
def removeUnusedPunctuation (tokens) :
    unusedPunctuation = '[]{}()*^~/.,;:"\'\\'
    processedTokens = []
    for token in tokens :
        if (token not in unusedPunctuation) :
            processedTokens.append(token)
    return processedTokens

# Stemming words
# Takes tokens and try to stem each token
# Uses sastrawi library, ported to python
# https://github.com/har07/PySastrawi
# Returns processed tokens
def stemWords (words) :
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    output = stemmer.stem(' '.join(words))
    return tokenizeTweet(output)

# Stopword removal
# Returns words that exist in the stopword list
def stopwordRemoval (words, stopwords) :
    filtered = []
    for word in words :
        if word not in stopwords :
            filtered.append(word)
    return filtered


# dummy pipeline
dummyTweet = '@Humanxsick: suka la cakaaap pasal [cinta] cintany ni. aku gk alergik {dsb}. bo la duk gewe nok-nok pese2 kuyyy. #eh http://www.fff.ccc/'

def preprocessPipeline1 (words) :
    # rule 1
    if (scanRtTag(words)) :
        return ''

    # rule 2 Remove twitter special tags
    processedWords = removeSpecialTags(words)

    # rule 3 Remove URL
    processedWords = removeUrl(processedWords)

    # rule 4 Tokenize
    processedWords = tokenizeTweet(processedWords)

    # rule 5 Abbreviation expanding
    expandedWord = []
    for word in processedWords :
        expandedTrial = abbreviationExpander(word, abbreviationDictionary)
        if type(expandedTrial) == list:
            for token in expandedTrial:
                expandedWord.append(token)
        else:
            expandedWord.append(expandedTrial)

    # rule 6 Duplicated letter removal
    duplicateLetterRemovedWords = []
    for word in expandedWord :
        duplicateLetterRemovedWords.append(duplicateLetterRemoval(word))

    # rule 7 reversed word
    reversedExpandedWords = []
    for word in duplicateLetterRemovedWords :
        reversedExpandedWords.append(reversedWordNormalisation(word, reversedWordDictionary))
    #print (reversedExpandedWords)

    # rule 8 Reduplication normalization
    reduplicationNormalizedWords = []
    for word in reversedExpandedWords :
        reduplicationNormalizedWords.append(reduplicationNormalization(word))

    # rule 8 remove unused punctuation
    unusedPunctuationRemovedWords = removeUnusedPunctuation (reduplicationNormalizedWords)
    #print (unusedPunctuationRemovedWords)

    # rule 9 stemming
    stemmedWords = stemWords(unusedPunctuationRemovedWords)

    # rule 11 Stopword removal
    stopwordRemoved = stopwordRemoval(stemmedWords, stopwords)

    return stopwordRemoved

# non stopwords
def preprocessPipeline2 (words) :
    # rule 1
    if (scanRtTag(words)) :
        return ''

    # rule 2 Remove twitter special tags
    processedWords = removeSpecialTags(words)

    # rule 3 Remove URL
    processedWords = removeUrl(processedWords)

    # rule 4 Tokenize
    processedWords = tokenizeTweet(processedWords)

    # rule 5 Abbreviation expanding
    expandedWord = []
    for word in processedWords :
        expandedTrial = abbreviationExpander(word, abbreviationDictionary)
        if type(expandedTrial) == list:
            for token in expandedTrial:
                expandedWord.append(token)
        else:
            expandedWord.append(expandedTrial)

    # rule 6 Duplicated letter removal
    duplicateLetterRemovedWords = []
    for word in expandedWord :
        duplicateLetterRemovedWords.append(duplicateLetterRemoval(word))

    # rule 7 reversed word
    reversedExpandedWords = []
    for word in duplicateLetterRemovedWords :
        reversedExpandedWords.append(reversedWordNormalisation(word, reversedWordDictionary))
    #print (reversedExpandedWords)

    # rule 8 Reduplication normalization
    reduplicationNormalizedWords = []
    for word in reversedExpandedWords :
        reduplicationNormalizedWords.append(reduplicationNormalization(word))

    # rule 8 remove unused punctuation
    unusedPunctuationRemovedWords = removeUnusedPunctuation (reduplicationNormalizedWords)
    #print (unusedPunctuationRemovedWords)

    # rule 9 stemming
    stemmedWords = stemWords(unusedPunctuationRemovedWords)

    return stemmedWords

print (preprocessPipeline1(dummyTweet))