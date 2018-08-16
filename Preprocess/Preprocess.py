import re
import nltk
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# paths
root_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
affix_file_path = 'D:/Workspace/python/Sarcasm Detector Study/Corpus/affix-list.txt'

# Stopwords
stopwords_file = open(root_path + 'id.stopwords.02.01.2016.txt', encoding='utf-8')
stopwords_list = stopwords_file.readlines()
stopwords = []

for line in stopwords_list :
    stopwords.append(line.rstrip('\n'))

#print (stopwords)

# Abbreviation Expanding Dictionary
abbreviationDictionary = {
    'gk'        : 'tidak',
    'gak'       : 'tidak',
    'ga'        : 'tidak',
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
    'rp'        : 'rupiah',
    'malem'     : 'malam',
    'pengen'    : 'ingin',
    'sampe'     : 'sampai',
    'tmn'       : 'teman',
    'kl'        : 'kalau',
    'inget'     : 'ingat',
    'nubruk'    : 'tabrak',
    'ttg'       : 'tentang',
    'ampe'      : 'sampai',
    'gw'        : 'saya',
    'gua'       : 'saya',
    'elu'       : 'anda',
    'lu'        : 'anda',
    'prnh'      : 'pernah',
    'krna'      : 'karena',
    'krn'       : 'karena',
    'krng'      : 'kurang',
    'bru'       : 'baru',
    'br'        : 'baru',
    'nyampe'    : 'sampai',
    'tau'       : 'tahu',
    'trus'      : 'terus',
    'aja'       : 'saja',
    'ilang'     : 'hilang',
    'oon'       : 'bodoh',
    'bgt'       : 'sekali',
    'pengin'    : 'ingin',
    'karna'     : 'karena'
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

def affixConcatenator (words, dictionary) :
    word_count = len(words)
    i = 0
    returned_words = []
    
    while i < word_count :
        if i+1 < word_count :
            if words[i] in dictionary :
                returned_words.append(words[i] + '_' + words[i+1])
                i = i+2
            else :
                returned_words.append(words[i])
                i = i+1
        else :
            returned_words.append(words[i])
            i = i+1

    return returned_words

# dummy pipeline
#dummyTweet = '@Humanxsick: suka la cakaaap pasal [cinta] cintany ni. aku gk alergik {dsb}. bo la duk gewe nok-nok pese2 kuyyy. #eh http://www.fff.ccc/'
dummyTweet = 'lo pernah ga sih nemu manusia saking pintarnya mereka jadi tidak bodoh. aka amuro + conan combo'

def preprocessPipeline1 (words) :
    # rule 1
    if (scanRtTag(words)) :
        return ''

    # rule 2 Remove twitter special tags
    processedWords = removeSpecialTags(words)

    print (processedWords)

    # rule 3 Remove URL
    processedWords = removeUrl(processedWords)

    print (processedWords)

    # rule 4 Tokenize
    processedWords = tokenizeTweet(processedWords)

    print (processedWords)

    # rule 5 Abbreviation expanding
    expandedWord = []
    for word in processedWords :
        expandedTrial = abbreviationExpander(word, abbreviationDictionary)
        if type(expandedTrial) == list:
            for token in expandedTrial:
                expandedWord.append(token)
        else:
            expandedWord.append(expandedTrial)

    print (expandedWord)

    # rule 6 Duplicated letter removal
    duplicateLetterRemovedWords = []
    for word in expandedWord :
        duplicateLetterRemovedWords.append(duplicateLetterRemoval(word))

    print (duplicateLetterRemovedWords)

    # rule 7 reversed word
    reversedExpandedWords = []
    for word in duplicateLetterRemovedWords :
        reversedExpandedWords.append(reversedWordNormalisation(word, reversedWordDictionary))
    
    print (reversedExpandedWords)

    # rule 8 Reduplication normalization
    reduplicationNormalizedWords = []
    for word in reversedExpandedWords :
        reduplicationNormalizedWords.append(reduplicationNormalization(word))

    print (reduplicationNormalizedWords)

    # rule 8 remove unused punctuation
    unusedPunctuationRemovedWords = removeUnusedPunctuation (reduplicationNormalizedWords)

    print (unusedPunctuationRemovedWords)

    # rule 9 stemming
    stemmedWords = stemWords(unusedPunctuationRemovedWords)

    print (stemmedWords)

    # rule 10 affix concatenator
    affixDictionary = [line.rstrip('\n') for line in open(affix_file_path, mode='r', encoding='utf-8')]
    afficConcatenated = affixConcatenator(stemmedWords, affixDictionary)

    print (afficConcatenated)

    # rule 11 Stopword removal
    stopwordRemoved = stopwordRemoval(afficConcatenated, stopwords)

    print (stopwordRemoved)
    
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

print (preprocessPipeline1('Pernah ga sih, pengin bgt beli sesuatu barang, karna mahal ngumpulin duit, saking lamaaaanya belum cukup cukup juga… https://t.co/Z4kUJyoxKz'))