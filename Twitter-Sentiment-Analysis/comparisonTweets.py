#import regex
import re
import csv
import pprint
import nltk.classify
import pickle
import requests
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
r = requests.get('http://localhost:9200') 
es.indices.delete(index="twitter", ignore=[400, 404])
mappings = {
    "doc": {
        "properties": {
            "elasticsearch": {
                 "properties": {
                     "sentiment": {
                         "type": "string"
                     }
                 }
             }
        }
    }
}
#creation of index with the mapping
es.indices.create(index='twitter', body=mappings)

#save negative words in a dictionary
negativedict={}
fp = open('negativewords.txt', 'r')
line = fp.readline()
while line:
	word = line.strip()
	negativedict[word]=1
	line = fp.readline()
fp.close()

#save positive words in a dictionary
positivedict={}
fp = open('positivewords.txt', 'r')
line = fp.readline()
while line:
	word = line.strip()
	positivedict[word]=1
	line = fp.readline()
fp.close()

#save emoji in a dictionary
emojidict={}
fp = open('emojicollection.txt', 'r')
line = fp.readline()
while line:
	word = line.strip()
	list=word.split(',')
	emojidict[list[1]]=list[0]
	line = fp.readline()
fp.close()

#save emoji Value in a dictionary
emojiValue={}
fp = open('emojicollection.txt', 'r')
line = fp.readline()
while line:
	word = line.strip()
	list=word.split(',')
	emojiValue[list[1]]=list[2]
	line = fp.readline()
fp.close()

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

def posnegcheck(tweet):
	num=0
	words = tweet.split()
	for w in words:
		if w in positivedict.keys():
			num=num+1
		if w in negativedict.keys():
			num=num-1
	return num
	
def findemoji(tweet):
	num=0
	i=0
	while i<len(tweet):
		if(tweet.find('\ud',i,len(tweet))==-1):
			break
		else:	
			i=tweet.find('\ud',i,len(tweet))		
			if tweet[i:i+12] in emojiValue.keys(): 
				#print(emojidict[tweet[i:i+12]])
				#tweet=tweet.replace(tweet[i:i+12]," "+emojidict[tweet[i:i+12]]+" "+emojidict[tweet[i:i+12]]+" ")
				if emojiValue[tweet[i:i+12]]>0:
					num=num+2
				elif emojiValue[tweet[i:i+12]]<0:
					num=num-2
				else:
					num=num
				i=i+3
			else: 
				i=i+1
	return num			

def extractemoji(tweet):
	i=0
	while i<len(tweet):
		if(tweet.find('\ud',i,len(tweet))==-1):
			break
		else:	
			i=tweet.find('\ud',i,len(tweet))		
			if tweet[i:i+12] in emojidict.keys(): 
				#print(emojidict[tweet[i:i+12]])
				tweet=tweet.replace(tweet[i:i+12]," "+emojidict[tweet[i:i+12]]+" "+emojidict[tweet[i:i+12]]+" ")
				i=i+3
			else: 
				i=i+1
	return tweet			

#start process_tweet
def processemojiTweet(tweet):
    # process the tweets
    tweet=extractemoji(tweet)
    #Convert to lower case
    tweet = tweet.lower()
    #Remove www.* or https?:
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)
    #Remove @username
    tweet = re.sub('@[^\s]+',' ',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 

#start process_tweet
def processnoemojiTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 


#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVectorBi(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    past_word=""
    word_count=0	
    for w in words:
		#use of lemmatizer
            w=lemmatizer.lemmatize(w)	
        #replace two or more with two occurrences 
            w = replaceTwoOrMore(w) 
        #strip punctuation
            w = w.strip('\'"?,.')        		
        #check if it starts with an alphabets
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
		#add features
            if word_count>0:
			  featureVector.append(past_word+" "+w.lower())
            past_word=w.lower()	
            word_count=word_count+1			
    return featureVector    
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
		#use of lemmatizer
        w=lemmatizer.lemmatize(w)	
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it starts with alphabets
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector  
	
#start extract_features
def extract_features(tweet, featureList):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end

stopWords = getStopWordList('d:\\stopwords.txt')

# no emoji bigram classifier
noemojibigramfeatureList = []
fp = open("bigramnaiveList.txt", 'r')
line = fp.readline()
while line:
        word = line.strip()
        noemojibigramfeatureList.append(word)
        line = fp.readline()
fp.close()

# open the pickle
classifier_f=open("noemojibigramnaivebayes.classifier","rb")
noemojibigramClassifier= pickle.load(classifier_f)
classifier_f.close()


# emoji bigram classifier

bigramemojifeatureList = []
fp = open("bigramemojiNaive.txt", 'r')
line = fp.readline()
while line:
        word = line.strip()
        bigramemojifeatureList.append(word)
        line = fp.readline()
fp.close()

# open the pickle
classifier_f=open("bigramemojinaivebayes.classifier","rb")
bigramemojiClassifier= pickle.load(classifier_f)
classifier_f.close()

# no emoji unigram 

noemojiunigramfeatureList = []
fp = open("noemojiList.txt", 'r')
line = fp.readline()
while line:
        word = line.strip()
        noemojiunigramfeatureList.append(word)
        line = fp.readline()
fp.close()

# open the pickle
classifier_f=open("noemojiunigarmnaivebayes.classifier","rb")
noemojiunigramClassifier= pickle.load(classifier_f)
classifier_f.close()

#unigram emoji

unigramemojifeatureList = []
fp = open("unigramnaiveList.txt", 'r')
line = fp.readline()
while line:
        word = line.strip()
        unigramemojifeatureList.append(word)
        line = fp.readline()
fp.close()

# open the unigram naive classifier
classifier_f=open("unigramemojinaive.classifier","rb")
unigramemojiClassifier= pickle.load(classifier_f)
classifier_f.close()

#test part
inpTweets = csv.reader(open('d:\\tweetsfilter.csv', 'rb'), delimiter=',', quotechar='|')
count = 0
counttest=0
j=0
for row in inpTweets:
  noemojibigramscore=0
  noemojiunigramscore=0
  bigramscore=0
  unigramscore=0
  if count>55000:
    testsentiment = row[0].lower()
    testtweet = row[1]
    processednoemojiTweet = processnoemojiTweet(testtweet)
    processedemojiTweet = processemojiTweet(testtweet)	
    noemojibigramsentiment = noemojibigramClassifier.classify(extract_features(getFeatureVectorBi(processednoemojiTweet, stopWords), noemojibigramfeatureList))
    bigramsentiment = bigramemojiClassifier.classify(extract_features(getFeatureVectorBi(processedemojiTweet, stopWords), bigramemojifeatureList))
    noemojiunigramsentiment = noemojiunigramClassifier.classify(extract_features(getFeatureVector(processednoemojiTweet, stopWords), noemojiunigramfeatureList))
    unigramsentiment = unigramemojiClassifier.classify(extract_features(getFeatureVector(processedemojiTweet, stopWords), unigramemojifeatureList))
    score=posnegcheck(processednoemojiTweet) 	
	# no emoji bigram
	
    if noemojibigramsentiment=="positive":
       noemojibigramscore =1	
    if noemojibigramsentiment=="negative":
       noemojibigramscore =-1	
    noemojibigramscore=score+noemojibigramscore 
    if noemojibigramscore >0:
       noemojibigramsentiment ="positive"	
    elif noemojibigramscore <0:
       noemojibigramsentiment ="negative"	
    else:
       noemojibigramsentiment ="neutral"	
	  
	  #bigram emoji
	  
    if bigramsentiment=="positive":
       bigramscore =1	
    if bigramsentiment=="negative":
       bigramscore =-1	
    bigramscore=score+bigramscore+findemoji(testtweet)
    if bigramscore>0:
       bigramsentiment="positive"	
    elif bigramscore<0:
       bigramsentiment="negative"	
    else:
       bigramsentiment="neutral"	

	   #noemojiunigramsentiment
    if noemojiunigramsentiment=="positive":
       noemojiunigramscore=1	
    if noemojiunigramsentiment=="negative":
       noemojiunigramscore=-1	
    noemojiunigramscore=score+noemojiunigramscore
    if noemojiunigramscore>0:
       noemojiunigramsentiment="positive"	
    elif noemojiunigramscore<0:
       noemojiunigramsentiment="negative"	
    else:
       noemojiunigramsentiment="neutral"	

	   #unigramsentiment
    if unigramsentiment=="positive":
       unigramscore=1	
    if unigramsentiment=="negative":
       unigramscore=-1	
    unigramscore=score+unigramscore+findemoji(testtweet)
    if unigramscore>0:
       unigramsentiment="positive"	
    elif unigramscore<0:
       unigramsentiment="negative"	
    else:
       unigramsentiment="neutral"	
	   	
    es.index(index='twitter', doc_type='elasticsearch', id=j, body={"doc": {"tweets": testtweet }})
    es.update(index="twitter", doc_type='elasticsearch', id=j, body={"doc": {"unigramnoemoji": noemojiunigramsentiment }})
    es.update(index="twitter", doc_type='elasticsearch', id=j, body={"doc": {"bigramnoemoji": noemojibigramsentiment }})
    es.update(index="twitter", doc_type='elasticsearch', id=j, body={"doc": {"unigramemoji": unigramsentiment }})
    es.update(index="twitter", doc_type='elasticsearch', id=j, body={"doc": {"bigramemoji": bigramsentiment }})								
    j=j+1		
  count=count+1		
print("Neutral")
resN = es.search(index="twitter", body={"query": {"match": {'unigramnoemoji':'neutral'}}})
print resN['hits']['total']        
resN = es.search(index="twitter", body={"query": {"match": {'bigramnoemoji':'neutral'}}})
print resN['hits']['total'] 
resN = es.search(index="twitter", body={"query": {"match": {'unigramemoji':'neutral'}}})
print resN['hits']['total']  
resN = es.search(index="twitter", body={"query": {"match": {'bigramemoji':'neutral'}}})
print resN['hits']['total']  
print("Positive")
resN = es.search(index="twitter", body={"query": {"match": {'unigramnoemoji':'positive'}}})
print resN['hits']['total']        
resN = es.search(index="twitter", body={"query": {"match": {'bigramnoemoji':'positive'}}})
print resN['hits']['total'] 
resN = es.search(index="twitter", body={"query": {"match": {'unigramemoji':'positive'}}})
print resN['hits']['total']  
resN = es.search(index="twitter", body={"query": {"match": {'bigramemoji':'positive'}}})
print resN['hits']['total']  
print("Negative")
resN = es.search(index="twitter", body={"query": {"match": {'unigramnoemoji':'negative'}}})
print resN['hits']['total']        
resN = es.search(index="twitter", body={"query": {"match": {'bigramnoemoji':'negative'}}})
print resN['hits']['total'] 
resN = es.search(index="twitter", body={"query": {"match": {'unigramemoji':'negative'}}})
print resN['hits']['total']  
resN = es.search(index="twitter", body={"query": {"match": {'bigramemoji':'negative'}}})
print resN['hits']['total']  
