from flask import Flask, request
from flask import render_template

app = Flask(__name__)
import json
import argparse
import urllib
import json
import os
import oauth2
import re
import csv
import nltk
import requests
import re, pickle, csv, os
import time
import operator
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from elasticsearch import Elasticsearch
lemmatizer=WordNetLemmatizer()
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
r = requests.get('http://localhost:9200') 
i = 1
countNounPhrase={}
es.indices.delete(index="twittermid", ignore=[400, 404])
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
es.indices.create(index='twittermid', body=mappings)
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
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    past_word=""
    word_count=0	
    for w in words:
            w=lemmatizer.lemmatize(w)	
        #replace two or more with two occurrences 
            w = replaceTwoOrMore(w) 
        #strip punctuation
            w = w.strip('\'"?,.')
        		
        #check if it consists of only words
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        #if(w in stopWords or val is None):
            #continue
        #else:		
            if word_count>0:
			  featureVector.append(past_word+" "+w.lower())
            past_word=w.lower()	
            word_count=word_count+1			
    return featureVector    
#end

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

#class to connect with twiiter api and get data related to text
class TwitterData:
    def parse_config(self):
        config = {}
        # from file args
        if os.path.exists('config.json'):
            with open('config.json') as f:
                config.update(json.load(f))
        else:
            # may be from command line
            parser = argparse.ArgumentParser()

            parser.add_argument('-ck', '--consumer_key', default=None, help='Your developper `Consumer Key`')
            parser.add_argument('-cs', '--consumer_secret', default=None, help='Your developper `Consumer Secret`')
            parser.add_argument('-at', '--access_token', default=None, help='A client `Access Token`')
            parser.add_argument('-ats', '--access_token_secret', default=None, help='A client `Access Token Secret`')

            args_ = parser.parse_args()
            def val(key):
                return config.get(key)\
                    or getattr(args_, key)\
                    or raw_input('Your developper `%s`: ' % key)
            config.update({
                'consumer_key': val('consumer_key'),
                'consumer_secret': val('consumer_secret'),
                'access_token': val('access_token'),
                'access_token_secret': val('access_token_secret'),
            })
        # should have something now
        return config
    #end

    def oauth_req(self, url, http_method="GET", post_body=None,
                  http_headers=None):
        config = self.parse_config()
        consumer = oauth2.Consumer(key=config.get('consumer_key'), secret=config.get('consumer_secret'))
        token = oauth2.Token(key=config.get('access_token'), secret=config.get('access_token_secret'))
        client = oauth2.Client(consumer, token)

        resp, content = client.request(
            url,
            method=http_method,
            body=post_body or '',
            headers=http_headers
        )
        return content
    #end

    #start getTwitterData
    def getData(self, keyword, maxID,j):
        maxTweets = 100
        url = 'https://api.twitter.com/1.1/search/tweets.json?'
        if maxID==0:		
            data = {'q': keyword, 'lang': 'en', 'result_type': 'recent', 'count': maxTweets, 'include_entities': 0}
        else:
            data = {'q': keyword, 'lang': 'en', 'result_type': 'recent', 'count': maxTweets, 'include_entities': 0, 'max_id':maxID}

        #Add if additional params are passed

        url += urllib.urlencode(data)

        response = self.oauth_req(url)
        jsonData = json.loads(response)
        tweets = []
        s=0
        if 'errors' in jsonData:
            print "API Error"
            print jsonData['errors']
        else:
            for item in jsonData['statuses']:
				s =item['id']
				sepwords = json.dumps(item['text']).encode("utf-8").split()
				i=0
				for word in sepwords:
					sepwords[i]=lemmatizer.lemmatize(word)
					i=i+1
				testtweet=" ".join(sepwords)				
				noemojibigramscore=0
				noemojiunigramscore=0
				bigramscore=0
				unigramscore=0
				processednoemojiTweet = processnoemojiTweet(testtweet)
				processedemojiTweet = processemojiTweet(testtweet)	
				noemojibigramsentiment = noemojibigramClassifier.classify(extract_features(getFeatureVector(processednoemojiTweet, stopWords), noemojibigramfeatureList))
				bigramsentiment = bigramemojiClassifier.classify(extract_features(getFeatureVector(processedemojiTweet, stopWords), bigramemojifeatureList))
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
	   	
				es.index(index='twittermid', doc_type='elasticsearch', id=j, body={"doc": {"tweets": testtweet }})
				es.update(index="twittermid", doc_type='elasticsearch', id=j, body={"doc": {"unigramnoemoji": noemojiunigramsentiment }})
				es.update(index="twittermid", doc_type='elasticsearch', id=j, body={"doc": {"bigramnoemoji": noemojibigramsentiment }})
				es.update(index="twittermid", doc_type='elasticsearch', id=j, body={"doc": {"unigramemoji": unigramsentiment }})
				es.update(index="twittermid", doc_type='elasticsearch', id=j, body={"doc": {"bigramemoji": bigramsentiment }})								
    
				j=j+1
				#time.sleep(2)				
        return s

#home template
@app.route('/')
def index():
    return render_template('home.html', title='Home')


@app.route('/hash', methods=['GET', 'POST'])
def search():
	if request.method == 'POST':
		td = TwitterData()
		maID=0
		i=0
		j=0
		while i<1:
				maID=td.getData(request.form['hashtext'],maID,j)
				j=j+100   
				i=i+1
		binoemonu = es.search(index="twittermid", body={"query": {"match": {'bigramnoemoji':'neutral'}}})
		biemonu = es.search(index="twittermid", body={"query": {"match": {'bigramemoji':'neutral'}}})
		binoemopo = es.search(index="twittermid", body={"query": {"match": {'bigramnoemoji':'positive'}}})
		biemopo = es.search(index="twittermid", body={"query": {"match": {'bigramemoji':'positive'}}})
		binoemone = es.search(index="twittermid", body={"query": {"match": {'bigramnoemoji':'negative'}}})
		biemone = es.search(index="twittermid", body={"query": {"match": {'bigramemoji':'negative'}}})
			
		uninoemonu = es.search(index="twittermid", body={"query": {"match": {'unigramnoemoji':'neutral'}}})
		uniemonu = es.search(index="twittermid", body={"query": {"match": {'unigramemoji':'neutral'}}})
		uninoemopo = es.search(index="twittermid", body={"query": {"match": {'unigramnoemoji':'positive'}}})
		uniemopo = es.search(index="twittermid", body={"query": {"match": {'unigramemoji':'positive'}}})
		uninoemone = es.search(index="twittermid", body={"query": {"match": {'unigramnoemoji':'negative'}}})
		uniemone = es.search(index="twittermid", body={"query": {"match": {'unigramemoji':'negative'}}})
		return render_template('unigram.html',uniemopo=uniemopo,uniemone=uniemone,uniemonu=uniemonu,uninoemopo=uninoemopo,uninoemone=uninoemone,uninoemonu=uninoemonu,biemopo=biemopo,biemone=biemone,biemonu=biemonu,binoemopo=binoemopo,binoemone=binoemone,binoemonu=binoemonu)
		
if __name__ == '__main__':
    app.run()
