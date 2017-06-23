
import re
import csv
import pprint
import nltk.classify
import pickle
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

#save emoji unicode with emoji name in a dictionary
emojidict={}
fp = open('emojicollection.txt', 'r')
line = fp.readline()
while line:
	word = line.strip()
	list=word.split(',')
	emojidict[list[1]]=list[0]
	line = fp.readline()
fp.close()

#save emoji unicode with score in a dictionary
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

#find emoji in tweets	
def findemoji(tweet):
	num=0
	i=0
	while i<len(tweet):
		if(tweet.find('\ud',i,len(tweet))==-1):
			break
		else:	
			i=tweet.find('\ud',i,len(tweet))		
			if tweet[i:i+12] in emojiValue.keys(): 
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

#extract emoji name from tweets emoji
def extractemoji(tweet):
	i=0
	while i<len(tweet):
		if(tweet.find('\ud',i,len(tweet))==-1):
			break
		else:	
			i=tweet.find('\ud',i,len(tweet))		
			if tweet[i:i+12] in emojidict.keys(): 
				tweet=tweet.replace(tweet[i:i+12]," "+emojidict[tweet[i:i+12]]+" "+emojidict[tweet[i:i+12]]+" ")
				i=i+3
			else: 
				i=i+1
	return tweet			

#start processing tweet
def processTweet(tweet):
    # process the emoji
    tweet=extractemoji(tweet)
    #Convert to lower case
    tweet = tweet.lower()
    #Remove www.* or https?://*
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


#start get stopwords in an array
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end


#start getfeatureVector for bigram
def getFeatureVector(tweet, stopWords):
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
        #check if it consists of only words
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #add feature
            if word_count>0:
			  featureVector.append(past_word+" "+w.lower())
            past_word=w.lower()	
            word_count=word_count+1			
    return featureVector    
#end 

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


#Read the tweets one by one and process it
inpTweets = csv.reader(open('tweets.csv', 'rb'), delimiter=',', quotechar='|')
stopWords = getStopWordList('stopwords.txt')
count = 0
featureList = []
tweets = []
for row in inpTweets:
  sentiment = row[0].lower()
  tweet = row[1]
  if count<55000:	
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    #saves feature list in a file	
    for w in featureVector:	
     savefile=open("bigramemojiNaive.txt","a")
     savefile.write(w+"\n")
     savefile.close()	
    tweets.append((featureVector, sentiment));	  
  count=count+1	
#end loop

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

#print 10 most common features
print NBClassifier.show_most_informative_features(10)

#save CLassifier
save_classifier=open("bigramemojinaivebayes.classifier","wb")
pickle.dump(NBClassifier, save_classifier)
save_classifier.close()

#test part
inpTweets = csv.reader(open('tweets.csv', 'rb'), delimiter=',', quotechar='|')
count = 0
counttest=0
for row in inpTweets:
  if count>55000:
    testsentiment = row[0].lower()
    testtweet = row[1]
    processedTestTweet = processTweet(testtweet)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
    print (sentiment)
  count=count+1		
