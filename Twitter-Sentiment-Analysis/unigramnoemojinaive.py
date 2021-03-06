#import regex
import re
import csv
import pprint
import nltk.classify
import pickle

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

#find the sentiment score on tweets through positive 
#and negative words
def posnegcheck(tweet):
	num=0
	words = tweet.split()
	for w in words:
		if w in positivedict.keys():
			num=num+1
		if w in negativedict.keys():
			num=num-1
	return num

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

	#extract emoji name from tweets emoji
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
def processTweet(tweet):
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


# save the processed tweet
def saveprocessed(processedTweet,fileName):
    fw = open(fileName, 'a')
    fw.write(processedTweet)
    fw.write("\n")
    fw.close()

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
inpTweets = csv.reader(open('d:\\tweetsfilter.csv', 'rb'), delimiter=',', quotechar='|')
stopWords = getStopWordList('d:\\stopwords.txt')
count = 0
featureList = []
tweets = []
for row in inpTweets:
    sentiment = row[0].lower()
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    for w in featureVector:	
     savefile=open("noemojiList.txt","a")
     savefile.write(w+"\n")
     savefile.close()	
    tweets.append((featureVector, sentiment)); 
  
#end loop

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

print NBClassifier.show_most_informative_features(10)

#save CLassifier
save_classifier=open("noemojiunigarmnaivebayes.classifier","wb")
pickle.dump(NBClassifier, save_classifier)
save_classifier.close()


#test part
inpTweets = csv.reader(open('d:\\tweetsfilter.csv', 'rb'), delimiter=',', quotechar='|')
count = 0
counttest=0
for row in inpTweets:
    testsentiment = row[0].lower()
    testtweet = row[1]
    processedTestTweet = processTweet(testTweet)
    sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
    #save features in a file
    savefile=open("noemojiunigarmnaivebayes.txt","a")
    savefile.write(sentiment+" | "+testsentiment)
    savefile.write("\n")	
    savefile.close()		


