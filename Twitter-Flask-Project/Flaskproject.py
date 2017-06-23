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
def checkhash(hashfun, value):
	if value in hashfun.keys():
		hashfun[value]= hashfun[value]+1
	else:
		hashfun[value]= 1

def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #Convert @username to blank
    tweet = re.sub('@[^\s]+','',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet=tweet.replace('rt',"")
    tweet = re.sub('\u[^\s]+','',tweet)
    tweet = re.sub('\\u[^\s]+','',tweet)	
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end


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
				sepwords=" ".join(sepwords)				
				processedTestTweet = processTweet(sepwords)
				#sentiment extraction by TextBlob
				sentimental = TextBlob(processedTestTweet)
				sentiment = sentimental.sentiment.polarity
				if sentiment<0.0:
					sentiment='negative'
				elif sentiment>0.1:
					sentiment='positive'
				else: 
					sentiment='neutral'
				#nounPhrase extraction
				for nounPh in sentimental.noun_phrases:
					checkhash(countNounPhrase, nounPh)
				es.index(index='twitter', doc_type='elasticsearch', id=j, body=json.loads(json.dumps(item).encode("utf-8")))
				es.update(index="twitter", doc_type='elasticsearch', id=j, body={"doc": {"sentiment": sentiment }})
				j=j+1			
        return s
    #end
#end class



@app.route('/')
def home():
	return 	render_template('authenticate.html')
	

@app.route('/search', methods=['GET', 'POST'])
def index():
    #authenticate process begins for twitter api by saving in a config.json file
    if request.method == 'POST':
        savefile=open("config.json","a")
        savefile.write('{'+'\n')		
        savefile.write('"'+"consumer_key"+'"'+":"+'"'+request.form['consumerkey']+'"'+','+"\n")
        savefile.write('"'+"consumer_secret"+'"'+":"+'"'+request.form['consumersecret']+'"'+','+"\n")
        savefile.write('"'+"access_token"+'"'+":"+'"'+request.form['accesstoken']+'"'+','+"\n")	
        savefile.write('"'+"access_token_secret"+'"'+":"+'"'+request.form['accesssecret']+'"'+"\n")			
        savefile.write('}')
        savefile.close()		
        td = TwitterData()
        print("Connected")		
        maID=0
        i=0
        j=0
		#collects around 100 tweets with python
        while i<1:
               maID=td.getData('python',maID,j)
               j=j+100   
               i=i+1        
        #elasticsearch query
        resP = es.search(index="twitter", body={"query": {"match": {'sentiment':'positive'}}})	
        resNe = es.search(index="twitter", body={"query": {"match": {'sentiment':'negative'}}})	        
        resN = es.search(index="twitter", body={"query": {"match": {'sentiment':'neutral'}}})
        total = resP['hits']['total']+resNe['hits']['total']+resN['hits']['total']
        averagesenti=(1*resP['hits']['total']+(-1)*resNe['hits']['total'])/(float)(resP['hits']['total']+resNe['hits']['total']+resN['hits']['total'])	
        nounP=sorted(countNounPhrase.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]		
        print("template call")#template call
        return render_template('search.html',resP=resP, resNe=resNe, resN=resN, nounP=nounP,total=total,averagesenti=averagesenti)
		
@app.route('/searchtext', methods=['GET', 'POST'])
def searchtext():
#checks for post and perform elastic query
	if request.method == 'POST':
		print(request.form.get('positive'))
		if request.form.get('positive') == 'positive' :
			res = es.search(index="twitter", body={"size" : 20, "query": {"filtered": {
           "filter": {
           "bool": {
           "must": [
            {"term": {"text": request.form['searchtext']}},
            {"term": {"sentiment": "positive"}}
          ]
            }
            }
           }}})
		elif request.form.get('negative') == 'negative' :
			res = es.search(index="twitter", body={"size" : 20, "query": {"filtered": {
           "filter": {
           "bool": {
           "must": [
            {"term": {"text": request.form['searchtext']}},
            {"term": {"sentiment": "negative"}}
          ]
            }
            }
           }}})	
		elif request.form.getlist('neutral') == 'neutral' :
			res = es.search(index="twitter", body={"size" : 20, "query": {"filtered": {
           "filter": {
           "bool": {
           "must": [
            {"term": {"text": request.form['searchtext']}},
            {"term": {"sentiment": "neutral"}}
          ]
            }
            }
           }}})		  
		else:
			res = es.search(index="twitter", body={"size" : 20, "query": {"match": {'text':request.form['searchtext']}}})
		searchtext=request.form['searchtext']
		return render_template('searchresult.html',res=res, searchtext=searchtext)

if __name__ == '__main__':
    app.run()
