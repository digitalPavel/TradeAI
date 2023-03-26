import os
from pyexpat import features
import nltk
import numpy as np
import pandas as pd
import requests 
import json
import unicodedata
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import TfidfModel, LdaModel
from collections import Counter
import seaborn as sns
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

#Url
url = 'https://api.sec-api.io?token=74fe4b923f976ef65d8ce52c7cbfeede8eac9cd63db892be8b0198e90995cd49'

#Amount of documents
inputSize = 5

#Getting URLs of the documents
payload = {
"query": {
    "query_string": {
        "query": "ticker:AMZN AND formType:\"10-K\""
    }
},
"from": "0",
"size": "inputSize",
"sort": [{ "filedAt": { "order": "desc" } }]
}
headers = {
'Content-Type': 'application/json'
}

#Verify and/or create if the documents folder is exist
folder_name = 'documents'
project_dir = os.getcwd()
folder_path = os.path.join(project_dir, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#Verify to avoid unnecessary downloads of the documents
file_list = os.listdir(folder_path)
num_files = len(file_list)
if inputSize != num_files:
    #make requests
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    json_response = response.json()
    #https://www.sec.gov/Archives/edgar/data/1018724/000101872420000010/amzn-20200331x10q.htm
    matching_items = []

    #extract links with description "10-K"
    for item in json_response['filings']:
        link = item['linkToFilingDetails']
        matching_items.append(link)

    print(matching_items)

    #Download the documents
    headers1 = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    # Create a new folder for the documents
    #folder_name = 'documents'
    #project_dir = os.getcwd()
    #folder_path = os.path.join(project_dir, folder_name)

    #if not os.path.exists(folder_path):
        #os.makedirs(folder_path)

    i =0
    for url in matching_items:
        i+=1
        response =  requests.get(url, headers=headers1)
        file_path = os.path.join(folder_path, f'file{i}.html')
        with open(file_path, 'a') as file:
            file.write(response.content.decode('utf-8'))

###############################################################################################################
#Data Cleaning
   
def DelTags(file_soup):  
    # Remove HTML tags 
    doc = file_soup.get_text()
    
    # Remove newline characters
    doc = doc.replace('\n', ' ')
    
    # Replace unicode characters with their "normal" representations
    doc = unicodedata.normalize('NFKD', doc)
    
    return doc

def DelTables(file_soup):
    def GetDigitPercentage(tablestring):
        if len(tablestring)>0.0:
            numbers = sum([char.isdigit() for char in tablestring])
            length = len(tablestring)
            return numbers/length
        else:
            return 1
    
    # Evaluates numerical character % for each table
    # and removes the table if the percentage is > 15%
    #[x.extract() for x in file_soup.find_all('table') if GetDigitPercentage(x.get_text())>0.15]
    for table in file_soup.find_all('table'):
        if GetDigitPercentage(table.get_text()) > 0.15:
            table.extract()

    return file_soup

def ConvertHTML():
    # Remove al the following such as newlines, unicode text, XBRL tables, numerical tables and HTML tags,    
    # Make a new directory with all the .txt files called "textonly"
    try:
        folder_name = 'textonly'
        project_dir = os.getcwd()
        folder_path1 = os.path.join(project_dir, folder_name)
        if not os.path.exists(folder_path1):
           os.makedirs(folder_path1)
    except OSError:
        pass
    
    file_list = []
    # List of file in that directory
    for document in os.listdir(folder_path):
          file_list.append(document)
    
    # Iterate over scraped files and clean
    for filename in file_list:
            
        # Check if file has already been cleaned
        new_filename = filename.replace('.html', '.txt')
        text_file_list = os.listdir('textonly')
        if new_filename in text_file_list:
            continue
        
        # If it hasn't been cleaned already, keep going...       
        # Clean file
        with open(folder_path + '/' + filename, 'r') as file:
            parsed = True
            soup = BeautifulSoup(file.read(), features="xml")
            soup = DelTables(soup)
            text = DelTags(soup)
            with open('textonly/'+new_filename, 'w', encoding="utf-8") as newfile:
                newfile.write(text)
    
    # If all files in the CIK directory have been parsed
    # then log that
    #??
    #if parsed==False:
    #    print("Already parsed CIK", document)
    os.chdir('..')
    return
#####################################################################################################
#Data Preprocessing

#String Lemmatization
def lemmetize(words):
    lemmatized_words = [WordNetLemmatizer().lemmatize(w.lower()) for w in words]
    return lemmatized_words

#Removing StopWords and Punctuations
stop_words = set(stopwords.words('english'))
fin_stop_words = ("million","including","billion","december","january")
stop_words.update(fin_stop_words)

# removing stop words, numbers , removing punctuations and special characters and spaces
def remove_stopwords(words):
    filtered = [re.sub(r'[^\w\s]','',w) for w in words if not re.sub(r'[^\w\s]','',w) in stop_words and  not re.sub(r'[^\w\s]','',w).isnumeric() and not re.search('^\s*[0-9]',re.sub(r'[^\w\s]','',w)) and len(re.sub(r'[^\w\s]','',w)) > 3  ]
    return filtered

#Stemming Words
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
   
#ps = PorterStemmer() 
ps = SnowballStemmer("english") 

def stem_words(words):
    stemmed = [ps.stem(w) for w in words]
    return stemmed

#String Tokenization for each 10K document

from nltk.tokenize import sent_tokenize,word_tokenize

wordcount={} #dictionary to count the sentences and tokens in each 10-K document
docs = {} #dictionary to save the tokens after preprocessing for each 10-K document
file_doc=[] # Saving only the first document for the scope of this project
#setting directory to the .txt files folder

ConvertHTML()
os.listdir()
os.chdir("C:\\Users\\gaevs\\PycharmProjects\\TradeAI\\textonly")
    
#listing files in directory
files = [j for j in os.listdir()]
files.sort(reverse=True)
    
#iterating over each 10-K file
for file in files:
        
    text = open(file,"r", encoding="utf-8").read()
    #using sentence 
    sents = sent_tokenize(text)
    file_doc = sents
               
    tokens = word_tokenize(text.lower())
    partial_lem = lemmetize(tokens)
    after_stopwords = remove_stopwords(partial_lem)
        
    docs[file] = after_stopwords
    
    counts = {}
    counts["tokens"]=len(tokens)
    counts["sentences"]= len(sents)
    wordcount[file] = counts
    continue #Looping over just one document for now-

###########################################################################################
#Exploratory Data Analysis(EDA) on the latest 10-K Document

#TokenID using Genism
dataset = [lemmetize(remove_stopwords(d.lower().split()))for d in file_doc]

dictionary = corpora.Dictionary(dataset)

#Creating BoW
corpus = [dictionary.doc2bow(file) for file in dataset]

#TF-IDF
tfidf= TfidfModel(corpus) 
#Printing words with more than .5 Tf-IDF score, these are words that occur rarely in the document(Might have a higher meaning)
for doc in tfidf[corpus]:
    for id, freq in doc:
        if np.around(freq,decimals=2)> .5:
            print([dictionary[id], np.around(freq, decimals=2) ])



#WordCloud for the given 10K document
#The wordCloud helps us visualize some of the most frequently used words in the document

wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white", width=800, height=400).generate(" ".join(docs[list(docs.keys())[0]])) #first documents tokens from docs(which contains many tokens from different docs)

# Display the generated image:
plt.figure( figsize=(20,10), facecolor='k' )
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#LDA Modelling
lda_model = LdaModel(corpus=corpus, num_topics=6, id2word=dictionary, passes=10)

topics = lda_model.show_topics()
for topic in topics:
    print(topic)

#pyLDAvis
# import pyLDAvis.gensim
# from pyLDAvis import prepare
# vis = pyLDAvis.prepare(lda_model, corpus, dictionary)
# pyLDAvis.display(vis)

#Top 20 used words in the 10K report

# Get tokens for first document tokens from docs(which contains many tokens from different docs)
doc_tokens = docs[list(docs.keys())[0]]

# Count token frequencies
counter = Counter(doc_tokens)
most_common = counter.most_common(20)

# Extract x and y data for plot
x = [word for word, count in most_common]
y = [count for word, count in most_common]

# Create plot
plt.figure(figsize=(16, 6))
sns.barplot(x=y, y=x)
plt.title("Most Common Words in First Document")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

###########################################################################################

#Positivity Score of the 10K document

#Finding the sentiment value of the document using the Texblob library

doc_text = " ".join(docs[list(docs.keys())[0]]) #first documents tokens from docs(which contains many tokens from different docs)

sentiment = TextBlob(doc_text)

polarity = sentiment.sentiment.polarity
subjectivity = sentiment.subjectivity

print("polarity={}, subjectivity={}".format(polarity, subjectivity))
def polarity(text):
    return TextBlob(" ".join(text)).sentiment.polarity

sentences = pd.DataFrame(columns=["sentences", "polarity_score"])

for sent in dataset:
    row = {'sentences': sent, 'polarity_score': polarity(sent)}
    sentences = pd.concat([sentences, pd.DataFrame(row)], ignore_index=True)

sentences['polarity_score'].hist(figsize=(10, 8))

plt.title('Histogram of Polarity Scores')
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.show()

## Displaying the sentences that had a polarity score of over .5

print(sentences[sentences['polarity_score']>.5])
