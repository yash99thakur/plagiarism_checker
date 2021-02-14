from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import re
import math
from requests import get 
import urllib
from bs4 import BeautifulSoup 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#nltk.download('stopwords') if it is first time uncomment it.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
app = Flask(__name__)
stopwords=stopwords.words('english')


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/new' )
def new():
    return render_template('index.html')

@app.route('/cosineSimilarity',methods=['POST'])

def cosineSimilarity():
    universalSetOfUniqueWords = []
    matchPercentage = 0

    ####################################################################################################
    inputQuery = request.form.get('plagtext')

    lowercaseQuery = inputQuery.lower()
    display1=lowercaseQuery.split(".")
    queryWordList = re.sub("[^\w]", " ",lowercaseQuery).split()			#Replace punctuation by space and split
	# queryWordList = map(str, queryWordList)					#This was causing divide by zero error
    for word in queryWordList:
        if word not in universalSetOfUniqueWords:
            universalSetOfUniqueWords.append(word)

	####################################################################################################
    database1=request.form.get('orgtext')
    yash=""
    # print(universalSetOfUniqueWords[0:20])
    # print(queryWordList[0:20])
    if(not database1):
            keyword = inputQuery
            url = "https://google.com/search?q="+keyword
            html = get(url).text

            
            soup = BeautifulSoup(html,features="lxml")
            temp="""
            class="kCrYT"
            """            
            links_with_text = []
          
            anurag=[]
            links_with_text=soup.find_all('a')
            for links in links_with_text:
                if 'href' in links.attrs:
                    anurag.append((links.attrs['href']))
                    # print(str(links.attrs['href'])+"\n")
            anurag_temp=[]
            for i in anurag:
                if "/url?q=" in i:
                    anurag_temp.append(i)  
            plag_result_links=[]
            for i in anurag_temp:
                first=i.find("q")
                last=i.find("&")
                plag_result_links.append(i[first:last])
            yash=plag_result_links[0]
            sarkari=plag_result_links[1]
            print(yash[2:])
            print(sarkari[2:])
            

            

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()    # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            database1 = '\n'.join(chunk for chunk in chunks if chunk)

    database1 = database1.lower()
    display2=database1.split(".")



    # print(database1)
    # print(queryWordList)
    databaseWordList = re.sub("[^\w]", " ",database1).split()	#Replace punctuation by space and split
	# databaseWordList = map(str, databaseWordList)			#And this also leads to divide by zero error

    for word in databaseWordList:
	    if word not in universalSetOfUniqueWords:
		    universalSetOfUniqueWords.append(word)

	####################################################################################################

    queryTF = []
    databaseTF = []
    # print(universalSetOfUniqueWords)
    # print(queryWordList)

    for word in universalSetOfUniqueWords:
	    queryTfCounter = 0
	    databaseTfCounter = 0

	    for word2 in queryWordList:
		    if word == word2:
			    queryTfCounter += 1
	    queryTF.append(queryTfCounter)

	    for word2 in databaseWordList:
		    if word == word2:
			    databaseTfCounter += 1
	    databaseTF.append(databaseTfCounter)


    # print(queryTF)
    # print(databaseTF)
    




    # ls1=[]
    # for i in universalSetOfUniqueWords:

    #     word_count = universalSetOfUniqueWords.count(i)  # Pythons count function, count()
    #     ls1.append((i,word_count))       


    # dict_1 = dict(ls1)


    # ls2=[]
    # for i in queryWordList:

    #     word_count = queryWordList.count(i)  # Pythons count function, count()
    #     ls2.append((i,word_count))       


    # dict_2 = dict(ls2)

    # display12=[]
    # for i in range(len(display1)):
    #     if display1[i] in display2:
    #         display12.append(display1[i])
    #     if (len(display2)-1)==i and len(display12)<len(display2):
    #         display12.extend(display1[i:len(display2)])
    # check1=len(display12)
    # check2=len(display2)
    # if check1!=check2:
    #     display12=display12[0:len(display2)]

    

    # def clean_string(text):
    #     text=''.join([word for word in text if word not in string.punctuation])
    #     text=text.lower()
    #     text=' '.join([word for word in text.split() if word not in stopwords])
    #     return text
    # display13=list(map(clean_string,display12))
    # display23=list(map(clean_string,display2))    

    # cv = CountVectorizer(analyzer = 'word', max_features = 5000, lowercase=True, preprocessor=None, tokenizer=None, stop_words = 'english')  
    # vectors = cv.fit_transform(display13)
    # kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
    # kmean_indices = kmeans.fit_predict(vectors)

    # pca = PCA(n_components=2)
    # scatter_plot_points = pca.fit_transform(vectors.toarray())

    # colors = ["r", "b", "c", "y", "m" ]

    # x_axis = [o[0] for o in scatter_plot_points]
    # y_axis = [o[1] for o in scatter_plot_points]
    # fig, ax = plt.subplots(figsize=(20,10))

    # ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

    

    # for i, txt in enumerate(display13):
    #     ax.annotate(txt, (x_axis[i], y_axis[i]))
    # plt.savefig("display1.png")

    # cv = CountVectorizer(analyzer = 'word', max_features = 5000, lowercase=True, preprocessor=None, tokenizer=None, stop_words = 'english')  
    # vectors = cv.fit_transform(display23)
    # kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
    # kmean_indices = kmeans.fit_predict(vectors)

    # pca = PCA(n_components=2)
    # scatter_plot_points = pca.fit_transform(vectors.toarray())

    # colors = ["r", "b", "c", "y", "m" ]

    # x_axis = [o[0] for o in scatter_plot_points]
    # y_axis = [o[1] for o in scatter_plot_points]
    # fig, ax = plt.subplots(figsize=(20,10))

    # ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

    # for i, txt in enumerate(display23):
    #     ax.annotate(txt, (x_axis[i], y_axis[i]))
    # plt.savefig("display2.png")
    # print(len(display12))
    # print(len(display2))
    # print(display12)
    # print(display2)
    

    # # # print(plt.show())
    
    # plt.show()

    # print(plt.show())

    dotProduct = 0
    for i in range (len(queryTF)):
	    dotProduct += queryTF[i]*databaseTF[i]

    queryVectorMagnitude = 0
    for i in range (len(queryTF)):
	    queryVectorMagnitude += queryTF[i]**2
    queryVectorMagnitude = math.sqrt(queryVectorMagnitude)

    databaseVectorMagnitude = 0
    for i in range (len(databaseTF)):
	    databaseVectorMagnitude += databaseTF[i]**2
    databaseVectorMagnitude = math.sqrt(databaseVectorMagnitude)

    matchPercentage = (float)(dotProduct / (queryVectorMagnitude * databaseVectorMagnitude))*100

    output = matchPercentage
    output=round(output,2)
  
   
    

    

    if (yash!=""):
            return render_template('index2.html', plag_meter=' {}%'.format(output) ,plag_meter1=' {}%'.format(round((100-output),2)), link1='{}'.format(yash[2:]),link2='{}'.format(sarkari[2:]));        
    else :   
            return render_template('index3.html', plag_meter=' {}%'.format(output),plag_meter1=' {}%'.format(round((100-output),2)) );    
    # plt.scatter(display1,Y2)

    # print(plt.show())
    # plt.scatter(display2,Y1)
    # print(plt.show())

    

if __name__ == "__main__":
    app.run(debug=True)