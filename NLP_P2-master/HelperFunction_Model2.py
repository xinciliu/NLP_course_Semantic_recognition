import os
import csv
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
#import Model1

def feature1_x(file): #only look at word
    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column0 = [row[0] for row in reader][1:]

    pos_window =[]

    for line in column0:

          word=line.split()

          for i in range(len(word)):
            if len(word)==1:  #if only one word exist
                window = [word[i], word[i], word[i]]


            elif len(word)!=1 and i==0:  #if it is the first word
                window = [word[i], word[i], word[i+1]]


            else:
                if i==len(word)-1:
                    window=[word[i-1],word[i], word[i]]

                else:
                    window=[word[i-1], word[i], word[i+1]]

            pos_window.append({'word-1':window[0], 'word':window[1], 'word+1':window[2]})


    print(len(pos_window))
    return pos_window


def feature2_x(file):  #only look at tag
    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column0 = [row[0] for row in reader][1:]

    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column1 = [row[1] for row in reader][1:]

    window=[]
    pos_window =[]
    count=-1 # count the row number
    for line in column0:
          count=count+1
          label=column1[count] # to get corresponding tag_seq
          word=line.split()
          taglist=label.strip('[]').split(',')
          for i in range(len(word)):
            if len(word)==1:  #if only one word exist

                tag = [taglist[i], taglist[i], taglist[i]]

            elif len(word)!=1 and i==0:  #if it is the first word

                tag = [taglist[i], taglist[i], taglist[i+1]]

            else:
                if i==len(word)-1:

                    tag = [taglist[i-1], taglist[i], taglist[i]]
                    #print(tag)
                else:

                    tag = [taglist[i-1], taglist[i], taglist[i+1]]
                    #print(tag)

            pos_window.append({'pos-1':tag[0], 'pos':tag[1], 'pos+1':tag[2]})

    print(len(pos_window))
    return pos_window




def feature3_x(file): #look at both tags and words
    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column0 = [row[0] for row in reader][1:]

    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column1 = [row[1] for row in reader][1:]

    big_listx = []
    window=[]
    tag=[]
    pos_window =[]
    count=-1 # count the row number
    for line in column0:
          count=count+1
          label=column1[count] # to get corresponding tag_seq
          word=line.split()
          taglist=label.strip('[]').split(',')
          for i in range(len(word)):
            if len(word)==1:  #if only one word exist
                window = [word[i], word[i], word[i]]
                tag = [taglist[i], taglist[i], taglist[i]]
                #print(tag)
            elif len(word)!=1 and i==0:  #if it is the first word
                window = [word[i], word[i], word[i+1]]
                tag = [taglist[i], taglist[i], taglist[i+1]]
                #print(tag)
            else:
                if i==len(word)-1:
                    window=[word[i-1],word[i], word[i]]
                    tag = [taglist[i-1], taglist[i], taglist[i]]
                    #print(tag)
                else:
                    window=[word[i-1], word[i], word[i+1]]
                    tag = [taglist[i-1], taglist[i], taglist[i+1]]
                    #print(tag)
            #pos_window.append({'word-1':window[0], 'word':window[1], 'word+1':window[2]})
            pos_window.append({'word-1': window[0],'pos-1':tag[0], 'word': window[1],'pos':tag[1], 'word+1': window[2],'pos+1':tag[2]})
            #big_listx.append(window)
    print(len(pos_window))
    return pos_window


def label_y(file):
    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column2 = [row[2] for row in reader][1:]
    big_listy = []
    intcolumn2 = [[int(i) for i in j[1:-1].split(",")] for j in column2]
    for line in intcolumn2:

       for i in range(len(line)):

         big_listy.append(line[i])
    return big_listy


def get_probability(trainfile, testfile, featuremode): #using NB classifier to do the training
    if featuremode==1:
        x=feature1_x(trainfile)
        tx = feature1_x(testfile)
    if featuremode==2:
        x=feature2_x(trainfile)
        tx = feature2_x(testfile)
    if featuremode==3:
        x=feature3_x(trainfile)
        tx = feature3_x(testfile)

    y=label_y(trainfile)
    trainy=np.array(y)

    vec= DictVectorizer()
    transform=vec.fit_transform(x+tx)
    trainx=transform[:len(x)]
    testx=transform[len(x):]

    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(trainx,trainy)
    prediction = classifier.predict(testx)

    prediction_prabability=classifier.predict_proba(testx)
    print(prediction_prabability)

    print(prediction_prabability[0][1])

    index = 0
    with open('output.csv', 'w+') as csvfile:
       wtr = csv.writer(csvfile, delimiter=',', lineterminator='\n')
       fieldnames = ['idx', 'label']
       wtr = csv.DictWriter(csvfile, fieldnames=fieldnames)
       wtr.writeheader()
       for x in prediction:
           index = index + 1
           wtr.writerow({'idx': index, 'label': x})
    return prediction_prabability

train = '/Users/zhaoxinglu/Desktop/P2_release/data_release/train.csv'
test='/Users/zhaoxinglu/Desktop/P2_release/data_release/test_no_label.csv'
val='/Users/zhaoxinglu/Desktop/P2_release/data_release/val.csv'
get_probability(train,val,2)




