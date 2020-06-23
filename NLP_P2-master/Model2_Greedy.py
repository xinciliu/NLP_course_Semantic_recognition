import os
import io
import csv
import re
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB


def feature_x(file):
    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column0 = [row[0] for row in reader][1:]

    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column1 = [row[1] for row in reader][1:]

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
            elif len(word)!=1 and i==0:  #if it is the first word
                window = [word[i], word[i], word[i+1]]
                tag = [taglist[i], taglist[i], taglist[i+1]]
            else:
                if i==len(word)-1:
                    window=[word[i-1],word[i], word[i]]
                    tag = [taglist[i-1], taglist[i], taglist[i]]
                else:
                    window=[word[i-1], word[i], word[i+1]]
                    tag = [taglist[i-1], taglist[i], taglist[i+1]]
            pos_window.append({'word-1': window[0],'pos-1':tag[0], 'word': window[1],'pos':tag[1], 'word+1': window[2],'pos+1':tag[2]})
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

trainfile = './data_release/train.csv'
x=feature_x(trainfile)
y=label_y(trainfile)
trainy=np.array(y)

testfile='./data_release/test_no_label.csv'
valfile='./data_release/val.csv'
tx=feature_x(valfile)
vec= DictVectorizer()
transform=vec.fit_transform(x+tx)
trainx=transform[:len(x)]
testx=transform[len(x):]

classifier = MultinomialNB(alpha=1.0)
classifier.fit(trainx,trainy)
prediction = classifier.predict(testx)
prediction_prabability=classifier.predict_proba(testx)


index=0
with open('output.csv', 'w+') as csvfile:
    wtr = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    fieldnames = ['idx', 'label']
    wtr = csv.DictWriter(csvfile, fieldnames=fieldnames)
    wtr.writeheader()
    for x in prediction:
        index = index + 1
        wtr.writerow({'idx': index, 'label': x})


