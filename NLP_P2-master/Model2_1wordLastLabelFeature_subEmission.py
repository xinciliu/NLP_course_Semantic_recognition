import os
import io
import csv
import re
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from collections import defaultdict

def preprocessing(file):
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column0 = [row[0] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[1] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column2 = [row[2] for row in reader][1:]

    word_original = []
    word_original_no_s = []
    for x in column0:
        n = x.split(' ')
        word_original_no_s.append(x.split(' '))
        word_original.append(n)

    label_original = []
    for x in column2:
        a = re.findall(r"\d+\.?\d*", x)
        label_original.append(a)

    words_num = defaultdict(int)
    pair_num = defaultdict(int)
    for review in label_original:
        for i in range(len(review)):
            words_num[review[i]] += 1
            if i == 0:
                words_num["<s>"] += 1
                pair_num["<s>_"+review[i]] += 1
            else:
                pair_num[review[i-1] + "_" + review[i]] += 1

    pair_pro = defaultdict(float)
    for pair, num in pair_num.items():
        words = pair.split('_')
        first_word = words[0]
        pair_pro[words[1]+"|"+words[0]] = num/words_num[first_word]

    label_sum = defaultdict(int)
    word_label = defaultdict(int)
    for index, labels in enumerate(label_original):
        for i in range(len(labels)):
            label_sum[labels[i]] += 1
            word_label[word_original[index][i]+ "_" + labels[i]] += 1

    word_pro = defaultdict(float)
    for word_label_pair, num in word_label.items():
        index = word_label_pair.rfind('_')
        word = word_label_pair[:index]
        label = word_label_pair[(index+1):]
        if label_sum[label] == 0:
            print(label)
        word_pro[word+"|"+label] = num/label_sum[label]

    return word_pro, pair_pro


def feature_x(file, preSetLabel):
    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column0 = [row[0] for row in reader][1:]

    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column1 = [row[1] for row in reader][1:]

    with open(file, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        column2 = [row[2] for row in reader][1:]

    pos_window =[]
    count=-1 # count the row number
    for line in column0:
        count=count+1
        tag=column1[count] # to get corresponding tag_seq
        label=column2[count]
        word=line.split()
        taglist=tag.strip('[]').split(',')
        labellist = label.strip('[]').split(',')
        for i in range(len(word)):
            if len(word)==1:  #if only one word exist
                window = word[i]
                tag = taglist[i]
                if preSetLabel:
                    label = preSetLabel
                else:
                    label = "0"
            elif len(word)!=1 and i==0:  #if it is the first word
                window = word[i]
                tag = taglist[i]
                if preSetLabel:
                    label = preSetLabel
                else:
                    label = "0"
            else:
                if i==len(word)-1:
                    window = word[i]
                    tag = taglist[i]
                    if preSetLabel:
                        label = preSetLabel
                    else:
                        label = labellist[i-1]
                else:
                    window = word[i]
                    tag = taglist[i]
                    if preSetLabel:
                        label = preSetLabel
                    else:
                        label = labellist[i-1]
            pos_window.append({'word': window,'pos':tag, 'tag-1':label})
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


def hmm(transitionDict, emisssionDict, labels, test):
    l = len(labels)

    score = [[None] * len(test) for i in range(l)]
    bptr = [[None] * len(test) for i in range(l)]

    for i in range(l):
        score[i][0] = np.log(transitionDict[labels[i]+"|<s>"]) + np.log(emisssionDict[labels[i]+"|("+test[0]+",0)"])
        bptr[i][0] = 0

    for t in range(1, len(test)):
        for i in range(l):
            maximum = float("-inf")
            for j in range(l):
                temp = score[j][t-1] + np.log(transitionDict[labels[i]+"|"+labels[j]]) + \
                    np.log(emisssionDict[labels[i]+"|("+test[t]+","+labels[j]+")"])
                if temp > maximum:
                    maximum = temp
                    score[i][t] = temp
                    bptr[i][t] = j

    results = [None] * len(test)
    results[-1] = np.argmax([score[i][-1] for i in range(l)])
    for i in range(len(test)-2, -1, -1):
        results[i] = bptr[results[i+1]][i+1]
    return results

if __name__ == "__main__":
    # validation verification
    trainfile = './data_release/train.csv'
    x = feature_x(trainfile, None)
    y = label_y(trainfile)
    trainy = np.array(y)

    valfile = './data_release/val.csv'
    tx0 = feature_x(valfile, "0")
    tx1 = feature_x(valfile, "1")
    vec = DictVectorizer()
    transform = vec.fit_transform(x + tx0 + tx1)
    trainx = transform[:len(x)]
    testx0 = transform[len(x):len(x+tx0)]
    testx1 = transform[len(x+tx0):]

    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(trainx, trainy)
    prediction_prabability_validation0 = classifier.predict_proba(testx0)
    prediction_prabability_validation1 = classifier.predict_proba(testx1)

    train_file = './data_release/train.csv'
    _, label_trans_dic_pro = preprocessing(train_file)

    test_file = './data_release/val.csv'
    with open(test_file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column0 = [row[0] for row in reader][1:]
    test_word_original = []
    for x in column0:
        n = x.split(' ')
        test_word_original.append(n)

    with open('test_model2_1wordLastLabelFeature_validation_subEmission.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['idx', 'label'])

        i = 1
        counterPredictionPro = 0
        for test in test_word_original:
            label_word_pro = defaultdict(float)
            for j in range(len(test)):
                word = test[j]
                label_word_pro["0|("+word+",0)"] = prediction_prabability_validation0[counterPredictionPro][0]
                label_word_pro["1|("+word+",0)"] = prediction_prabability_validation0[counterPredictionPro][1]
                label_word_pro["0|("+word+",1)"] = prediction_prabability_validation1[counterPredictionPro][0]
                label_word_pro["1|("+word+",1)"] = prediction_prabability_validation1[counterPredictionPro][1]
                counterPredictionPro += 1
            result = hmm(label_trans_dic_pro, label_word_pro, ['0', '1'], test)
            for label in result:
                report_writer.writerow([str(i), str(label)])
                i += 1

