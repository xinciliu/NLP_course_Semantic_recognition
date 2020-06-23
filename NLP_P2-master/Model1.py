from collections import defaultdict
import numpy as np
import csv
import re


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


def hmm(transitionDict, emisssionDict, unknown_pro, unknown_tran_pro, labels, test):
    # remember to remove <s> if <s> are in the labels
    # remember to change the dict key words
    l = len(labels)

    score = [[None] * len(test) for i in range(l)]
    bptr = [[None] * len(test) for i in range(l)]

    for i in range(l):
        if (labels[i] + "|<s>") not in transitionDict:
            transitionDict[labels[i] + "|<s>"] = unknown_tran_pro
        if (test[0]+"|"+labels[i]) not in emisssionDict:
            emisssionDict[test[0]+"|"+labels[i]] = unknown_pro
        score[i][0] = np.log(transitionDict[labels[i]+"|<s>"]) + np.log(emisssionDict[test[0]+"|"+labels[i]])
        bptr[i][0] = 0

    for t in range(1, len(test)):
        for i in range(l):
            maximum = float("-inf")
            for j in range(l):
                if (labels[i]+"|"+labels[j]) not in transitionDict:
                    transitionDict[labels[i] + "|" + labels[j]] = unknown_tran_pro
                if (test[t]+"|"+labels[i]) not in emisssionDict:
                    emisssionDict[test[t]+"|"+labels[i]] = unknown_pro
                temp = score[j][t-1] + np.log(transitionDict[labels[i]+"|"+labels[j]]) + np.log(emisssionDict[test[t]+"|"+labels[i]])
                if temp > maximum:
                    maximum =temp
                    score[i][t] = temp
                    bptr[i][t] = j

    results = [None] * len(test)
    results[-1] = np.argmax([score[i][-1] for i in range(l)])
    for i in range(len(test)-2, -1, -1):
        results[i] = bptr[results[i+1]][i+1]
    return results

if __name__ == "__main__":
    train_file = './data_release/train.csv'
    pro_word_label, label_trans_dic_pro = preprocessing(train_file)

    test_file = './data_release/val.csv'
    with open(test_file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column0 = [row[0] for row in reader][1:]
    test_word_original = []
    for x in column0:
        n = x.split(' ')
        test_word_original.append(n)

    with open('test_model1_validation.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['idx', 'label'])

        i = 1
        for test in test_word_original:
            result = hmm(label_trans_dic_pro, pro_word_label, 0.000000001, 0.000000001, ['0', '1'], test)
            for label in result:
                report_writer.writerow([str(i), str(label)])
                i += 1
