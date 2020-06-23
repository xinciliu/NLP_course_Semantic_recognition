import spacy
import csv

def preprocessing(file):
        with open(file,'r',encoding='ISO-8859-1') as csvfile:
             reader = csv.reader(csvfile)
             column0 = [row[0]for row in reader][1:]
        return  column0

def embedding_dic(file):
        lis=preprocessing(file)
        dic={}
        nlp = spacy.load('en_core_web_sm')
        for x in lis:
            doc=nlp(x)
            for y in doc:
                dic[str(y)]=y.vector
        return dic

###find the minimize for each feature in list
def minmize_dic(emd_dic):
    min_dic={}
    for x in range(96):
        min_dic[x]=0
    for x in emd_dic:
        lis=emd_dic[x]
        j=0
        while j<96:
            now=lis[j]
            now_min=min_dic[j]
            if now<now_min:
                min_dic[j]=now
            j=j+1
    return  min_dic     

###change all feature in embedding list to positve
def change_emmbeding_dic(emd_dic):
    mini_dic=minmize_dic(emd_dic)
    new_dic={}
    for x in emd_dic:
        lis=emd_dic[x]
        i=0
        while i<96:
            a=lis[i]
            if mini_dic[i]<0:
                lis[i]=a-mini_dic[i]
            else:
                lis[i]=lis[i]
            i=i+1
        new_dic[x]=lis
    return new_dic    


###to get all features, first call : emd_dic=embedding_dic(file)
###then: positive_embedding_dic = change_emmbeding_dic(emd_dic)
