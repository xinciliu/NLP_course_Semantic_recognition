import csv
import re
def preprocessing(file):
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column0 = [row[0]for row in reader][1:]
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[1]for row in reader][1:]
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column2 = [row[2]for row in reader][1:]    
    word_original=[]
    word_original_no_s=[]
    for x in column0:
        n=x.split(' ')
        word_original_no_s.append(x.split(' '))
        n.insert(0,'<s>')
        word_original.append(n)    
    type_original=[]
    for a in column1:
        b=a[1:len(a)-1]
        c=b.split(', ')
        n=[]
        for x in c:
            n.append(x[1:len(x)-1])
        type_original.append(n)   
    label_original=[]
    for x in column2:
        a=re.findall(r"\d+\.?\d*",x)
        label_original.append(a)   

    word_label={}
    a=0
    while a<len(word_original_no_s):
        word=word_original_no_s[a]
        type_=label_original[a]
        b=0
        while b<len(word):
            if word[b]+'|'+type_[b] not in word_label:
                word_label[word[b]+'|'+type_[b]]=1
            else:
                c=word_label[word[b]+'|'+type_[b]]
                word_label[word[b]+'|'+type_[b]]=c+1
            b=b+1
        a=a+1    
    label_add_s=[]
    for x in label_original:
        orig=[]
        orig.append('<s>')
        orig=orig+x
        label_add_s.append(orig)
    label_trans_dic={}
    for y in label_add_s:
        z=0
        while z<len(y)-1:
            a=y[z+1]+'|'+y[z]
            if a not in label_trans_dic:
                label_trans_dic[a]=0
            else:
                c=label_trans_dic[a]
                label_trans_dic[a]=c+1
            z=z+1
    label_trans_dic_pro={}
    total_num=0
    for x in label_trans_dic:
        total_num=total_num+label_trans_dic[x]
    for n in label_trans_dic:
        label_trans_dic_pro[n]=label_trans_dic[n]/total_num
    label_dic_s={}
    for x in label_add_s:
        for y in x:
            if y not in label_dic_s:
                label_dic_s[y]=1
            else:
                c=label_dic_s[y]
                label_dic_s[y]=c+1
    pro_label_dic_s={}
    num_label=0
    for x in label_dic_s:
        num_label=num_label+label_dic_s[x]
    for y in label_dic_s:
        pro_label_dic_s[y]=label_dic_s[y]/num_label
    print(pro_label_dic_s)    
    print(label_trans_dic_pro)
    pro_word_label_type={}
    for x in label_trans_dic_pro:
        s=x.split('|')
        ori=s[1]
        pro_word_label_type[x]=label_trans_dic_pro[x]/pro_label_dic_s[ori]          
    pro_word_label={}
    num_w_l=0
    for x in word_label:
        num_w_l=num_w_l+word_label[x]
    for y in word_label:
        pro_word_label[y]=word_label[x]/num_w_l
    ###pro_word_label=P(wi|ti)
    ###pro_word_label_transition=p(wi|ti)/p(ti)
    label_dic={}
    for x in label_original:
        for y in x:
            if y not in label_dic:
                label_dic[y]=1
            else:
                c=label_dic[y]
                label_dic[y]=c+1
    pro_label_dic={} 
    num_label=0
    for x in label_dic:
        num_label=num_label+label_dic[x]
    for y in label_dic:
        pro_label_dic[y]=label_dic[x]/num_label
    pro_word_label_transition={}
    for x in pro_word_label:
        s=x.split('|')
        ori=s[1]
        pro_word_label_transition[x]=pro_word_label[x]/pro_label_dic[ori]  
    i=0
    label_type_list=[]
    while i<len(label_original):
        label=label_original[i]
        type_=type_original[i]
        j=0
        n=[]
        while j<len(label):
            label_j=label[j]
            type_j=type_[j]
            label_type=label_j+'_'+type_j
            n.append(label_type)
            j=j+1
        label_type_list.append(n)    
        i=i+1    
    label_type_list_add_s=[]
    list_label=[]
    for x in label_type_list:
        for y in x:
            if y not in list_label:
                list_label.append(y)
    for x in label_type_list:
        x.insert(0,'<s>')
        label_type_list_add_s.append(x)    
    new_dic={}
    for x in label_type_list_add_s:
        i=0
        while i<len(x)-1:
            hh=x[i+1]+'|'+x[i]
            if hh in new_dic:
                k=new_dic[hh]
                new_dic[hh]=k+1
            else:
                new_dic[hh]=1
            i=i+1 
    new_single_dic={}
    for y in label_type_list_add_s:
        for z in y:
            if z in new_single_dic:
                kk=new_single_dic[z]
                new_single_dic[z]=kk+1
            else:
                new_single_dic[z]=1
    num=0
    new_probability_dic={}
    for n in new_dic:
        num=num+new_dic[n]
    for z in new_dic:  
        new_probability_dic[z]=new_dic[z]/num     
    num_i=0
    new_single_probability_dic={}
    for n in new_single_dic:
        num_i=num_i+new_single_dic[n]
    for z in new_single_dic:  
        new_single_probability_dic[z]=new_single_dic[z]/num_i    
    final_tj_dic={}
    for x in new_probability_dic:
        s=x.split('|')
        ori=s[1]
        final_tj_dic[x]=new_probability_dic[x]/new_single_probability_dic[ori]    
    word_type_dic={}
    for x in word_original:
        i=0
        while i<len(x):
            f=label_type_list_add_s[i]
            k=word_original[i]
            j=1
            while j<len(f):
                f_k=k[j]+'|'+f[j]
                if f_k in word_type_dic:
                    n=word_type_dic[f_k]
                    word_type_dic[f_k]=n+1
                else:
                    word_type_dic[f_k]=1
                j=j+1    
            i=i+1  
    word_type_dic_pro={}
    num_w=0
    for x in word_type_dic:
        num_w=num_w+word_type_dic[x]
    for y in word_type_dic:
        word_type_dic_pro[y]=word_type_dic[y]/num_w        
    pro_single_no_s={}
    num_pro=0
    for x in new_single_dic:
        if x!='<s>':
            num_pro=num_pro+new_single_dic[x]
    for y in new_single_dic:
        if y!='<s>':
            pro_single_no_s[y]=new_single_dic[y]/num_pro   
    final_tj_wi_dic={}
    for x in word_type_dic_pro:
        s=x.split('|')
        ori=s[1]
        final_tj_wi_dic[x]=word_type_dic_pro[x]/pro_single_no_s[ori]  
        
    ranked_lis= sorted(final_tj_wi_dic.items(),key=lambda item:item[1])  
    unknown_pro=ranked_lis[0][1]
    ranked_lis_1= sorted(final_tj_dic.items(),key=lambda item:item[1])  
    unknown_tran_pro=ranked_lis_1[0][1]
    return (list_label,final_tj_dic,final_tj_wi_dic,unknown_pro,unknown_tran_pro,pro_word_label_transition,pro_word_label_type)  
            
file='/Users/xinciliu/Desktop/P2_release 2/data_release/train.csv'
d=preprocessing(file)

###list_label means all label_type, eg ['0_VERB','0_ADV','0_PART'....]
###final_tj_dic means transition dictionary eg ['0_VERB|<s>': 0.08036264875432632,'0_ADV|0_VERB': 0.1518684204179868,'0_VERB|0_ADV': 0.2674160791732948]
###final_tj_wi_dic means (wi|label_type) eg [{'Ca|0_VERB': 0.018000109977754072,"n't|0_ADV": 0.17806708111698233...]
###unknown_pro means minimize pro(wi|label_type) 
###unknown_tran_pro means minimize transition probability 
###pro_word_label_transition means the transition dic for labels eg({'0|<s>': 0.04998456472525211,'0|0': 0.7471873499348288...]
###pro_word_label_type means(wi|label) eg[{'Ca|0': 8.57471146095934e-06, "n't|0": 8.57471146095934e-06...]
###
