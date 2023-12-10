from audioop import avg
from genericpath import isdir
import pandas as pd
import numpy as np
import os
import textstat
import string
data = os.listdir("./")

listofmodels = [
    "twmkn9/albert-base-v2-squad2",
    "valhalla/bart-large-finetuned-squadv1",
    "deepset/bert-base-cased-squad2",
    "google/bigbird-roberta-base",
    "google/bigbird-pegasus-large-arxiv",
    "dmis-lab/biobert-v1.1",
    "deepset/roberta-base-squad2",
    "Splend1dchan/canine-c-squad",
    "YituTech/conv-bert-base",
    "Palak/microsoft_deberta-large_squad",
    "microsoft/deberta-v2-xlarge",
    "distilbert-base-uncased",
    "bhadresh-savani/electra-base-squad2",
    "nghuyong/ernie-1.0-base-zh",
    "xlm-mlm-en-2048",
    "google/fnet-base",
    "funnel-transformer/small",
    "EleutherAI/gpt-neo-1.3B",
        "hf-internal-testing/tiny-random-gptj",
    "gpt2",
    "kssteven/ibert-roberta-base",
    "allenai/led-base-16384",
    "allenai/longformer-large-4096-finetuned-triviaqa",
    "facebook/mbart-large-cc25",
    "mnaylor/mega-base-wikitext",
    "csarron/mobilebert-uncased-squad-v2",
    "microsoft/mpnet-base",
    "google/mt5-small",
    "RUCAIBox/mvp",
    "sijunhe/nezha-cn-base",
    "uw-madison/nystromformer-512",
    "facebook/opt-350m",
    "bert-base-uncased",
    "google/rembert",
    "roberta-base",
    "andreasmadsen/efficient_mlm_m0.40",
    "ArthurZ/dummy-rocbert-qa",
    "tau/splinter-base",
    "squeezebert/squeezebert-uncased",
    "t5-small",
    "xlm-mlm-en-2048",
    "xlnet-base-cased",
    "uw-madison/yoso-4096",
    "SRDdev/QABERT-small",
"bert-large-uncased-whole-word-masking-finetuned-squad",
"facebook/bart-large-cnn",
"ahotrod/electra_large_discriminator_squad2_512",
]
translator = str.maketrans('', '', string.punctuation)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stop_words(sentence):
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)

    # Remove stop words
    
    return result
def cleandata(data):
    data = [s.strip() for s in data]
    data = [s.translate(translator) for s in data]
    data = [s.lower() for s in data]
    newdata = []
    stop_words = set(stopwords.words('english'))
    for d in data:
        filtered_words = [word for word in [d] if word.lower() not in stop_words]
        newdata.append(' '.join(filtered_words))

    return newdata

from fuzzywuzzy import fuzz

def getsimilarity(a,b):
    t = []
    for loop in range(0,len(a)):
        if b[loop]=="":
            t.append(0)
            continue
        if a[loop]==b[loop]:
            t.append(1)
            continue
        elif a[loop] in b[loop]:
            t.append(1)
            continue
        elif b[loop] in a[loop]:
            t.append(1)
            continue
        elif fuzz.ratio(a[loop], b[loop])>80:
            t.append(1)
            continue
        else:
            t.append(0)


        #print(a[loop],"-",b[loop],fuzz.ratio(a[loop], b[loop]))
    return t

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def getmemeasures(actual_labels,predicted_labels):
    # Calculate and print accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)
    #precision = precision_score(actual_labels, predicted_labels)
    #recall = recall_score(actual_labels, predicted_labels)
    #f1 = f1_score(actual_labels, predicted_labels)
    #conf_matrix = confusion_matrix(actual_labels, predicted_labels)
    #print(conf_matrix)
    #print(accuracy,precision,recall,f1,conf_matrix)
    return accuracy


averagecontextcount = []
averagequestionlength = []
averageanswerlength = []
averagepassagecomplexity = []
def count_words(text):
    return len(text.split())

def text_complexity(text):
    return textstat.flesch_kincaid_grade(text)

from matplotlib import pyplot as plt
plt.figure(figsize=(15, 15))
datasets = [
'atlas-math-sets',
'bioasq10b-factoid',
'biomedical_cpgQA',
'ielts',
'JournalQA',
'QuAC',
'ScienceQA',
'wikipedia']
import sys

for d in [sys.argv[1]]:
    if d=="wikipedia":
        data = d+os.sep+d+".csv"
        data = pd.read_csv(data,encoding="iso-8859-1")   
        print(data.head())
        temptotallength = 0
        tempcomplexity = []
        for c in data['context'].values:
            with open(d+os.sep+c, 'r',encoding="iso-8859-1") as file:
                file_contents = file.read()
                tempcomplexity.append(text_complexity(file_contents))

        data['Passage_Complexity'] = tempcomplexity   
        data['Passage_Complexity'] = data['Passage_Complexity'].round(0)        

        actual = data["answer"].values
        accuracy = []


        for model in range(0,len(listofmodels)):
            try:
                tempresults = pd.read_csv(d+os.sep+d+"_results"+os.sep+str(model)+".csv")
                predvalues = tempresults["Answer_"+str(model)].values
                actual = cleandata(actual)
                predvalues = cleandata(predvalues)  
                results = getsimilarity(actual,predvalues)
                data["Result_"+str(model)] = results
            except:
                data["Result_"+str(model)] = [0]*len(actual)
                pass


        resultss = pd.DataFrame()
        resultss["Models"] = listofmodels

        for c in sorted(data['Passage_Complexity'].unique()):
            yes = []
            no = []
            
            for model in range(0,len(listofmodels)):
                temp = data[data['Passage_Complexity']==c]
                #print(temp["Result_"+str(model)].count(0))
                yes.append((temp["Result_"+str(model)] == 1).sum()/len(temp["Result_"+str(model)]))
                no.append((temp["Result_"+str(model)] == 0).sum()/len(temp["Result_"+str(model)]))
            resultss[str(int(c))+"yes1"] = yes
            resultss[str(int(c))+"no0"] = no

        resultss["Accuracy"] = pd.read_csv(d+os.sep+d+"_results.csv")["Accuracy"].values
        
        temp = pd.read_csv("Accuracy_plot.csv")
        temp = temp[["Overall Performance","Models"]]
        resultss = pd.merge(resultss, temp, on='Models')
    

        resultss = resultss.sort_values(by='Overall Performance', ascending=False)
        #del 

        import seaborn as sns
        import pandas as pd
        from matplotlib import pyplot as plt
        


        resultss = resultss.set_index('Models')
        resultss.to_csv('plot8_'+d+".csv")
        #resultss = resultss[resultss["Accuracy"]>0.6]

        sns.heatmap(resultss.values, annot=True,linewidths=0.1,annot_kws={"size": 5},cmap='viridis',vmin=0, vmax=1)
        plt.xticks(ticks=range(len(resultss.columns.tolist())), labels=resultss.columns.tolist(),rotation=90,size=5)
        plt.yticks(ticks=range(len(resultss.index.tolist())), labels=resultss.index.tolist(),rotation=0,size=5)
        plt.tight_layout() 
        plt.savefig('plot8_'+d,dpi=1000)    
        continue
    
    if d=="JournalQA":
        data = d+os.sep+d+".csv"
        data = pd.read_csv(data,encoding="iso-8859-1")   
        print(data.head())
        temptotallength = 0
        tempcomplexity = []

        for c in data['context'].values:
            with open(d+os.sep+c.replace("/","\\").replace(".\\",""), 'r',encoding="iso-8859-1") as file:
                file_contents = file.read()
                
                tempcomplexity.append(text_complexity(file_contents))

        data['Passage_Complexity'] = tempcomplexity   
        data['Passage_Complexity'] = data['Passage_Complexity'].round(0)        

        actual = data["answer"].values
        accuracy = []


        for model in range(0,len(listofmodels)):
            try:
                tempresults = pd.read_csv(d+os.sep+d+"_results"+os.sep+str(model)+".csv")
                predvalues = tempresults["Answer_"+str(model)].values
                actual = cleandata(actual)
                predvalues = cleandata(predvalues)  
                results = getsimilarity(actual,predvalues)
                data["Result_"+str(model)] = results
            except:
                data["Result_"+str(model)] = [0]*len(actual)
                pass


        resultss = pd.DataFrame()
        resultss["Models"] = listofmodels

        for c in sorted(data['Passage_Complexity'].unique()):
            yes = []
            no = []
            
            for model in range(0,len(listofmodels)):
                temp = data[data['Passage_Complexity']==c]
                #print(temp["Result_"+str(model)].count(0))
                yes.append((temp["Result_"+str(model)] == 1).sum()/len(temp["Result_"+str(model)]))
                no.append((temp["Result_"+str(model)] == 0).sum()/len(temp["Result_"+str(model)]))
            resultss[str(int(c))+"yes1"] = yes
            resultss[str(int(c))+"no0"] = no

        resultss["Accuracy"] = pd.read_csv(d+os.sep+d+"_results.csv")["Accuracy"].values
        
        temp = pd.read_csv("Accuracy_plot.csv")
        temp = temp[["Overall Performance","Models"]]
        resultss = pd.merge(resultss, temp, on='Models')
    

        resultss = resultss.sort_values(by='Overall Performance', ascending=False)
        #del 

        import seaborn as sns
        import pandas as pd
        from matplotlib import pyplot as plt
        


        resultss = resultss.set_index('Models')
        resultss.to_csv('plot8_'+d+".csv")
        #resultss = resultss[resultss["Accuracy"]>0.6]

        sns.heatmap(resultss.values, annot=True,linewidths=0.1,annot_kws={"size": 5},cmap='viridis',vmin=0, vmax=1)
        plt.xticks(ticks=range(len(resultss.columns.tolist())), labels=resultss.columns.tolist(),rotation=90,size=5)
        plt.yticks(ticks=range(len(resultss.index.tolist())), labels=resultss.index.tolist(),rotation=0,size=5)
        plt.tight_layout() 
        plt.savefig('plot8_'+d,dpi=1000)    
        continue


    data = d+os.sep+d+".csv"
    data = pd.read_csv(data,encoding="iso-8859-1")
    data['Passage_Complexity'] = data['context'].apply(lambda x: text_complexity(x))
    data['Passage_Complexity'] = data['Passage_Complexity'].round(0)

    actual = data["answer"].values
    accuracy = []


    for model in range(0,len(listofmodels)):
        try:
            tempresults = pd.read_csv(d+os.sep+d+"_results"+os.sep+str(model)+".csv")
            predvalues = tempresults["Answer_"+str(model)].values
            actual = cleandata(actual)
            predvalues = cleandata(predvalues)  
            results = getsimilarity(actual,predvalues)
            data["Result_"+str(model)] = results
        except:
            data["Result_"+str(model)] = [0]*len(actual)
            pass


    resultss = pd.DataFrame()
    resultss["Models"] = listofmodels

    for c in sorted(data['Passage_Complexity'].unique()):
        yes = []
        no = []
        
        for model in range(0,len(listofmodels)):
            temp = data[data['Passage_Complexity']==c]
            #print(temp["Result_"+str(model)].count(0))
            yes.append((temp["Result_"+str(model)] == 1).sum()/len(temp["Result_"+str(model)]))
            no.append((temp["Result_"+str(model)] == 0).sum()/len(temp["Result_"+str(model)]))
        resultss[str(int(c))+"yes1"] = yes
        resultss[str(int(c))+"no0"] = no

    resultss["Accuracy"] = pd.read_csv(d+os.sep+d+"_results.csv")["Accuracy"].values
    
    temp = pd.read_csv("Accuracy_plot.csv")
    temp = temp[["Overall Performance","Models"]]
    resultss = pd.merge(resultss, temp, on='Models')
 

    resultss = resultss.sort_values(by='Overall Performance', ascending=False)
    #del 

    import seaborn as sns
    import pandas as pd
    
    


    resultss = resultss.set_index('Models')
    resultss.to_csv('plot8_'+d+".csv")
    #resultss = resultss[resultss["Accuracy"]>0.6]

    sns.heatmap(resultss.values, annot=True,linewidths=0.1,annot_kws={"size": 5},cmap='viridis',vmin=0, vmax=1)
    plt.xticks(ticks=range(len(resultss.columns.tolist())), labels=resultss.columns.tolist(),rotation=90,size=5)
    plt.yticks(ticks=range(len(resultss.index.tolist())), labels=resultss.index.tolist(),rotation=0,size=5)
    plt.tight_layout() 
    plt.savefig('plot8_'+d,dpi=1000)

    continue

