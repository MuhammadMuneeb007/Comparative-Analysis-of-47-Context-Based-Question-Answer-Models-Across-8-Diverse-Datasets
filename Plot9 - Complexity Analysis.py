from symbol import pass_stmt
from unittest import result
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from pybtex.database.input import bibtex
import os
from collections import Counter
import pandas as pd
import numpy as np
import json
#from scholarly import scholarly
from tika import parser # pip install tika
from paraphraser import paraphrase
import os
import sys
import scihub
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
#import mpld3
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#from wordcloud import WordCloud
import sys
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
diseasename = ["alzheimer's"]
#import fitz

import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.enums import TA_LEFT
allanswers = []
allscores = []
translator = str.maketrans('', '', string.punctuation)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.enums import TA_LEFT
allanswers = []
allscores = []
translator = str.maketrans('', '', string.punctuation)
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def converttime(time_str):
    # Extract hours, minutes, and seconds using regular expressions
    print(time_str)
    match = re.match(r'(\d+)m([\d.]+)s', time_str.strip())
    if match:
        minutes, seconds = map(float, match.groups())
        # Convert to total hours
        total_hours =  minutes / 60 + seconds / 3600
        return total_hours
    else:
        raise ValueError("Invalid time format")

import seaborn as sns
import matplotlib.pyplot as plt 
def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment
def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 4
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor",size=5
        ) 




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

def alphanumeric_key(value):
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', value)]

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

import sys
import os
import subprocess
import re

data = os.listdir("./")
datasets = [
'atlas-math-sets',
'bioasq10b-factoid',
'biomedical_cpgQA',
'ielts',
'JournalQA',
'QuAC',
'ScienceQA',
'wikipedia']
rrrr = []
accur = []
timee = []
for a in datasets:
    #accur.append(a+"_Accuracy")
    accur.append(a)
    
    timee.append(a+"_Time")

results = pd.DataFrame()
for loop in datasets:
    data = pd.read_csv(loop+os.sep+loop+"_results.csv",index_col=0)
    data = data.sort_index()
    results[loop] = data["Accuracy"].values

    #results[loop+"_Accuracy"] = data["Accuracy"].values
    #try:
    #    results[loop+"_Time"] = data["Time in seconds"].values/(3600*24)
    #except:
    #    results[loop+"_Time"] = data["Time"].values/(3600*24)



results['Overall Performance'] = results[accur].sum(axis=1)
results['Overall Performance'] = results['Overall Performance']/len(datasets)
rrrr = results['Overall Performance'].values


results = pd.DataFrame()

def calculate_length(text):
    return len(text.split())


results = pd.DataFrame()
yes1 = []
yes2 = []
yes3 = []
yes4 = []
yes5 = []

no1 = []
no2 = []
no3 = []
no4 = []
no5 = []
datasets = [
'atlas-math-sets',
'bioasq10b-factoid',
'biomedical_cpgQA',
'ielts',
'JournalQA',
'QuAC',
'ScienceQA',
'wikipedia']

columns  = []
for c in range(5,18,2):
    columns.append(str(c)+'yes1')
    columns.append(str(c)+'no0')

results = pd.DataFrame()

for c in columns:
    count = 0
    array_list = []
    for d in datasets:
        data = pd.read_csv("plot8_"+d+".csv")
        try:
            array_list.append(data[c].values)
            count=count+1
        except:
            #frames.append(data)
            pass
    r = np.sum(array_list, axis=0)
    r = np.round(r / count, 2).astype(float)

    results["Complexity_"+str(c)] = r


frames = []
df = pd.DataFrame()
df["models"] = listofmodels
print(results.head())
print(results.columns)

results['models'] = pd.read_csv("plot8_"+datasets[0]+".csv")['Models'].values
results['Overall Performance'] = pd.read_csv("plot8_"+datasets[0]+".csv")['Overall Performance'].values 
results = results.set_index('models')
results =  results.head(47)

# Display the result
print(results)

sns.heatmap(results.values, annot=True,linewidths=0.1,annot_kws={"size": 4},cmap='viridis',vmin=results.values.min(), vmax=results.values.max())
plt.xticks(ticks=range(len(results.columns.tolist())), labels=results.columns.tolist(),rotation=90,size=5)
plt.yticks(ticks=range(len(results.index.tolist())), labels=results.index.tolist(),rotation=0,size=5)
plt.tight_layout() 
plt.savefig('plot9',dpi=1000)










 



