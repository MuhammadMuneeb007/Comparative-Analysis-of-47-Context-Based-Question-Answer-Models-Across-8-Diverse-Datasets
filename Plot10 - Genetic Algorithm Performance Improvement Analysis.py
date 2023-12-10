from audioop import avg
from genericpath import isdir
import pandas as pd
import numpy as np
import os
import textstat
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
import seaborn as sns


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

accur = []
timee = []
for a in datasets:
    accur.append(a+"_Train")
    timee.append(a+"_Test")

results = pd.DataFrame()

def getmefolddirec(solutionlength,alldirec):
    newdirec = []
    for d in alldirec:
        if "Fold" in d and "Length" in d and "_" in d:
            #print(d.split("_")[2].split(".csv")[0])
            if d.split("_")[2].split(".csv")[0]==str(solutionlength):
                newdirec.append(d)
    #print(newdirec)
    return newdirec


results = pd.DataFrame()

for loop in datasets:
    finaltrain = []
    finaltest = []
    for solutionlength in range(2,48):
        alldirec = os.listdir(loop)
        alldirec =getmefolddirec(solutionlength,alldirec)
        test = []
        train = []
  
        for newdic in alldirec:
            train.append(pd.read_csv(loop+os.sep+newdic)["train"].values)
            test.append(pd.read_csv(loop+os.sep+newdic)["test"].values)
        train = np.mean(train)
        test = np.mean(test)
        finaltrain.append(train)
        finaltest.append(test)
    results[loop+"_Train"] = finaltrain
    results[loop+"_Test"] = finaltest
    
 
results = results.sub(results.iloc[0])

results['Models'] = list(range(2,48))


results = results.set_index('Models')
sns.heatmap(results.values, annot=True,linewidths=0.1,annot_kws={"size": 3},cmap='viridis',vmin=0, vmax=results.values.max())
plt.xticks(ticks=range(len(results.columns.tolist())), labels=results.columns.tolist(),rotation=90,size=4)
plt.yticks(ticks=range(len(results.index.tolist())), labels=results.index.tolist(),rotation=0,size=4)
#plt.show()
plt.tight_layout() 
plt.savefig('Plot10.png',dpi=1000)




