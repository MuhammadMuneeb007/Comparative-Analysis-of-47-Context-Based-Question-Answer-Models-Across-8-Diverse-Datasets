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
import numpy as np
from numpy.polynomial import Polynomial

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
'QuAC',
'ScienceQA',
'biomedical_cpgQA',
'bioasq10b-factoid',
'ielts',
'wikipedia',
'JournalQA'
]
X = [
    3,
     84,
     147,
     161,
     351,
     907,
     5227,
     5568]
numberofquestions =[
10000,
618,
1378,
535,
983,
50,
2037,
273

]


accur = []
timee = []
for a in datasets:
    #accur.append(a+"_Accuracy")
    accur.append(a)
    
    timee.append(a+"_Time")

results = pd.DataFrame()
results2 = pd.DataFrame()

count=0
for loop in datasets:

    data = pd.read_csv(loop+os.sep+loop+"_results.csv",index_col=0)
    data = data.sort_index()
    results[loop] = data["Accuracy"].values

    #results[loop+"_Accuracy"] = data["Accuracy"].values
    try:
        results2[loop] = data["Time in seconds"].values/(numberofquestions[count]*60)
    except:
        results2[loop] = data["Time"].values/(numberofquestions[count]*60)
    count=count+1


def interpolate(X,Y):
    coefficients = np.polyfit(X, Y, 1)
    execution_time_equation = Polynomial(coefficients)
    print(execution_time_equation)
    return execution_time_equation


Executiontimetrend = []

for index, row in results2.iterrows():
    row_as_array = row.to_numpy()  # or row.values
    Executiontimetrend.append(interpolate(X,row_as_array).coef[0])




results['Overall Performance'] = results[accur].sum(axis=1)
results['Overall Performance'] = results['Overall Performance']/len(datasets)

results2['Overall Performance'] = results['Overall Performance'].values

results2['Overall Time in minutes'] = results2[accur].sum(axis=1)
results2['Overall Time in minutes'] =results2['Overall Time in minutes']/len(datasets)



sample_array = pd.read_csv("Final_model_info.csv")["Model Size (MB)"].values
results2["Model Size (MB)"] = sample_array

for loop in datasets:
    print(round(results2[loop].corr(results2['Overall Performance']),2))

print("-")

for loop in datasets:
    print(round(results2[loop].corr(results2["Model Size (MB)"]),2))
print("-")

for loop in datasets:
    print(round(results2[loop].corr(results2['Overall Time in minutes']),2))
print("-")

print(round(results2['Overall Time in minutes'].corr(results2['Overall Performance']),2))
print(round(results2['Overall Time in minutes'].corr(results2["Model Size (MB)"]),2))
print(round(results2['Overall Performance'].corr(results2["Model Size (MB)"]),2))
#exit(0)

del results2["Model Size (MB)"]
normalized_array = (sample_array - np.min(sample_array)) / (np.max(sample_array) - np.min(sample_array))


print(Executiontimetrend)

#results2 = results2.sort_values(by='Overall Performance',ascending=False)


del results

results = results2.copy()

results['Models'] = pd.read_csv(loop+os.sep+loop+"_results.csv",index_col=0)['Models'].values
results["Model Size"] = normalized_array
#results["Execution time trend"] = Executiontimetrend

results = results.sort_values(by='Overall Performance',ascending=False)
#del results['Overall Performance']

results.to_csv("Time_plot.csv")

results = results.set_index('Models')
sns.heatmap(results.values, annot=True,linewidths=0.1,annot_kws={"size": 4},cmap='viridis',vmin=0, vmax=1)
plt.xticks(ticks=range(len(results.columns.tolist())), labels=results.columns.tolist(),rotation=90,size=4)
plt.yticks(ticks=range(len(results.index.tolist())), labels=results.index.tolist(),rotation=0,size=4)
#plt.show()
plt.tight_layout() 
plt.savefig('plot2.png',dpi=1000)




