import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Create a sample DataFrame

df = pd.read_csv("Accuracy_plot.csv",index_col="Models")
del df["Unnamed: 0"]
df = df.sort_values(by='Overall Performance',ascending=False)
#df = df[df["Overall Performance"]>0.1]
del df["Overall Performance"]
df = df.T



 

# Perform t-test for each pair of columns
results = pd.DataFrame(index=df.columns, columns=df.columns)
alpha=0.1
for col1 in df.columns:
    for col2 in df.columns:
        t_statistic, p_value = ttest_ind(df[col1], df[col2])
        try:
            if p_value<alpha:
                results.at[col1, col2] = 1
            else:
                results.at[col1, col2] = 0
        except:
            if p_value[0]<alpha:
                results.at[col1, col2] = 1
            else:
                results.at[col1, col2] = 0



results.to_csv("Significance_plot_performance.csv")
results = results.apply(pd.to_numeric)
plt.figure(figsize=(15, 15))
sns.heatmap(results, annot=True, cmap='coolwarm', fmt=".0f", linewidths=.5)

plt.title('Heatmap of p-values between Models performance')
plt.tight_layout()

plt.savefig("plot3.png",dpi=1000)

#plt.show()
# Display the results
print(results)
