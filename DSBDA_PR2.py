import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats

url = "StudentsPerformance.csv"
df = pd.read_csv(url)

df.head()

series1 = pd.notnull(df["math score"])
df[series1].head()

le = LabelEncoder()
# male = 1 & female = 0
df['gender'] = le.fit_transform(df['gender'])
newdf=df
df.head()

missing_values = ["Na", "na"]
#converts Na to NaN
df = pd.read_csv("StudentsPerformance.csv",na_values = missing_values)
df.head()

ndf=df # creating a view 
ndf.fillna(0).head() # filling 0 to NaN fields

# taking mean
df['math score'] = df['math score'].fillna(df['math score'].mean()) 

# taking median
df['math score'] = df['math score'].fillna(df['math score'].median())

# taking sd
df['math score'] = df['math score'].fillna(df['math score'].std())
df.head()

m_v=df['math score'].mean()
df['math score'].fillna(value=m_v, inplace=True)
df.head()

ndf.replace(to_replace = np.nan, value = -99)
ndf.head()

ndf.replace(to_replace = np.nan, value = -99)
ndf.head()

ndf.dropna(how = 'all').head()

# making new data frame with dropped NA values
new_data = ndf.dropna(axis = 0, how ='any')
new_data.head()

new_data.describe()

# boxplot
col = ['math score', 'reading score' , 'writing score','placement score']
df.boxplot(col)
plt.show()

print(np.where(df['math score']>90))
print(np.where(df['reading score']<25))
print(np.where(df['writing score']<30))

fig, ax = plt.subplots(figsize = (9,5))
ax.scatter(df['placement score'], df['placement offer count'])

# labels
ax.set_xlabel('(Proportion non-retail business acres)/(town)')
ax.set_ylabel('(Full-value property-tax rate)/($10,000)')
plt.show()

print(np.where((df['placement score']<50)&(df['placement score']>85)&(df['placement offer count']>1)))
print(np.where((df['placement offer count']<3)))

z = np.abs(stats.zscore(df['math score']))
z.head()

threshold = 0.18
sample_outliers = np.where(z <threshold)
sample_outliers

sorted_rscore= sorted(df['reading score'])

q1 = np.percentile(sorted_rscore, 25)
q3 = np.percentile(sorted_rscore, 75)
q1,q3

IQR = q3-q1
lwr_bound=q1-(1.5*IQR)
upr_bound=q3+(1.5*IQR)
lwr_bound, upr_bound

# Print Outliers

r_outliers = []
for i in sorted_rscore:
    if (i<lwr_bound or i>upr_bound):
        r_outliers.append(i)
print(r_outliers)

new_df=df
for i in sample_outliers:
    new_df.drop(i,inplace=True)
new_df.head()

f=pd.read_csv(url)

df_stud=df

nintieth_percentile = np.percentile(df_stud['math score'], 90)

b = np.where(df_stud['math score']>nintieth_percentile,nintieth_percentile, df_stud['math score'])
print("New array:")
b

df_stud.insert(1,"m score",b,True)
df_stud.head()

col = ['reading score']
df.boxplot(col)
plt.show()

median=np.median(sorted_rscore)
median

df2=df
df2['reading score']=np.where(df2['reading score']>upr_bound,median,df2['reading score'])
df2.head()

col = ['reading score']
df.boxplot(col)
plt.show()

new_df['math score'].plot(kind = 'hist')
plt.show()

df['log_math'] = np.log10(df['math score'])
df['log_math'].plot(kind = 'hist')
plt.show()