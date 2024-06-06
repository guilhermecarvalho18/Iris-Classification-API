import pandas as pd
from preprocessing.data_preprocessor import DataPreprocessor

data = DataPreprocessor('data/raw/iris.csv',
                  header = None,
                  names = ['sepalLength', 
                           'sepalWidth', 
                           'petalLength', 
                           'petalWidth',
                           'class'])

df = data.load_data()


print(df.shape)
print(df.isnull().sum())
print(df.describe())

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['sepalLength'], kde=True)
plt.show()
sns.histplot(df['sepalWidth'], kde=True)
plt.show()
sns.histplot(df['petalLength'], kde=True)
plt.show()
sns.histplot(df['petalWidth'], kde=True)
plt.show()
sns.pairplot(df, hue='class')
plt.show()
sns.countplot(x='class', data=df)
plt.show()
sns.heatmap(df.drop(columns=['class'], axis=1).corr(method='pearson'), annot = True); 
plt.show()




