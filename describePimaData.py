#Using descriptive statistics to reveal more information on the Iris dataset

#Load dataset into dataframe
import pandas
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
#Make output more readable
pandas.set_option('display.width', 150)
pandas.set_option('display.precision', 3)

#View first 20 rows of data
peek = data.head(20)
print(peek)

#Get the dimensions of the data
shape = data.shape
print('\n', shape)

#Get the data types for each attribute
types = data.dtypes
print('\n', types)

#Output descriptive statistics
description = data.describe()
print('\n', description)

#Output class distribution for the class attribute
class_counts = data.groupby('class').size()
print('\n', class_counts)

#Output correlations of the attributes
correlations = data.corr(method='pearson')
print('\n', correlations)

#Output the skew for each attribute
skew = data.skew()
print('\n', skew)