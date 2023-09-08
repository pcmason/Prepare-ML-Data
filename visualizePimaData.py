#Use pandas methods to visualize the data from the Pima Indians dataset on diabetes
import matplotlib.pyplot as plt
import pandas
import numpy

#Get dataframe
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)

#Create a histogram for each attribute in dataframe
data.hist()
plt.show()

#Create density plots for each attribute
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

#Create box & whisker plots for each attribute
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False)
plt.show()

#Plot the correlation matrix
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

#Create a scatterplot matrix for the attributes
from pandas.plotting import scatter_matrix
scatter_matrix(data)
plt.show()