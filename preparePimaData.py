#Using 4 different methods to preprocess the data from the Pima Indians dataset
import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer

#Load dataset into a pandas dataframe
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
array = data.values
#Set print options for summarizations
numpy.set_printoptions(precision=3)

#Separate array into input & output components
x = array[:, 0:8]
y = array[:, 8]

#1) Rescale attribute data to values between 0 & 1
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(x)

#Summarize transformed data
print(rescaledX[0:5, :])

#2) Standardize attribute data
standScaler = StandardScaler().fit(x)
standRescaledX = standScaler.transform(x)

#Summmarize standardized data
print('\n', standRescaledX[0:5, :])

#3) Normalize attribute data
normScaler = Normalizer().fit(x)
normRescaledX = normScaler.transform(x)

#Summarize normalized data
print('\n', normRescaledX[0:5, :])

#4) Binarize attribute data
binScaler = Binarizer(threshold=0.0).fit(x)
binRescaledX = binScaler.transform(x)

#Summarize binarized data
print('\n', binRescaledX[0:5, :])




