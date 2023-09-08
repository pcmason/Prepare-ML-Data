#Using 4 different methods to select the best features for the Pima Indians dataset
import pandas
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
#Set precision for output values
set_printoptions(precision=3)

#Load data into dataframe
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
array = data.values

#Separate array into input & output components
x = array[:, 0:8]
y = array[:, 8]

#1) Univariate feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(x, y)

#Summarize scores
print(fit.scores_)
features = fit.transform(x)

#Summarize selected features
print('\n', features[0:5, :])

#2) Recursive feature elimination extracction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, n_features_to_select=3)
fitRFE = rfe.fit(x, y)

#Summarize feature selection
print('\nNum Features: %d' % fitRFE.n_features_)
print('Selected Features: %s' % fitRFE.support_)
print('Feature Ranking: %s' % fitRFE.ranking_)

#3) Principle component analysis feature extraction
pca = PCA(n_components=3)
fitPCA = pca.fit(x)

#Summarize the components
print('\nExplained Variance: %s' % fitPCA.explained_variance_ratio_)
print(fitPCA.components_)

#4) Extra trees classifier feature extraction
treeModel = ExtraTreesClassifier(n_estimators=10)
treeModel.fit(x, y)
print('\n', treeModel.feature_importances_)