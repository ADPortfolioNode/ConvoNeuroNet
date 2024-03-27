from __future__ import print_function
import opendatasets as od

# download the dataset (this is a Kaggle dataset)
# during download you will be required to input your Kaggle username and password
od.download("https://www.kaggle.com/mlg-ulb/creditcardfraud")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

#Data info: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.htm
# read the input data
raw_data = pd.read_csv('creditcardfraud/creditcard.csv')
print("There are " + str(len(raw_data)) + " observations in the credit card fraud dataset.")
print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")

# display the first rows in the dataset
print('dataset rows loaded',raw_data.head())

#Uncomment the following lines if you are unable to download the dataset using the Kaggle website.

#url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
#raw_data=pd.read_csv(url)
#print("There are " + str(len(raw_data)) + " observations in the credit card fraud dataset.")
#print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")
#raw_data.head()

#In practice, a financial institution may have access to a much larger dataset of #transactions. To simulate such a case, we will inflate the original one 10 times.

n_replicas = 10

# inflate the original dataset
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)

print("The inflated data:" + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
print("The inflated data: " + str(len(big_raw_data.columns)) + " variables in the dataset.")

print(' first rows in the new dataset', big_raw_data.head())


print ('getting the set of distinct classes in the dataset')
labels = big_raw_data.Class.unique()

print ('get the count of each class')
sizes = big_raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.title("Class distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

print("////practice////")
# plot the class value counts
plt.hist(big_raw_data.Amount.values, 6, histtype='bar', facecolor='g')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('Amount Frequency')
plt.show()

print("minimum amount value is", np.min(big_raw_data.Amount.values))
print("maximum amount value is", np.max(big_raw_data.Amount.values))
print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))

#DATA PREPROCESSING
print("DATA PREPROCESSING")
# data preprocessing such as scaling/normalization is typically useful for 
# linear models to accelerate the training convergence

print (' standardize features by removing the mean and scaling to unit variance')
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30])
data_matrix = big_raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# print (data normalization
X = normalize(X, norm="l1")

# print the shape of the features matrix and the labels vector
print('X.shape=', X.shape, 'y.shape=', y.shape)

#DATA TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)       
print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

#divide the pre-processed dataset into a subset to be used for training the model (the train set) and a subset to be used for evaluating the quality of the model (the test set).
#The train set is used to train the model, while the test set is used to evaluate the model's performance on unseen data.

print('BUILDING A DECISION TREE CLASSIFIER MODEL WITH SCIKIT-LEARN')
# compute the sample weights to be used as input to the train routine so that 
# it takes into account the class imbalance present in this dataset
print('computing the sample weights to be used as input to the train routine')
w_train = compute_sample_weight('balanced', y_train)

print('importing the Decision Tree Classifier Model from scikit-learn')
from sklearn.tree import DecisionTreeClassifier

# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

print('training a Decision Tree Classifier using scikit-learn')
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
snapml_time = time.time() - t0
print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))

sklearn_time = time.time() - t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

# Evaluate: the Scikit-Learn and Snap ML Decision Tree Classifier Models
print('Snap ML vs Scikit-Learn training speedingup')
training_speedup = sklearn_time / snapml_time
print('[Decision Tree Classifier] Snap ML vs. Scikit-Learn speedup : {0:.2f}x '.format(training_speedup))

print('run inference and compute the probabilities of the test samples')
# to belong to the class of fraudulent transactions
sklearn_pred = sklearn_dt.predict_proba(X_test)[:, 1]

('EVALUATION: the Compute Area Under the Receiver Operating Characteristic') 
print('Curve (ROC-AUC) score from the predictions scores')
sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred)
print('[Scikit-Learn] ROC-AUC score : {0:.3f}'.format(sklearn_roc_auc))

# instantiate the snapml_dt variable as a DecisionTreeClassifier object
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

# run inference and compute the probabilities of the test samples
# to belong to the class of fraudulent transactions
snapml_dt.fit(X_train, y_train)
snapml_pred = snapml_dt.predict_proba(X_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic
# Curve (ROC-AUC) score from the prediction scores
snapml_roc_auc = roc_auc_score(y_test, snapml_pred)   
print(f'[Snap ML] ROC-AUC score : {0:.3f}'.format(snapml_roc_auc))

print('Build a Support Vector Machine model with Scikit-Learn')
# import the linear Support Vector Machine (SVM) model from Scikit-Learn
from sklearn.svm import LinearSVC

print(' instatiate a scikit-learn SVM model')
# to indicate the class imbalance at fit time, set class_weight='balanced'
# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

print('train a linear Support Vector Machine model using Scikit-Learn')
t0 = time.time()
sklearn_svm.fit(X_train, y_train)
sklearn_time = time.time() - t0
print("[Scikit-Learn] Training time (s):  {0:.2f}".format(sklearn_time))

print('Build a Support Vector Machine model with Snap ML')
# import the Support Vector Machine model (SVM) from Snap ML
from snapml import SupportVectorMachine

# in contrast to scikit-learn's LinearSVC, Snap ML offers multi-threaded CPU/GPU training of SVMs
print('to use the GPU, setting the use_gpu parameter to True')
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, use_gpu=True, fit_intercept=False)
print('set the number of threads used at training time, one needs to set the n_jobs setting')
snapml_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
print('snapML params',snapml_svm.get_params())

print('training an SVM model using Snap ML')
t0 = time.time()
model = snapml_svm.fit(X_train, y_train)
snapml_time = time.time() - t0
print("[Snap ML] Training time (s):  {0:.2f}".format(snapml_time))

#Evaluate the Scikit-Learn and Snap ML Support Vector Machine Models
print('computing the Snap ML vs Scikit-Learn training speedup')
training_speedup = sklearn_time/snapml_time
print('[Support Vector Machine] Snap ML vs. Scikit-Learn training speedup : {0:.2f}x '.format(training_speedup))

# run inference using the Scikit-Learn model
print('getting the confidence scores for the test samples using the Scikit-Learn model to run inference')
sklearn_pred = sklearn_svm.decision_function(X_test)

print('evaluating accuracy on test set')
acc_sklearn  = roc_auc_score(y_test, sklearn_pred)
print("[Scikit-Learn] ROC-AUC score:   {0:.3f}".format(acc_sklearn))