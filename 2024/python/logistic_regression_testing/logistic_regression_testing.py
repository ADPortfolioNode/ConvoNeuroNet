import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import requests

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv'
response = requests.get(url)

with open('ChurnData.csv', 'wb') as file:
    file.write(response.content)
#!wget -O ChurnData.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv

print('downloading data and reading data----->')
churn_df = pd.read_csv("ChurnData.csv")
print(churn_df.head())

print('data pre-processing and selection----->')
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

print('practice ----->')
print(churn_df.shape)

print('normalize the data set----->')
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

y = np.asarray(churn_df['churn'])


X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

print('train/test dataset----->')

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

print('Modeling (Logistic Regression with Scikit-learn)')
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)

print('predicting with test set X_test----->')
yhat = LR.predict(X_test)
print(yhat)

#predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):

yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

print('Evaluation----->')
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat,pos_label=0))

#confusion matrix
print('confusion matrix----->')
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

print('Computing confusion matrix')
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


print('Plotting non-normalized confusion matrix----->')
plt.figure()
plt.legend()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')


plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()

print('log loss ----->')
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)

print('logLoss test 2 : LR2----->')
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))

print('calculate hinge loss for both models')
# get the confidence scores for the test samples
from sklearn import svm

# Create an instance of the SVM model
sklearn_svm = svm.SVC()

# Fit the model to your data (replace X_train and y_train with your data)
sklearn_svm.fit(X_train, y_train)

# Now you can use the decision_function method
# to get the confidence scores for your test data
sklearn_pred = sklearn_svm.decision_function(X_test)

# import the hinge_loss metric from scikit-learn
from sklearn.metrics import hinge_loss

# evaluate the hinge loss from the predictions
loss_snapml = hinge_loss(y_test, yhat_snapml)
print("[Snap ML] Hinge loss:   {0:.3f}".format(loss_snapml))

# evaluate the hinge loss metric from the predictions
loss_sklearn = hinge_loss(y_test, yhat_sklearn)
print("[Scikit-Learn] Hinge loss:   {0:.3f}".format(loss_sklearn))

# the two models should give the same Hinge loss

print('done----->')
 
# Now, let's plot the hinge loss
plt.figure(figsize=(10, 6), label= 'Hinge Loss')
plt.bar(['loss sklearn', 'loss snapml'], [loss_sklearn, loss_snapml])
plt.ylabel('Hinge Loss')
plt.xlabel('Model')
plt.title('Hinge Loss of the two models')
plt.grid(True)
plt.show()

