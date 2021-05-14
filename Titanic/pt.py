##  基础函数库
import numpy as np
import pandas as pd

## 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns

## 我们利用 sklearn 中自带的 iris 数据作为数据载入，并利用Pandas转化为DataFrame格式
from sklearn.datasets import load_iris
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn import metrics
import xlwt
import xlrd

warnings.filterwarnings("ignore")  # 忽略版本问题
f2 = open("test.csv")
f = open("train.csv")
names = ['Survived','Pclass','sex_no','Age','SibSp','Parch','Fare','Em_no']
data=read_csv(f,names=names)
data_test = read_csv(f2, names = names)
data_test_NO = data_test.fillna(method="pad")
data_test_use = data_test_NO[['Pclass','sex_no','Age','SibSp','Parch','Fare','Em_no']]
#data.info()
data_test_use.info()
data_no_N = data.fillna(method="pad")
data_no_N.info()
data_target = data_no_N["Survived"]
data_feature = data_no_N[['Pclass','sex_no','Age','SibSp','Parch','Fare','Em_no']]
data_no_N_copy = data_feature.copy()
data_no_N_copy["target"] = data_target
#sns.pairplot(data=data_no_N_copy,diag_kind='hist', hue= 'target')
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(data_feature, data_target, test_size = 0.2, random_state = 2020)
clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000,C=0.3)
clf.fit(x_train, y_train)
## 查看其对应的w
print('the weight of Logistic Regression:',clf.coef_)
## 查看其对应的w0
print('the intercept(w0) of Logistic Regression:',clf.intercept_)
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)

# 利用热力图对于结果进行可视化
predict_list = clf.predict(data_test_use)
dt = pd.DataFrame(predict_list)
dt.to_excel('list.xls')

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
