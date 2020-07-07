#1.libraries
import matplotlib.pyplot as plt
import pandas as pd
from	pandas.tools.plotting	import	scatter_matrix
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

#2. Data reading
data = pd.read_csv("red_wine.csv")

data.loc[data['quality'] <= 5, 'quality'] = 0
data.loc[data['quality'] > 5, 'quality'] = 1


#Division of data into features and quality
data_x = data.iloc[:,0:11]
data_y = data.iloc[:,11:12]


#Division of data into education and testing
x_train, x_test,y_train,y_test = train_test_split(data_x,data_y ,test_size=0.2, random_state=0)


#Data creation
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
'''
#headings of features are written
features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['quality']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

#two-dimensional drawing of data set
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])



finalDf = pd.concat([principalDf, data[['quality']]], axis = 1)

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1]
colors = ["blue" , "red" ]
for quality, color in zip(targets,colors):
    indicesToKeep = finalDf['quality'] == quality
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


#three-dimensional drawing of data set
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2',"principal component 3"])



finalDf = pd.concat([principalDf, data[['quality']]], axis = 1)

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111 , projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel("principal component 3", fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = [0,1]
colors = [ 'blue' , "red" ]
for quality, color in zip(targets,colors):
    indicesToKeep = finalDf['quality'] == quality
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
                  ,marker = "o", c = color
               , s = 20)
ax.legend(targets)
ax.grid()
'''
#Application of NAive Bayes algorithm and evaluation with confusion_matrix
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(priors=None)

gnb.fit(X_train, y_train) #Train

pred_gnb = gnb.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_gnb)
print('     Gaussian Naive Bayes')
print("           quality")
print("    3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((108+122)/(1598*0.2))*100)
print("Succes Rate: ",s)



#Application of Logistic Regression algorithm and evaluation with confusion_matrix
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=None)

logr.fit(X_train,y_train) #Train

pred_logr = logr.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_logr)
print('     Logistic Regression')
print("           quality")
print("    3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((114+130)/(1598*0.2))*100)
print("Succes Rate: ",s)

#Application of Logistic Regression algorithm and evaluation with confusion_matrix
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=None ,multi_class = "multinomial" ,solver = "lbfgs" ,C =1 ,tol = 2 ,max_iter =100 )

logr.fit(X_train,y_train) #Train

pred_logr = logr.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_logr)
print('     Logistic Regression')
print("           quality")
print("    3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((115+130)/(1598*0.2))*100)
print("Succes Rate: ",s)

#Implementation of K-Nearest Neighbors algorithm and evaluation with confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train) #Train

pred_knn = knn.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_knn)
print("K-Nearest Neighbors")
print(cm)
s = (((98+130)/(1598*0.2))*100)
print("Succes Rate: ",s)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=100 ,  weights = "distance" , algorithm="kd_tree" 
                           , leaf_size=75 , p=4.5 , metric="minkowski")
knn.fit(X_train,y_train) #Train

pred_knn = knn.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_knn)
print("K-Nearest Neighbors")
print(cm)
s = (((107+139)/(1598*0.2))*100)
print("Succes Rate: ",s)



#Decision tree algorithm implementation and evaluation with confusion_matrix
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)

dtc.fit(X_train,y_train) #Train

pred_dtc = dtc.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_dtc)
print('Decision Tree Classifier')
print("           quality")
print("   3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((113+122)/(1598*0.2))*100)
print("Succes Rate: ",s)

#Decision tree algorithm implementation and evaluation with confusion_matrix
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0 , criterion="gini" , splitter="best" , min_samples_leaf=4)

dtc.fit(X_train,y_train) #Train

pred_dtc = dtc.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_dtc)
print('Decision Tree Classifier')
print("           quality")
print("   3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((117+122)/(1598*0.2))*100)
print("Succes Rate: ",s)


#Application of Support Vector Classifier algorithm and evaluation with confusion_matrix
from sklearn.svm import SVC
rbf_svc = SVC(random_state=0 , probability = True )
rbf_svc.fit(X_train, y_train) #Train

pred_svc = rbf_svc.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test, pred_svc)
print("Support Vector Classifier")
print("           quality")
print("    3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((110+131)/(1598*0.2))*100)
print("Succes Rate: ",s)

#Application of Support Vector Classifier algorithm and evaluation with confusion_matrix
from sklearn.svm import SVC
rbf_svc = SVC(random_state=0 , probability = True , kernel="rbf" , C=0.3)
rbf_svc.fit(X_train, y_train) #Train

pred_svc = rbf_svc.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test, pred_svc)
print("Support Vector Classifier")
print("           quality")
print("    3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((114+130)/(1598*0.2))*100)
print("Succes Rate: ",s)


#Implementation of Random Forest algorithm and evaluation with confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( random_state=0 )

rfc.fit(X_train,y_train) #Train

pred_rfc = rfc.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_rfc)
print('Random Forest Classifier')
print("           quality")
print("    3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((124+134)/(1598*0.2))*100)
print("Succes Rate: ",s)

#Implementation of Random Forest algorithm and evaluation with confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( random_state=0 ,n_estimators=12 , min_samples_split=2 )

rfc.fit(X_train,y_train) #Train

pred_rfc = rfc.predict(X_test) #Predict

#Confusion_matrix
cm = confusion_matrix(y_test,pred_rfc)
print('Random Forest Classifier')
print("           quality")
print("    3" , "  4" , "  5" , "  6" , "  7" , "  8")
print(cm)
s = (((124+135)/(1598*0.2))*100)
print("Succes Rate: ",s)



print("votingggggg")

clf2 = KNeighborsClassifier(n_neighbors=100 ,  weights = "distance" , algorithm="kd_tree" 
                           , leaf_size=75 , p=4.5 , metric="minkowski")
clf3 = SVC(random_state=0 , probability = True , kernel="rbf" , C=0.3)
clf4 = RandomForestClassifier( random_state=0 ,n_estimators=12 , min_samples_split=2 )

clfs = [('knn', clf2),('SVC', clf3),("rf" , clf4) ]

for clf_tuple in clfs:
    clf_name, clf = clf_tuple
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print('Model:{} - Accuracy:{:.2f}%'.format(clf_name, acc*100))
    
    
hard_clf = VotingClassifier(estimators=clfs, voting='hard' )

hard_clf.fit(X_train, y_train)

predict_proba = hard_clf.predict(X_test)

print('hard voting: {:.2f}'.format(hard_clf.score(X_test, y_test)*100))


soft_clf = VotingClassifier(estimators=clfs, voting='soft'  )
soft_clf.fit(X_train, y_train)
predict_proba = hard_clf.predict(X_test)
print('soft voting: {:.2f}'.format(soft_clf.score(X_test, y_test)*100))

