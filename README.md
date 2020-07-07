Big data is the process of producing perspectives for decision making. Machine learning is to analyze a large amount of data stored in one place and develop a perspective to produce decisions in the area of interest. The purpose of this project research is to classify a specific set of data according to defined criteria. Various algorithms are used for this classification. The goal is to make quality determination using these classifications. Python coding language is selected to make these classifications. Python is used with the help of Anaconda Navigator.

Data sets will be used for learning and testing in the project. Our data set doesn't have enough data to call big data, but this is enough to understand machine learning and big data concepts. The data will be made more regular to use the program. In such projects, missing data is a problem. Therefore, our first job is to check the missing data and complete the missing parts. The edited codes will be transferred to the program and the data will be processed. The data set will be divided into two sets as training and testing for this process. According to the learning data, our machine will learn after make predictions according to test data. We will check these estimates using different algorithms. According to these results, we will find the algorithm that produces the best results for our data set.

We will use the "confusion_matrix" command to see the results of the algorithm more clearly. As a result of this command, our program will draw a matrix. Looking at this matrix we will see the data that the algorithm predicts correctly and incorrectly. Then we will calculate the performance of the algorithm. When working with very large data, it is almost impossible to capture true or false estimates. So this matrix will help us.

When the project is completed, big data, machine learning, data set, data processing, data analysis, training data, test data will be understood. In this way, we will have more information about big data and machine learning.


### INTODUCTION

Machine learning is used in many areas today. Machine learning, which is the greatest opportunity for big data, analyzes data from different sources and learns from these data. Machine learning creates systems that are difficult to establish mathematically. Machine learning is a method paradigm that uses mathematical and statistical methods to make inferences from existing data and make predictions about these unknowns. Different algorithms, different features and different data are made trial and error. Models are created using trial and error. For machine learning;
>	Obtaining data,,
	Preprocessing of data and editing of data,
	Creating models,
	Testing the models,
	Measuring the performance of the model,
	
steps are followed.
Python is the most convenient and most preferred programming language for machine learning. For this reason, our project was created in Python programming language. There are many projects developed using Python. For this reason, a faster solution to the problems that may occur in the project can be found. Python allows for easy system installation with the libraries and tools it has. During complex operations such as processing large data, the programmer looks for convenience. For this reason, Python is attractive.

#### 1.1	Research

Before the project was created, it was aimed to learn the concept of big data and researches were made according to this purpose. As we started the project, we examined the code sets and selected a data set called “red wine quality”. The data were analyzed and edited so that the data could be used. We understand what the missing data means and we have done the necessary tests in this direction. 
 In order to learn the data by analyzing the data, we searched the Python compilers and we selected the program Anaconda Navigator. We decided that this program is the most usable program based on the experience of people who have used large data before. We have researched and selected the program "Spyder" which contains many compilers because Spyder for visualization process has more useful interface.
In order to perform data analysis in Python, we searched the background of the algorithms that are available. Gaussian Naive Bayes, Logistic Regression, Decision Trec Classifier, K-Nearest Neighbors, Support Vector Classifier, Random Forest Classifier algorithms are applied one by one and it is aimed to get the best result.

#### 1.2	Problem

Nowadays people are aiming to find the best and perfect in every field. In order to achieve the best, the products that were produced were determined as the benchmarks. Saving of the previous products according to the desired features enabled the creation of large data sets. The capture of the standard will be possible by processing this data. However, it is impossible for such a large and complex data to be analyzed manually by a human hand. In order to overcome this problem, machine learning method has been found and developed.
This project has been created to detect quality wine. The data set we select consists of 1598 rows and 12 columns. It is very difficult to examine and analyze such a lot of data manually and to reach a quality product as a result. In order to overcome this problem, machine learning and various algorithms are used.
 
#### 1.3	Project Goals

In determining the quality of wine, our data set is examined according to 11 different characteristics. These properties are given under the titles of fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
It is aimed to determine the quality status of the products analyzed according to these characteristics. Thus, we will create quality order for products according to the determined standards. We will find quality wines with this sort.
 ### METHOD
	
Machine learning method was used for the project. Python scripting language is preferred to take advantage of the best possibilities for machine learning. In order to get the best results from the data set, we searched previously created projects. We have chosen the Spyder compiler in the Anaconda Navigator program because it has many advantages. As can be seen in Annex-1, in the creation of the project;

- 	To scale data  “from sklearn.preprocessing import StandardScaler”,
- 	To use the Gaussian Algorithm “from sklearn.naive_bayes import GaussianNB”,
-	To use the Decision Tree Classifier Algorithm “from sklearn.tree import DecisionTreeClassifier”,
-	To use the K-Nearest Neighbors Classifier Algorithm “from sklearn.neighbors import KNeighborsClassifier”,
-	For the SVC algorithm to be used “from sklearn.svm import SVC”,
-	To use the Random Forest Algorithm “from sklearn.ensemble import RandomForestClassifier”,
-	To use the Logistic Regression Algorithm ·”from sklearn.linear_model import LogisticRegression”,
-	For visualization “import matplotlib.pyplot as plt” ,
-	To organize data “import pandas as pd”,
-	To separate data into a set of training and tests “from sklearn.model_selection import train_test_split”,
-	For table visualization “from pandas.tools.plotting	import	scatter_matrix”,
-	To use the Confusion Matrix “from sklearn.metrics import confusion_matrix”
-	To use Voting Classifier Algorithm “from sklearn.ensemble import VotingClassifier”
-	To use the PCA visualization method “from sklearn.decomposition import PCA”
-	To 3D visualization “from mpl_toolkits.mplot3d import Axes3D”
-	To use Voting Classifier “from sklearn.ensemble import VotingClassifier”

also “from sklearn.metrics import accuracy_score” and “import seaborn as sns” commands were used in libraries. By calling these libraries, which are one of the attractive points of the program we use, the necessary commands will be used directly.

#### 2.1 Data Processing

The codes written for the processing of the data and the formation of the program are given in Annex-1. In this section, we will describe the code fragments included in these codes that enable us to process the data.

The data we will use includes the 11 chapters of the wine samples which were previously produced. A “quality” column was created using these features. There are 1598 wine samples in this data set. This data set was created with the csv file type to make it suitable for our program. 

Our data set is called red_wine.csv.  While importing this data set, we received an equalization to the variable named data. After this part of the code, the variable we use as data will represent our data group. As shown in the following code block, the transfer of the data set to the program is complete.

```sh
data = pd.read_csv('red_wine.csv')
```

![](https://user-images.githubusercontent.com/46966075/86847085-7488f380-c0b4-11ea-8a81-8e6b8b244074.png)

- Figure  1: A Part of a Data Set Taken into the Program

The biggest problem after getting the data set into the program is that some data is missing in the data set. We've looked at whether our dataset has this problem. We couldn't find the missing in data in the data set we were using.

After making sure that there is no data missing, we can make the data ready for use. First of all, when we investigate the logic of working and creating our data, we can say that the quality of wine is created according to the given 11 properties. The quality of the wine is determined by processing the 11 columns that contain these 11 properties and is written to the quality column, which is our 12th column. In order to be able to process our data set with these 12 columns, we must first separate the quality column from the other 11 properties lines. Thus, we can use our properties and quality separately. In order to accomplish this partitioning process, we first created a data_x variable and have the first 11 columns imported into this variable. For the quality column, we created the data_y variable. We used the data.iloc [] command to split our data set called data. We provided these operations with the following code block.

```sh
data_x = data.iloc[:,0:11]
data_y = data.iloc[:,11:12]
```

We have created our variables to operate with our data set. After this process we know that data_x represents columns of properties and data_y represents quality columns. In order for a machine learning to take place, we have indicated that the machine should be given the data it can learn. In this direction, we need to create a training data set for the machine to learn. In addition, we need to create a test dataset to measure the performance of this learning process.

Our data set is in 6 levels (3, 4, 5, 6, 7, 8). The data set formed as such is shown in Figure 2.
 
![](https://user-images.githubusercontent.com/46966075/86847088-75ba2080-c0b4-11ea-9668-8360109123f6.png)
- Figure 2:  Visualization of Data in Two Dimensions by PCA Method

As shown in the figure, the 6-featured data is difficult for classification. It is better to divide the data into 0 and 1 instead of this. So 3, 4, 5 are chosen as poor quality 6, 7, 8 are chosen as good quality. As a result of this process, a dataset as in Figure 3 was obtained.

![](https://user-images.githubusercontent.com/46966075/86847090-75ba2080-c0b4-11ea-932b-cd429803128e.png)
- Figure 3: A Part of a Our New Data Set

We will divide the data_x data into x_train training and x_test test datasets for this process that we will create with the train_test_split command. We will also apply this partition to our data_y quality dataset. As a result of this process, we will create y_train training and test datasets. We will use 20 percent of our data for the training of our machine. This is accomplished with the test_size = 0.2 command. The process of creating training and test datasets is provided by the code block given below. In order to use these operation commands, the library related to the from sklearn.cross_code was called in the code. It is seen in the following code block.
```sh
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(data_x,data_y ,test_size=0.2, random_state=0)
```

We created our training and test datasets so that our machine can learn our data. However, this data can be made more usable as well as available. Thus, our program can use the data easier and more measured. The StandardScaler function is used to perform this operation. In order to use this function, it is necessary to call the required library with the from sklearn.preprocessing import StandardScaler  command. We set x_train data to the X_train variable and X_test variable to the x_train data. We use the following codes in our code set for these operations.
```sh
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
```

We said that the quality of wine was determined according to 11 properties. We used the Correlation Matrix to find out how much each of these 11 features affects quality. This matrix is a 12x12 value matrix. It shows us how these qualities affect the quality. Thus, we can learn which features are dominant in quality creation. The creation of this matrix is provided by the following codes.

```sh
corr_matrix = data.corr()
corr_matrix["quality"].sort_values(ascending=False)
print(corr_matrix)
```
The matrix created with the help of these codes is given in Figure 4. Colorings were applied according to the similarities of the values to the standard value. 

![](https://user-images.githubusercontent.com/46966075/86847091-7652b700-c0b4-11ea-9f15-7ea7766bb5a4.png)
- Figure  4:  Correlation Matrix

In addition, the correlation in Figure 5 is given as a clearer table. This image has been created with the logic of the table. Sort from the most effective material to less effective material.

![](https://user-images.githubusercontent.com/46966075/86847093-7652b700-c0b4-11ea-84dd-b64884c0ddac.png)
- Figure 5:  Correlation Matrix Table

According to this matrix, we found the characteristics which have the highest positive correlation and the quality status related to these properties. Also, we drawed a table according to these features. Figure 6 is this table.

In unregulated data, the quality of wines between 3 and 8 is stated. According to the quality criteria, 3, 4, 5 point wines are of poor quality; the wines with 6, 7 and 8 points were given in good quality. In order to see how much of our data is clustered, we tried to visualize what quality. For this visualization to be performed by the Principal Component Analysis (PCA) method, first decide which features of the data set to use. 

According to the above features, we have created a two-dimensional table after having expressed our data as 0 and 1. After this feature determination, x is created to keep the data under these headings in our dataset called data and these values are equalized to this variable. As using codes the datas under the quality heading are equalized to y. With the x = StandardScaler().fit_transform(x)  command, the variable x is made more available. As a result of this process, our data is ready for use with PCA visualization method.

```sh
features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
x = data.loc[:, features].values
y = data.loc[:,['quality']].values

```


![](https://user-images.githubusercontent.com/46966075/86847095-76eb4d80-c0b4-11ea-8b79-83d79226da00.png)
- Figure 6: Correlation Table

The data were first placed on a two-dimensional plane. These two planes we have created in the names Principal Component 1 and Principal Component 2 appear in Figure 7. These two sizing processes are based on our quality criteria: 0  and 1. The visual quality of the data 0 as blue , the data 1 as red. This will enable us to see the visualization of our data in our quality criteria. The code block used for this process is below.

```sh
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
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
```
![](https://user-images.githubusercontent.com/46966075/86847102-781c7a80-c0b4-11ea-8e89-0a806f50437a.png) 
- Figure 7: Visualization of Data in Two Dimensions by PCA Method

 Although the two-dimensional visualization of the data gives an idea of the situation, the surplus data cannot be understood. Therefore, it is better to visualize data in three dimensions. For this purpose, in addition to the steps we follow in two dimensions, the difference will only be at the columns determination point. The codes we will use for this visualization which we will examine the grouping on the same criteria are like following code blog. The difference is to add only one dimension.
When the codes are examined, the difference is easily understood. We created another dimension named principal component 3 in addition to the plane we created from Principal Component 1 and Principal Component 2 dimensions. Then we put the data that we placed in two dimensions in Figure 8 as well as in three dimensions. As in the same two-dimensional visualization, in the three-dimensional visualization, data with the criteria of 0  and  were blue and red.

![](https://user-images.githubusercontent.com/46966075/86847104-78b51100-c0b4-11ea-859e-aba3a0d635dc.png)
- Figure 8: Visualization of Data in Three Dimensions by PCA Method

With this visualization, we can better understand the distribution of our data. With this situation, we have realized the importance of visualization during data analysis.
The processing of the data is completed in this section. So far data processing and visualization has been provided. After this part, Gaussian Naive Bayes, Logistic Regression, Decision Trec Classifier, K-Nearest Neighbors, Support Vector Classifier, Random Forest Classifier algorithms to analyze our data and to find the most accurate way of analyzing our data is based on these analyzes. The finding of this algorithm will show us how accurate our machine produces our data and will enable us to make inferences.

### CONCLUSIONS AND COMMENTS
	
We have been able to process and manipulate our data. We will experiment with our algorithms and find the best results. We will try to get the best results by experimenting with different algorithms in the default settings. Then we will try to increase the success of the algorithm by experimenting with various parameters.

The code that allows us to process our data with the Gaussian Naive Bayes algorithm without parameters is as follows. In the algorithm we created with the gnb variable, we provided the evaluation according to our quality criteria 0 and 1.
```sh
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(priors=None)
gnb.fit(X_train, y_train) #Train
	pred_gnb = gnb.predict(X_test) #Predict
#Confusion_matrix
cm = confusion_matrix(y_test,pred_gnb)
print('     Gaussian Naive Bayes')
print("           quality")
print("0”,”1”)
print(cm)
s = (((108+122)/(1598*0.2)))
print("Succes Rate: ",s)
```

The confusion matrix created to evaluate the performance of the algorithm is as in Figure 9. The color shown in the diagonal shows the correct predicted values. When we calculate these results as a percentage, we achieve success at 0.71 levels.

|| 0| 1| 
| ------ | ------ | ------ |
|0| 108| 40|
|1|50|122|
- Figure 9:  Gaussian Naive Bayes Algorithm Without Parameters Matrix Representation

 The code that allows us to process our data with the Logistic Regression algorithm with default value is as follows. In the algorithm we created with the logr variable, we made the evaluation according to our quality criteria 0 and 1.

The confusion matrix used to evaluate the performance of the algorithm is as in Figure 10. The color shown in the diagonal shows the correct predicted values. When we calculate these results as a percentage, we achieve a success of 0.76.

```sh
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=None)
logr.fit(X_train,y_train) #Train
pred_logr = logr.predict(X_test) #Predict
#Confusion_matrix
cm = confusion_matrix(y_test,pred_logr)
print('     Logistic Regression')
print("           quality")
print("0", “1”)
print(cm)
s = (((114+130)/(1598*0.2)))
print("Succes Rate: ",s)
```


|| 0| 1| 
| ------ | ------ | ------ |
|0| 114| 34|
|1|42|130|
- Figure  10:  Logistic Regression Without Parameters Matrix Representation

The code that allows us to process our data with the Decision Tree Classifier algorithm with default value is as follows. In the algorithm we created with the dtc variable, we provided the evaluation according to our quality criteria 0 and 1. The confusion matrix used to evaluate the performance of the algorithm is as in Figure 11.
```sh
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train,y_train) #Train
pred_dtc = dtc.predict(X_test) #Predict
#Confusion_matrix
cm = confusion_matrix(y_test,pred_dtc)
print('Decision Tree Classifier')
print("           quality")
print("0”, “1”)
print(cm)
s = (((113+122)/(1598*0.2)))
print("Succes Rate: ",s)
```


The color shown in the diagonal shows the correct predicted values. When we calculate these results as a percentage, we achieve a success of 0.73.

|| 0| 1| 
| ------ | ------ | ------ |
|0| 113| 35|
|1|50|122|
- Figure  11:  Decision Tree Classifier Algorithm Without Parameters Matrix Display

The code that allows us to process our data with K-Nearest Neighbors algorithm with default value is as follows. With the logr variable, we provided the evaluation according to our quality criteria 0 and 1. 
```sh
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train) #Train
pred_knn = knn.predict(X_test) #Predict
#Confusion_matrix
cm = confusion_matrix(y_test,pred_knn)
print("K-Nearest Neighbors")
print(cm)
s = (((98+130)/(1598*0.2)))
print("Succes Rate: ",s)
```

The confusion matrix used to evaluate the performance of the algorithm is as in Figure 12. The color shown in the diagonal shows the correct predicted values. When we calculate these results as a percentage, we achieve a success of 0.71.

|| 0| 1| 
| ------ | ------ | ------ |
|0|98|50|
|1|42|130|
- Figure 12:  K-Nearest Neighbors Algorithm Without Parameters Matrix Representation

With the Support Vector Classifier algorithm with default value, the code that allows us to process our data is as follows. rbf_svc variable in the algorithm we have created with our quality criteria 0 and 1 have been made according to the evaluation.

```sh
from sklearn.svm import SVC
rbf_svc = SVC(random_state=0 , probability = True )
rbf_svc.fit(X_train, y_train) #Train
pred_svc = rbf_svc.predict(X_test) #Predict
#Confusion_matrix
cm = confusion_matrix(y_test, pred_svc)
print("Support Vector Classifier")
print("           quality")
print("0”, “1”)
print(cm)
s = (((110+131)/(1598*0.2)))
print("Succes Rate: ",s)
```

The confusion matrix used to evaluate the performance of the algorithm is as in Figure 13. The color shown in the diagonal shows the correct predicted values. When we calculate these results as a percentage, we achieve a success of 0.75.

|| 0| 1| 
| ------ | ------ | ------ |
|0|110|38|
|1|41|131|
	- Figure 13: Support Vector Classifier Algorithm Without Parameters Matrix Representation

The code that allows us to process our data with the default Random Forest Classifier algorithm is as follows. Algorithm was created with ”rfc“ variable.
```sh
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( random_state=0 )
rfc.fit(X_train,y_train) #Train
pred_rfc = rfc.predict(X_test) #Predict
#Confusion_matrix
cm = confusion_matrix(y_test,pred_rfc)
print('Random Forest Classifier')
print("           quality")
print(“0”,”1”)
print(cm)
s = (((124+134)/(1598*0.2)))
print("Succes Rate: ",s)

```


The confusion matrix used to evaluate the performance of the algorithm is as in Figure 14. The color shown in the diagonal shows the correct predicted values. When we calculate these results as a percentage, we achieve a success of 0.80.

|| 0| 1| 
| ------ | ------ | ------ |
|0|124|24|
|1|38|134|
- Figure  14: Random Forest Classifier Matrix Without Parameters Display of Algorithm

·	Gaussian Naive Bayes	:	71.96495619524404
·	Logistic Regression	:	76.34543178973716
·	Decision Tree Classifier	:	73.52941176470587
·	Support Vector Classifier	:	75.40675844806007
·	Random Forest Classifier	:	80.72590738423028
·	K-Nearest Neighbors	:	71.33917396745932

Success rates of our algorithms without parameters;

To make these algorithms more successful, we added parameters to the algorithms. Gaussian Naive Bayes did not make any changes in each algorithm or produced a negative result. When we added parameters to the Logistic Regression, K-Nearest Neighbors, Decision Tree Classifier, Support Vector Classifier and Random Forest Classifier algorithms, we were able to achieve a positive change. We have recorded the best results in these algorithms with many parameters.

We have added the “random_state=None ,multi_class = "multinomial" ,solver = "lbfgs" ,C =1 ,tol = 2 ,max_iter =100” parameters to the Logistic Regression algorithm as in the following code block.

```sh
logr = LogisticRegression(random_state=None ,multi_class = "multinomial" ,solver = "lbfgs" ,C =1 ,tol = 2 ,max_iter =100 )
```

The success value of the algorithm that we obtained the matrix in Figure 15 was calculated as 0.76. The algorithm has an increase of 0.003 according to the algorithm without parameter. This result increased the success value for this algorithm. No better results were obtained with the parameters and parameter values. This value is considered to be the best result for the Logistic Regression algorithm for this data set and for our project.

|| 0| 1| 
| ------ | ------ | ------ |
|0|115|33|
|1|42|130|

- Figure 15  Matrix Representation of the  Logistic Regression Algorithm With Parameters

The other algorithm that we can get recovery is added to the K-Nearest Neighbors algorithm “n_neighbors=100 ,  weights = "distance" , algorithm="kd_tree", leaf_size=75 , p=4.5 , metric="minkowski” parameters are shown in the following code block.
```sh
knn = KNeighborsClassifier(n_neighbors=100 ,  weights = "distance" , algorithm="kd_tree", leaf_size=75 , p=4.5 , metric="minkowski")
```
The success value of the algorithm we obtained in Figure 16 is calculated as 0.76. An increase of 0.056 was achieved according to the algorithm's default settings. This result increased the success value for this algorithm. No better results were obtained with the parameters and parameter values. This value is considered to be the best result of the K-Nearest Neighbors algorithm for this data set and for our project.

 || 0| 1| 
| ------ | ------ | ------ |
|0|107|41|
|1|33|139|
- Figure 16: Matrix Representation of  K-Near Neighbors Algorithm With Parameters

We have added the “random_state=0 , criterion="gini" , splitter="best" , min_samples_leaf=4” parameters to the Decision Tree algorithm as in the following code block.

```sh
dtc = DecisionTreeClassifier(random_state=0 , criterion="gini" , splitter="best" , min_samples_leaf=4)
```
The success value of the algorithm that we obtained the matrix in Figure 17 was calculated as 0.74. The algorithm has an increase of 0.012 according to the default settings. This result increased the success value for this algorithm. No better results were obtained with the parameters and parameter values. This value is considered to be the best result for the Decision Tree algorithm for this data set and for our project.
 
 || 0| 1| 
| ------ | ------ | ------ |
|0|117|31|
|1|50|122|
- Figure 17:  Matrix Representation of the Decision Tree Algorithm With Parameters

We have added the “random_state=0, probability = True, kernel="rbf", C=0.3” parameters to the Support Vector Classifier algorithm as in the following code block.
```sh
rbf_svc = SVC(random_state=0, probability = True, kernel="rbf", C=0.3)
```

The success value of the algorithm that we obtained the matrix in Figure 18 was calculated as 0.76. The algorithm has an increase of 0.009 according to the default settings. This result increased the success value for this algorithm. No better results were obtained with the parameters and parameter values. This value is considered to be the best result for the Support Vector Classifier algorithm for this data set and for our project.

 || 0| 1| 
| ------ | ------ | ------ |
|0|114|34|
|1|42|130|
- Figure  18:  Matrix Representation of the Support Vector Classifier Algorithm With Parameters

We have added the “multi_class = "multinomial",  solver = "lbfgs", C = 0.01, tol = 0.7,  max_iter = 100” parameters to the Random Forest Classifier algorithm as in the following code block.
```sh
rbf_svc = SVC(random_state=0, probability = True, kernel="rbf", C=0.3)
```

The success value of the algorithm that we obtained the matrix in Figure 19 was calculated as 0.81. The algorithm has an increase of 0.003 according to the default settings. This result increased the success value for this algorithm. No better results were obtained with the parameters and parameter values. This value is considered to be the best result for the Random Forest Classifier algorithm for this data set and for our project.
|| 0| 1| 
| ------ | ------ | ------ |
|0|124|24|
|1|37|135|
 
- Figure 19  Matrix Representation of the Random Forest Classifier Algorithm With Parameters

The success rates of our algorithms after the parameters we applied to our algorithms;
          
![](https://user-images.githubusercontent.com/46966075/86848926-9768d700-c0b7-11ea-9f88-b6239b0a34b8.png) 

After using algorithms, we added another method. This method, known as Voting Classifier, is created by combining other algorithms. This method consists of two parts, soft and hard. At the beginning of the process we get our other algorithms.
```sh
clf2 = KNeighborsClassifier(n_neighbors=100 ,  weights = "distance" , algorithm="kd_tree", leaf_size=75 , p=4.5 , metric="minkowski")
clf3 = SVC(random_state=0 , probability = True , kernel="rbf" , C=0.3)
clf4 = RandomForestClassifier( random_state=0 ,n_estimators=12 , min_samples_split=2 )
clfs = [('knn', clf2),('SVC', clf3),("rf" , clf4) ]
for clf_tuple in clfs:
    clf_name, clf = clf_tuple
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print('Model:{} - Accuracy:{:.2f}%'.format(clf_name, acc*100)
```

After we get our argoritms, we produce value according to hard property. According to the hard property, the success value was found to be 0.77. The code block is below.

After hard property, we produce value according to soft property. According to the soft property, the success value was found to be 0.78. The code block is below.

After Voting Classifier, our highest success is 0.80. We can see that Random Forest Classifier algorithm gives the best result for the data set. 

Random Forest Classifier algorithm aims to increase the classification value by producing more than one decision tree during the classification process. The low deviation and low correlation allowed the Random Forest Classifier algorithm to achieve better results.
### RESULT

Nowadays, in order to create better quality products, it must be compared with pre-manufactured products. We have learned that datasets have been created for this comparison. The larger the data sets, the better results are obtained. This benchmarking process contains many criteria. The comparison of hundreds of manufactured products according to these criteria is very difficult without machine help. For this, we learned that machine learning is used as a result of our project.

We discovered that there are 11 different criteria affecting the quality of a wine. In line with these criteria, we learned how to process the data in order to find quality wine. We analyzed this data with the help of Python programming language. Before we started analyzing, we made our data set more usable. In this process, we have understood the importance of data editing in processing large data and the role of computer in these processes.

 We have learned that there are many different algorithms to analyze our data and we understand the logic of working with these algorithms. We analyzed the data we analyzed with these algorithms and found the best result algorithm. While implementing these algorithms, we have checked the accuracy of our algorithm by separating our data set into training and test data. We used various visualization methods in this accuracy control.
 
With the project, we have understood the use of Python language and classifying with machine learning. We have learned how to make sense from large data sets and use the data according to our interests. We understand the working principles of the algorithms we use. To achieve better results from these algorithms, we applied various parameters.
As a result, we have understood what many concepts such as big data, machine learning, data analysis, data processing, data visualization, test data, and training data mean and how much space they occupy in our lives. We have learned and used these concepts in detail.

### REFERENCES

[1] Belgiu, M., & Drăguţ, L. (2016). Random forest in remote sensing: A review of applications and future directions. ISPRS Journal of Photogrammetry and Remote Sensing, 114, 24-31.

[2] Deng, Z., Zhu, X., Cheng, D., Zong, M., & Zhang, S. (2016). Efficient kNN classification algorithm for big data. Neurocomputing, 195, 143-148.

[3] Dietterich, T. G. (2000, June). Ensemble methods in machine learning. In International workshop on multiple classifier systems(pp. 1-15). Springer, Berlin, Heidelberg.

[4] John Walker, S. (2014). Big data: A revolution that will transform how we live, work, and think.
[5] Menard, S. (2002). Applied logistic regression analysis (Vol. 106). Sage.

[6] Mitchell, T. M. (2005). Logistic Regression. Machine learning, 10, 701.

[7] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

[8] Russell, Stuart J., and Peter Norvig. Artificial intelligence: a modern approach. Malaysia; Pearson Education Limited, 2016
