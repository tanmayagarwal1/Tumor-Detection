# Tumor-Detection
Past years have experienced increasing mortality rate due to lung cancer and thus it becomes crucial to predict whether the tumor has transformed to cancer or not, if the prediction is made at an early stage then many lives can be saved and accurate prediction also can help the doctors start their treatment.


With the rapid increase in population rate, the rate of diseases like cancer, chikungunya, cholera etc., are also increasing. Among all of them, cancer is becoming a common cause of death. Cancer can start almost anywhere in the human body, which is made up of trillions of cells. Normally, human cells grow and divide to form new cells as the body needs them. When cells grow older or become damaged, they die, and new cells take their place. When cancer cells develop, however, this orderly process breaks down.

Tumors are of two types benign and malignant where benign (non-cancerous) is the mass of cell which lack in ability to spread to other part of the body and malignant (cancerous) is the growth of cell which has ability to spread
in other part of body this spreading of infection is called metastasis.


To diagnose lung cancer various techniques are used like chest X-Ray, Computed Tomography (CT scan), MRI (magnetic resonance imaging) through which doctor can decide the location of tumor based on that treatments are given. Now it is important that the disease diagnose should be done in early stage so that many life’s can be saved.

This is where our project comes in clutch. By passing the data ob- tained from the computer tomography scan they can determine before hand whether the patient has malignant or benign tumor or not. This can help save a lot of lives and also prevent cases from becoming severe

## Aim and Motivation 
The primary objective of our project is to build a machine learning model which can classify whether there are traces of malignant or benign cancer given the parameters and values which can be calculated from a computer tomography scan. We ensure that the model uses robust machine learning algorithms as to be able to classify the given data with the highest precision

Tumors are of two types benign and malignant where benign (non-cancerous) is the mass of cell which lack in ability to spread to other part of the body and malignant (cancerous) is the growth of cell which has ability to spread
in other part of body this spreading of infection is called metastasis.

There is various type of cancer like Lung cancer, leukemia, and colon cancer etc. The incidence of lung cancer has significantly increased from the early 19th century. There is various cause of lung cancer like smoking, exposure to radon gas, secondhand smoking, and exposure to asbestos etc. Lung cancer is of two type small cell lung cancer (SCLC) and non small cell lung cancer (NSCLC). Non-small cell lung cancer is more common than SCLC and it generally grows and spreads more slowly.

To diagnose lung cancer various techniques are used like chest X-Ray, Computed Tomography (CT scan), MRI (magnetic resonance imaging) through which doctor can decide the location of tumor based on that treatments are given. Now it is important that the disease diagnose should be done in early stage so that many life’s can be saved.

## Technology Used 
1. Python v3.8 
2. Tensorflow v2.8 
3. Sci-kit Learn 
4. Jupyter Notebooks
5. Pandas Framework 
6. MatPlotLib 

## Visual Analysis 
We have used the matplotlib and pandas library to get a visual intuition of the dataset. We have formed a co-relation graph which gives us a heat map of how each attribute in our dataset is linked to each other. This gives us a measure of how much of which attributes affect out output classification the most which can help in future debugging

<p align = 'center'> <img width="694" alt="Screenshot 2021-06-23 at 7 48 53 PM" src="https://user-images.githubusercontent.com/81710149/123113323-15a2c280-d45c-11eb-82bc-c06e84fa294e.png">
</p>

As we observe the heat map of co-relations, the values will range from between -1 to +1. A value of 0 indicates no co-relation, a value of -1 indi- cates negative co-relation and a value of +1 indicates that the data is highly co-related

## Models 
The KNN has been implemented using the KNeighborsClassifier from the sklearn.neighbors library. We have also used the Euclidian distance function as cited in the previous sections
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=40, metric='euclidean')
knn.fit(scaled_X_train, y_train)
```

The logistic regression is implemented using the LogisticRegression class of the sklearn.linear-model model. In this we will be fitting our training and testing data
```python
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(X_train,y_train)
```

The Decision Tree is implemented using the DecisionTreeClassifier class from the sklearn-tree model. The training and testing data is fit to it as follows
```python
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
```

The random forest model is implemented using the RandomForestClassifier from the sklearn-ensemble model. The training and testing data is then fit to it as follows
```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
```


The support vector machines are implemented using the SVC class of the sklearn-svm model.
```python
from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train)
```

## Accuracy Validation 
Now we have concluded the training, testing and deployment of our models. After cross-validating the output of the models with the actual outputs from the dataset we have displayed the accuracy of each and every model

1. Random Forests :This has got an accuracy score of 95.9%
2. Logistic Regression : This has got an accuracy score of 95%
3. K Nearest Neighbours : This has got an accuracy score of 94%
4. Decision Tree : This has got an accuracy score of 94%
5. Support Vector Machines : This has got an accuracy score of 90%

