### Required imports

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
```



### Initializing your model

```python
def __init__(self, kernel, tau=0.5):
        self.kernel = kernel
        self.tau = tau
```

To create an instance of your Lowess class, you need to first decide which kernel function and tau values you want. The base value of tau is a value value of 0.5.

Here are some kernel function examples:

```python
def tricubic(x):
  return np.where(np.abs(x)>1,0,(1-np.abs(x)**3)**3)

def Gaussian(x):
  return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))

def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2))

def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)
```

Note that you must define these functions **before** this Lowess class. 



### Fitting the model to your selected feature(s) and target

```python
def fit(self, x, y):
      	ss=StandardScaler()
        self.xtrain_=ss.fit_transform(x)
        self.yhat_ = y.reshape(-1,1)
```

Since this class's intended purpose was to allow for Lowess implementation, the *fit* function's x and y parameters are meant to be the training sets of your features (x) and target (y). As you'll see in a later function in the class, the class has cross-validation capabilities that can handle the splitting of your data into training and testing sets internally. While not necessary for models where there is only one feature being used, the *fit* function takes time to scale the feature data. This is important for Lowess with multiple features to prevent one feature from exerting much more influence over distance calculations within weights than another.



### Predicting target outcomes for the test set

```python
def predict(self,x_new,model=Ridge(alpha=0.01)):
        check_is_fitted(self)
    		self.xtest_=ss.transform(x_new)
        w = self.kernel(cdist(self.xtrain_, self.xtest_, metric='euclidean')/(2*tau))

        local_model=model

        if np.isscalar(self.xtest_):
    			local_model.fit(np.diag(w)@(self.xtrain_.reshape(-1,1)),np.diag(w)@(self.yhat_.reshape(-1,1)))
          yest_test = local_model.predict([[self.xtrain_]])[0][0]
        else:
          n = len(self.xtest_)
          yest_test = np.zeros(n)
          for i in range(n):
            local_model.fit(np.diag(w[:,i])@(self.xtrain_),np.diag(w[:,i])@(self.yhat_.reshape(-1,1)))
            yest_test[i] = local_model.predict(self.xtest_[i].reshape(1,-1))
        return yest_test
```

This is a very involved function, so I'll take the time to break this down step-by-step.

1. First, we must double-check that the model has been fitted to training data before proceeding. Otherwise, the *predict* function will raise an error.
2. Once we can confirm that it has been fitted, we must transform the test feature values to fit the same scale that the train data values were on. 
3. Our next step is to calculate the weights, represented by *w*. This can be done by taking the pairwise Euclidean distance between our train and test sets, which are then restricted in value by dividing by *tau*. *w* will be an *i X j* matrix, with *i* being the length of our train set and *j* being the length of our test set. Our kernel function will give higher weights to points close to each other, and smaller (or zero) weights to points far apart.
4. We want to be able to model these local relationships between points, so the *model* parameter establishes what regression technique we wish to use at this level.
5. The conditional statement simply accounts for the scenario in which you only have one testing point. The code would look slightly different in this case because that changes *w* from a 2-D to a 1-D array, but the code within the if and else blocks functionally do the same thing, just at different dimensionality.
6. It's important to mention that each column within a 2-D weights matrix represents the weight given to each training point with respect to the same testing point. For example, *w[: , 1]* represents each training point's weight with respect to the first testing point. With that understanding, we can deduce from the code that we are essentially fitting a new model for each testing point and using it to predict the target value of only that testing point, leading to *n* different regression models being fitted, where *n* represents the length of the test set.
7. We want to fit the local model to the weights multiplied by the actual feature and target values of the train set. However, first, we must diagonalize our weights for that specific point. This is because matrix (or matrix/vector) multiplication requires the length of the second dimension of our weights matrix and the length of our first dimension of the training matrix (or vector) to be the same. Without diagonalization, this wouldn't be possible, as we're only using a column at a time of the weights matrix. This means that our column's second dimension has a value of **1**, while the length of our training matrix/vector is simply how many points are in the train set, which should be **>1**. Diagonalization transforms the weights column into a square matrix with the number of rows and columns equal to the length of the training set.
8. Finally, we predict each testing point one at a time using the model that was fit using the weights associated with pairwise distances of the training data to that point. Once we have all of our predictions for each point, the *predict* function of the class returns an array of predicted target values for the test inputs.



### A simple K-fold cross-validation function

```python
def do_Kfold(self, X, y, k=10, random_state = 146):

    kf = KFold(n_splits=k, random_state = random_state, shuffle=True)

    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        self.fit(Xtrain,ytrain)
        test_scores.append(mse(self.predict(Xtest),ytest))

    return np.mean(test_scores)
```

This function runs standard K-fold cross-validation on the data. It splits the data into *k* pairs of training and testing data, which can then be plugged into the aforementioned *fit* and *predict* functions to get our predicted target values. To check how well the model performs, the list *test_scores* will contain the mean squared error of the predictions and actual target values for each train/test split. The function returns the mean of the MSEs. The lower this result is, the better the model performs.



### An example

```
import pandas as pd
cars = pd.read_csv("drive/My Drive/DATA 440 Capstone/data/mtcars.csv")
x=cars[['wt','hp']].values
y=cars['mpg'].values
lowess_model=Lowess(kernel=tricubic,tau=0.3)
print(lowess_model.do_Kfold(x,y,k=5,random_state=440))
```

