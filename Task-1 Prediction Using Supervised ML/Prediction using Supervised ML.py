#Task-1 Prediction Using Supervised ML (Level-Beginner)
                                                                         

   ## By : Alok Yadav 
    
  ## Data Science and Business Analytics intern at 

  ## Sparks Foundation


# Problem Statement
#Predict the percentage of an student based on the no. of study hours.
#- What will be predicted score if a student studies for 9.25 hrs/ day?

## It will contains
#- Importing libraries
#- importing dataset
#- Data Exploration
#- Data visualization
#- Data preparation
#- Spliting data into testing and traning 
#- Training model
#- Visualizing the model
#- Prediction for test data
#- Prediction for particular new data
#- Checking accuracy of the model



## Importing libraries

# Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

## Importing dataset

# import data set
url = "http://bit.ly/w-data"
df = pd.read_csv(url)

## Data Exploration

# top 10 rows of dataset
df.head(11)

#### There are two columns in this dataset - Hours: independent variable and Scores : dependent variable

# Get quick statistical information
df.describe()

# lets check dimension of data
df.shape

# all columns names 
df.columns    

## Data visualization

# using ggplot view 
plt.style.use('ggplot')

# Scatte plot 
fig = plt.figure(figsize=(8,6))
plt.scatter(x=df['Hours'],y=df["Scores"],color='blue')
plt.title('Study of Hours VS Scores of student')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

df.corr()

#### From above scatter plot and table we conclude that study hours and student scores are positive correlated
#### linear regression is best model for this type of dataset


## Data preparation

x = df.iloc[:,:-1].values

y = df.iloc[:,1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size =0.3)

## Training the model

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(x_train,y_train)

## Visualizing the model

# Plotting regression line
plt.figure(figsize=(8,6))
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,linear_model.predict(x_train))
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Study Hours VS Student Scores(Training dataset)")
plt.show()

# get y-intercept and coefficient of linear equation model
print("The intercept of  linear equation is", linear_model.intercept_)
print("The coefficient of  linear equation is", linear_model.coef_)

#### Linear Equation is given as :
#### Scores = 2.549610 + 9.6380 * (Hours)



### Above equation says one unit increase in  Hours associated with 9.6380 unit change in Scores.

## Prediction for test data

y_predict = linear_model.predict(x_test)

y_predict

y_test

# Comparing the Actual vs Predicted 
compare = pd.DataFrame({"Actual":y_test,"Predicted":y_predict})
compare

# Actual vs Predicted plot
plt.scatter(y_test,y_predict)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual VS Predicted")
plt.show()

### Actual and Predicted scatter plot are linearly correlated thats means our model is predicting with good accuracy

## Accuracy of model

# Mean Absolute Error of model
from sklearn import metrics
print("Mean Absolute Error :",metrics.mean_absolute_error(y_test,y_predict))

# Checking Accuracy of model
print("R Square value of the model is: ",metrics.r2_score(y_test,y_predict))

## Prediction for new data

# we can predict new data
data = np.array(9.25)
data = data.reshape(-1,1)
pred = linear_model.predict(data)
print("if any Student gives 9.25 an hours for study it will score{}. marks".format(pred))

## Conclusion

### Hence model predicted If any student gives 9.25 an hours for study it will score 91.7017320 marks in exam.

                                                      
                                             # Task Completed

