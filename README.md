# Liver_function_analysisimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing

 from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('/content/archive (1).zip')

data

data.shape

data.info()

data.head()

data.tail()

data.describe()

data["Gender"].value_counts()
 
data["Albumin"].value_counts()

# See the min, max, mean values
print('The highest protien was of:',data['Total_Protiens'].max())
print('The lowest protien was of:',data['Total_Protiens'].min())
print('The average protien in the data:',data['Total_Protiens'].mean())

import matplotlib.pyplot as plt

# Line plot
plt.plot(data['Albumin'])
plt.xlabel("Total_Protiens")
plt.ylabel("Aspartate_Aminotransferase")
plt.title("Line Plot")
plt.show()


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Dataset']==1]['Total_Protiens'].value_counts()

ax1.hist(data_len,color='green')
ax1.set_title('Having protiens')

data_len=data[data['Dataset']==0]['Total_Protiens'].value_counts()
ax2.hist(data_len,color='red')
ax2.set_title('NOT Having protiens')

fig.suptitle('Protien Levels')
plt.show()


-data.duplicated()

newdata=data.drop_duplicates()

newdata

data.isnull().sum() #checking for total null values

data[1:5]

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['Gender']
encoder = OneHotEncoder(sparse=False, drop='first')  # 'drop' parameter removes one of the one-hot encoded columns to avoid multicollinearity
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
data = pd.concat([data, encoded_cols], axis=1)
data.drop(categorical_cols, axis=1, inplace=True)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

data 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Assuming 'target_column' is the name of your target variable
X_train, X_test, y_train, y_test = train_test_split(data, data['Dataset'], test_size=0.2, random_state=42)


import pandas as pd

# Assuming 'df' is your DataFrame
feature_names = data.columns.tolist()


feature_names


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

# Assuming 'classification' is a variable containing the target column name
classification = 'Dataset'  # Replace with your actual target column name

# Select features (X) and target variable (y)
feature_columns = ['Age',
 'Total_Bilirubin',
 'Direct_Bilirubin',
 'Alkaline_Phosphotase',
 'Alamine_Aminotransferase',
 'Aspartate_Aminotransferase',
 'Total_Protiens',
 'Albumin',
 'Albumin_and_Globulin_Ratio',
 'Gender_Male']
X = data[feature_columns]
y = data[classification]

# Replace '\t?' with NaN
X.replace('\t?', np.nan, inplace=True)

# Convert columns to numeric (assuming that they are numeric features)
X = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
predictions = model.predict(test_X)

# Evaluate the model
accuracy = metrics.accuracy_score(predictions, test_Y)
print('The accuracy of the Logistic Regression model is:', accuracy)

# Display the classification report
report = classification_report(test_Y, predictions)
print("Classification Report:\n", report)


import matplotlib.pyplot as plt
import numpy as np

# Replace these values with your actual scores
precision = [0.79, 0.73]
recall = [0.89, 0.56]
f1_score = [0.83, 0.63]

labels = ['Class 0', 'Class 1']

# Plotting the bar chart
width = 0.2
x = np.arange(len(labels))

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Adding labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Logistic Regression Model Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.show()


# Create and fit the Linear Regression model
from sklearn.model_selection import train_test_split # training and testing data split
from sklearn import metrics # accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,classification_report # for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())

# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)
#Evaluate the model using various metrics
mse = mean_squared_error(test_Y,prediction)
rmse = mean_squared_error(test_Y,prediction,squared=False) #Caluclate the square root of mse
mae = mean_absolute_error(test_Y,prediction)
r_squared=r2_score(test_Y,prediction)

print('Mean squared error:',mse)
print('Root Mean squared error:',rmse)
print('Mean absolute error:',mae)
print('R-squared:',r_squared)
