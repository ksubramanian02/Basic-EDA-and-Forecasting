#For Step 1
import pandas
#For Step 2
from sklearn import linear_model
#For Step 3
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## Data Pre-processing
dataset_2 = pandas.read_csv('nyc_bicycle_counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']                = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out data for testing

## Functions
def plot(x,y, x_label, y_label, title):
  plt.scatter(x,y, s=2)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  plt.show()

def stats(data):
  mean = np.mean(data)
  std = np.std(data)

  return mean, std

## Objective 1: Collection of Bridge Statistics 

mean_data = []
std_data = []
bridge_data = [dataset_2['Brooklyn Bridge'], dataset_2['Manhattan Bridge'],dataset_2['Queensboro Bridge'],dataset_2['Williamsburg Bridge']]

for i in bridge_data:
  mean,std = stats(i)
  mean_data.append(mean)
  std_data.append(std) 


## Objective 2: Temperature & Precipitation vs Bike Count (3-variable Linear Regression)

# Useful Functions
def normalized_train(x_train):
  mean = np.mean(x_train)
  std = np.std(x_train)
  normal_data = (x_train - mean) / std
  
  return normal_data, mean, std

def normalize_test(x_test, train_mean, train_std):
  normal_data = (x_test-train_mean)/train_std

  return normal_data

def get_lambda_range():
  lmbda = np.logspace(-1,3,51)

  return lmbda

def train_model(x,y,l):
  model = linear_model.Ridge(alpha=l,fit_intercept=True)
  model.fit(x,y)

  return model

def error(x,y,model):
  predicted = model.predict(x)
  mse = ((predicted-y)**2).mean()

  return mse

# Feature Selection
df_objective2 = dataset_2.copy()
remove_features = [["Date"],["Day"]]
low_temp = df_objective2['Low Temp'] #X1
high_temp = df_objective2['High Temp'] #X2
prec = df_objective2['Precipitation'] #X3
x = df_objective2[['Low Temp', "High Temp", "Precipitation"]] #X
y = df_objective2["Total"] #Y
y = np.array(y).reshape(-1,1)

# Normalize Data
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
[normal_x_train,train_mean, train_std] = normalized_train(X_train)
normal_x_test = normalize_test(X_test, train_mean, train_std)


# Explore Relationship (Uncomment to show explanatory plots)
#plot(low_temp,total,"Low Temp","Total Traffic","Title")    # Plot of Explanatory Vs Total
#plot(high_temp,total,"High Temp","Total Traffic","Title")    # Plot of Explanatory Vs Total
#plot(prec,y, "Precipitation","Total Traffic","Title")

# Method 1: Regular Multi-Linear Regression
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Intercept: ', model.intercept_)
print('Coefficients: ', model.coef_)
print("MAE:", metrics.mean_absolute_error(y_test,y_pred))
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
print("R-squared:", metrics.r2_score(y_test,y_pred))
#y_pred = model.predict(x)


'''
#Method 2: Multi-Variable Ridge Regression (Like HW7 regularize-cv)

# Find Best Value of Lambda
lmbda = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
MODEL = []
MSE = []
for l in lmbda:
  # Train the regression model using a regularization parameter of l
  model = train_model(X_train, y_train, l)
  # Evaluate the MSE on the test set
  mse = error(X_test, y_test, model)
  # Store the model and mse in lists for further processing
  MODEL.append(model)
  MSE.append(mse)
  print('MSE (lamba=', l,'):', mse)

# Plot Results
# plt.plot(lmbda, MSE)
# plt.xlabel('Lambda values', fontsize=10)
# plt.ylabel('MSE')
# plt.title("Graph of Lambda vs MSE to Determine Best Lambda Value")
# plt.show()
  
ind = MSE.index(min(MSE))
[lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]

print(f"\nBest lambda tested is {str(lmda_best)} which yields an MSE of {str(MSE_best)}")
'''

# #Method 3: Plot Relationship between Temperatures and Total Cyclists
low_temp_unique = []
high_temp_unique = []
low_total = []
high_total = []
low_temp_unique = df_objective2['Low Temp'].unique()
high_temp_unique = df_objective2['High Temp'].unique()

# Plot 1: Low vs Total
for uniqueTemp in low_temp_unique:
    common_list = []
    for index, row in df_objective2.iterrows():
        if row['Low Temp'] == uniqueTemp:
            common_list.append(row.iloc[9])  # Assuming 'Total' is in the 9th column
    low_total.append(np.mean(common_list))

#print("Unique Temperatures:", low_temp_unique)
#print("Average Total Cyclists for Each Temperature:", low_total)
plt.scatter(low_temp_unique, low_total, marker='o')
plt.xlabel('Temperature (Farenheit)')
plt.ylabel('Average TotalNumber of Cylists')
plt.title('Low Temperature vs. Total Cyclists')
plt.show()

# Plot 2: High vs Total
for uniqueTemp in high_temp_unique:
    common_list = []
    for index, row in df_objective2.iterrows():
        if row['High Temp'] == uniqueTemp:
            common_list.append(row.iloc[9])  # Assuming 'Total' is in the 9th column
    high_total.append(np.mean(common_list))

#print("Unique Temperatures:", high_temp_unique)
#print("Average Total for Each Temperature:", high_total)
fig2 = plt.scatter(high_temp_unique, high_total, marker='o', color = "orange")
plt.xlabel('Temperature (Farenheit)')
plt.ylabel('Average Total Number of Cyclists')
plt.title('High Temperature vs. Total Cyclists')
plt.show()