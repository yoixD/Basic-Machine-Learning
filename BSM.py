import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import os


#Dictionary 
training_log = {
    'model_type': '',
    'mse': np.nan,     # Chỉ dùng cho Linear
    'r2_score': np.nan, # Chỉ dùng cho Linear
    'accuracy': np.nan # Chỉ dùng cho Logistic
}

# DATA
# read file csv

# get file data.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data.csv")
log_path = os.path.join(script_dir, "training_history.csv")

# read file
df = pd.read_csv(file_path)

# PREPROCESSING

df.columns = ['index', 'salary', 'exp']

# delete data that is not needed to analysis
df.drop

# handle missing
df['exp'] = df['exp'].replace('', int('1'))
df['salary'] = df['salary'].replace('', float('0'))

# convert data types
df['exp'] = df['exp'].astype(int)
df['salary'] = df['salary'].astype(float)

# add junior column
df['junior'] = df['exp'].apply(lambda x: 1 if x >= 5 else 0)

# handle invalid data
df.dropna(inplace=True)

#print(df.head())

# initialize x, y (track datashape)
x= np.array(df['exp']).reshape(-1,1)
y= np.array(df['salary']).reshape(-1,1)

# [TRACKING 1]
print(x.shape, y.shape) 

# choose model
which_model= input("Which model do you want to use? (0 for linear / 1 forlogictis regression): ")

if (int(which_model)==0) :
 # MODEL
 print("You chose Linear Regression")
 training_log['model_type'] = 'Linear Regression'
 
 # TRAINING MODEL
 
 # DATA SPLITTING of linear regression
 x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

 # standard scaling
 scaler = StandardScaler()
 x_std = scaler.fit_transform(x_train)
 x_test_std = scaler.transform(x_test)


 # creating the model
 model=LinearRegression()
 model.fit(x_train, y_train)
 y_pred= model.predict(x_test)

 # EVALUATING

 # Compare predictions vs actuals
 print("Predictions vs Actuals: ")
 print(y_pred[:5])
 print(y_test[:5])

 # evaluating coeficient
 model.score(x_test_std, y_test)

 print(f"Model coefficient: {model.coef_}")
 print(f"Model intercept: {model.intercept_}")

 # evaluating MSE
 MSE= mean_squared_error(y_test, y_pred)
 print(f"Mean Squared Error: {MSE}")

 # evaluating accuracy
 print(f"Model R^2 score: {model.score(x_test, y_test)}")  

 #Store results from multiple training runs
 training_log['mse'] = MSE
 training_log['r2_score'] = model.score(x_test, y_test)

 # sumarize best configuration

 # plotting

 # sorting
 sorted_indices = np.argsort(x_test.flatten())
 x_test_sorted = x_test[sorted_indices]
 y_pred_sorted = y_pred[sorted_indices]

 plt.scatter(x_test, y_test, color = 'blue', label = 'Actual')
 plt.plot(x_test_sorted, y_pred_sorted, linewidth= 2, color = 'cyan', label = 'Predict (linear)')
 plt.title("Linear Regression Model")
 plt.xlabel("Experience (years)")
 plt.ylabel("Salary")
 plt.legend()
 plt.show()

else :
 print("You chose Logistic Regression")
 training_log['model_type'] = 'Logistic Regression'

 #  MODEL
 y_LoRe= np.array(df['junior']).reshape(-1,1)
 x_train, x_test, y_train, y_test = train_test_split(x, y_LoRe.ravel(), test_size=0.2, random_state=42)

 # creating the model
 model=LogisticRegression()
 model.fit(x_train, y_train)
 y_pred= model.predict(x_test)

 #  EVALUATING
 cl_report = classification_report(y_test, y_pred)
 print(cl_report)

 # sumarize best configuration
 print("Confusion Matrix: ")
 print(confusion_matrix(y_test, y_pred))
 acc_score = model.score(x_test, y_test)
 training_log['accuracy'] = acc_score
 training_log['r2_score'] = np.nan

 # plotting

 # create range x 
 x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    
 # predict probabilities
 y_prob = model.predict_proba(x_range)[:, 1]
    
 # plot sigmoid
 plt.plot(x_range, y_prob, color='red', linewidth=2, label='Predict (Sigmoid)')

 # print shape of x_train, x_test 
 print("Shapes of x_train and x_test:")
 print(x_train.shape, x_test.shape)

 # scatter plot with jitter
 jitter_x = df['exp'] + np.random.normal(0, 0.1, size=len(df))
 jitter_y = df['junior'] + np.random.normal(0, 0.02, size=len(df))

 plt.title("Logistic Regression Model")
 plt.xlabel("Experience (years)")
 plt.ylabel("Probability")
 plt.scatter(jitter_x, jitter_y, color='blue', label='Actual', alpha=0.5)
 plt.legend()
 plt.show()


# Store training results to CSV file
print("-" * 30)
print("Saving training results...")

log_df = pd.DataFrame([training_log])
file_exists = os.path.isfile(log_path)
log_df.to_csv(log_path, mode='a', header=not file_exists, index=False)
print(f"Results saved to: {log_path}")