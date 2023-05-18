# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split


# In this section, you can use a search engine to look for the functions that will help you implement the following steps

# Load dataset and show basic statistics
dataset = pd.read_csv('pulsar_stars.csv')

# 1. Show dataset size (dimensions)
print("----Dataset Size----")
print(dataset.size, "\n")

# 2. Show what column names exist for the 9 attributes in the dataset
print("----Column Names----")
print(dataset.columns, "\n")

# 3. Show the distribution of target_class column
print("----Distribution of target_class column----")
print(dataset['target_class'].value_counts(), "\n")

# 4. Show the percentage distribution of target_class column
print("----Percentage distribution of target_class----")
print(dataset['target_class'].value_counts(normalize=True) * 100, "\n\n")


#%%
# Separate predictor variables from the target variable (X and y as we did in the class)
X = dataset.iloc[:, :-1] # all columns except the last
Y = dataset.iloc[:, -1] # only the last column

#%%
# Create train and test splits for model development. Use the 80% and 20% split ratio
#splitting the train set and test set for X and Y (test size is 20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .2, random_state = 0)

#%%
# Standardize the features (Import StandardScaler here)
from sklearn.preprocessing import StandardScaler

#standardizing the the X data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#%%
# Below is the code to convert X_train and X_test into data frames for the next steps
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

#%%
# Train SVM with the following parameters.
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3
   
from sklearn.svm import SVC

#training the SVM with the specified parameters
classifier = SVC(kernel = 'rbf', C = 10, gamma = .3, random_state = 0 )
classifier.fit(X_train, Y_train)

#%%
# Test the above developed SVC on unseen pulsar dataset samples
from sklearn.metrics import confusion_matrix, accuracy_score
Y_pred = classifier.predict(X_test)

#%%
# compute and print accuracy score
print("----Accuracy Score----")
print(accuracy_score(Y_test, Y_pred) * 100, "\n")

#%%
# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
import pickle

#saving the svm model to a .sav file
with open('pulsar_svm_model.sav', 'wb') as f:
    pickle.dump(classifier, f)

#%%
# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
cm = [[]]
cm = confusion_matrix(Y_test, Y_pred)

print("----Confusion Matrix----")
print(cm, "\n")
# Below are the metrics for computing classification accuracy, precision, recall and specificity
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


#accuracy is correct predictions / all predictions
accuracy = (TP + TN) / (TP + TN + FP + FN) # accuracy score is already shown in the final output so not printing it

# Compute Precision and use the following line to print it
#precision is true positive prediction / all positive predictions
precision = (TP / (TP + FP)) * 100
print('Precision (%): {0:0.3f}'.format(precision))


# Compute Recall and use the following line to print it
# recall is true postiive predictions / actual positive predictions
recall = (TP / (TP + FN)) * 100
print('Recall or Sensitivity (%): {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
# specificity is true negative predictions / true negative and false positive predictions
specificity = (TN / (TN + FP)) * 100
print('Specificity (%): {0:0.3f}'.format(specificity))
