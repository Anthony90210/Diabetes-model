import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import pickle
from flask import Flask, render_template, request
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier


# create an array for column labels if it does not exist in the dataset
# cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
#         "Age", "Outcome"]

# set the outcome or class to int 0 0r 1 if not provided as so
# df["Outcome"] = (df["Outcome] == " ").astype(int)

df = pd.read_csv("Data/diabetes-dataset.csv")

# To handle missing data in our dataset using the mean statistical method
# Create an instance of SimpleImputer with strategy='mean'
imputer = SimpleImputer(strategy='mean')

# Identify columns with missing values
missing_columns = df.columns[df.isnull().any()].tolist()
# print(len(missing_columns))
# Impute missing values with the mean of each column after checking if there are missing values
if len(missing_columns) != 0:
    df[missing_columns] = imputer.fit_transform(df[missing_columns])

# use df.column to fetch the column names & run a for loop to plot a histogram for each label against the outcome value
# [:-1 implies ] to get a list of all the column except for the last item "outcome"
# for label in df.columns[:-1]:
#     # for outcome value == 1
#     # inside the dataframe get all values with outcome value 1, for each specific label
#     plt.hist(df[df["Outcome"] == 1][label], color='blue', label='Positive', alpha=0.7, density=True)
#     # for outcome value == 0
#     plt.hist(df[df["Outcome"] == 0][label], color='red', label='Negative', alpha=0.7, density=True)
#     plt.title(label)
#     plt.ylabel("probability")
#     plt.xlabel(label)
#     plt.legend()

#     plt.show())


'''
To split the dataset to - Train, validation and test 
    test_size = 0.2: 20% for Test and the remaining 80% for Training
    random_state: control data shuffling before split
We will also oversample the data to aggregate the values of the features and Outcome to match for a more accurate 
prediction
'''

X = df.drop(columns=['Outcome'])  # Features
y = df['Outcome']  # Target variable

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split Data
# Data is split into Training set (Features x and Outcome y) and Test (Features x and Outcome y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Perform oversampling
# take more of the less and keep sampling from there to increase the size of that data set so they match
# ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
# X_train, y_train = ros.fit_resample(X_train, y_train)


# print(len(y_train))
# print(sum(y_train == 1))
# print(sum(y_train == 0))


'''
K-nearest neighbors - classification
'''

# # Train Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Evaluate Model
y_pred = knn_model.predict(X_test)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", knn_accuracy)
#
# print(y_pred)
# print("---------------------------------")
# print(y_test.values)
#
# # Generate classification report
# print(classification_report(y_test, y_pred))
#
# # Convert accuracy to integer for plot
print('KNN model has an accuracy score of:', math.floor(knn_accuracy*100), '%')

'''
At number of neighbours = 3
Accuracy of this model is 84%
Precision: how many labeled are actually that labelled class = 85% and 82%  
Recall: how many out of those that are actually the class did we get = 91% and 73%

f1-score is the combination of the precision and Recall prediction value = 88% and 77%.
'''


'''
Logistic Regression
Here we are estimating the probability between our two outcomes 0 and 1.
 - Basically we are trying to fit our data to the Sigmoid function - 1/1+e^-x
 - When we only have one feature value (x) it is a simple Log Reg, 
    multiple Log Reg for multiple values of x (x1, x2, x3, ...xn). 
'''
# # Train Model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate Model
y_pred = lr_model.predict(X_test)

# Calculate accuracy
lr_accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", lr_accuracy)
#
# print(y_pred)
# print("---------------------------------")
# print(y_test.values)
#
# # Generate classification report
# print(classification_report(y_test, y_pred))
#
# # Convert accuracy to integer for plot
print('LR model has an accuracy score of:', math.floor(lr_accuracy*100), '%')


'''
Accuracy of this model is 79%

Precision: how many labeled are actually that labelled class = 79% and 80%  
Recall: how many out of those that are actually the class did we get = 92% and 57%

f1-score is the combination of the precision and Recall prediction value = 85% and 67%.
'''


'''
Support Vector Machine - SVM
So we want to determine the best line or Hyperplane that best differentiate or separate our two outcomes or classes.
- We want the line that provides the largest margin, between itself and the outcome points.
- So the data points that lie on the margin line, which help us find our divider are the SUPPORT VECTORS 

'''

# # Train Model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate Model
y_pred = svm_model.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", svm_accuracy)
#
# print(y_pred)
# print("---------------------------------")
# print(y_test.values)
#
# # Generate classification report
# print(classification_report(y_test, y_pred))
#
# # Convert accuracy to integer for plot
print('SVM model has an accuracy score of:', math.floor(svm_accuracy*100), '%')

'''
Accuracy of this model is 84%

Precision: how many labeled are actually that labelled class = 84% and 85%  
Recall: how many out of those that are actually the class did we get = 94% and 69%

f1-score is the combination of the precision and Recall prediction value = 88% and 77%.
'''

'''
Decision Tree
Decision trees seek to find the best split to subset data, and are typically trained through the Classification and 
    Regression Tree (CART).
- We use metrics such as Information gain, Gini impurity or Mean square error (MSE) to evaluate the quality of a split.

'''

# Train Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate Model
y_pred = dt_model.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", dt_accuracy)
#
# print(y_pred)
# print("---------------------------------")
# print(y_test.values)
#
# # Generate classification report
# print(classification_report(y_test, y_pred))
#
# # Convert accuracy to integer for plot
print('Decision Tree model has an accuracy score of:', math.floor(dt_accuracy*100), "%")

'''
Accuracy of this model is 98%

Precision: how many labeled are actually that labelled class = 100% and 95%  
Recall: how many out of those that are actually the class did we get = 97% and 100%

f1-score is the combination of the precision and Recall prediction value = 98% and 97%.
'''

'''
Random Forest
This combines the output of multiple decision trees to make predictions
'''

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
rf_model.fit(X_train, y_train)

# Evaluate Model
y_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", rf_accuracy)
#
# print(y_pred)
# print("---------------------------------")
# print(y_test.values)
#
# # Generate classification report
# print(classification_report(y_test, y_pred))
#
# # Convert accuracy to integer for plot
print('Random Forest model has an accuracy score of:', math.floor(rf_accuracy*100), '%')

'''
Accuracy of this model is 99%

Precision: how many labeled are actually that labelled class = 99% and 99%  
Recall: how many out of those that are actually the class did we get = 99% and 99%

f1-score is the combination of the precision and Recall prediction value = 99% and 99%.
'''


'''
# Combining the Models in Bagging with Voting Classifier:
Bagging (Bootstrap Aggregating):
Bagging involves training multiple instances of the same model on different subsets of the training data and then 
averaging the predictions for regression or taking a vote for classification.
-Random Forest is a popular implementation of bagging where decision trees are trained on random subsets of the training 
and then aggregated to make predictions.
-Bagging helps reduce over-fitting and variance by introducing randomness into the model.
'''
# Combine models using Voting Classifier
voting_model = VotingClassifier(estimators=[
    ('knn', knn_model),
    ('svm', svm_model),
    ('rf', rf_model),
    ('dt', dt_model),
    ('lr', lr_model)
], voting='hard')

voting_model.fit(X_train, y_train)
pickle.dump(voting_model, open('model.pkl', 'wb'))

# # Evaluate model
# y_pred = voting_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, y_pred)
print('Bagging model has an accuracy score of:', math.floor(bagging_accuracy*100), '%')
# print(classification_report(y_test, y_pred))

#
# print(y_pred)
# print("---------------------------------")
# print(y_test.values)  # convert panda series to np array
# print("Voting Classifier Accuracy:", accuracy)
# Accuracy of this model is 92%

'''
# Combining 5 Models in Boosting with Stacking Classifier:
Boosting:
Boosting involves sequentially training models where each subsequent model focuses on the examples that previous models 
have misclassified.
-AdaBoost (Adaptive Boosting) and Gradient Boosting are two popular boosting algorithms.
-Boosting helps reduce bias and improve model performance by iteratively correcting the errors made by previous models.
'''
# Combine models using Stacking Classifier
base_models = [
    ('rf1', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('rf2', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb1', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('gb2', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
]
stacking_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier())

stacking_model.fit(X_train, y_train)

# # Evaluate model
# y_pred = stacking_model.predict(X_test)
boosting_accuracy = accuracy_score(y_test, y_pred)
print('Boosting model has an accuracy score of:', math.floor(boosting_accuracy*100), '%')

# print(classification_report(y_test, y_pred))

#
# print(y_pred)
# print("---------------------------------")
# print(y_test.values)  # convert panda series to np array
#
# print("Stacking Classifier Accuracy:", accuracy)
# # Accuracy of this model is 97%


# -------------------------- VISUALISATION -------------------------------------
# store the accuracy scores of each model in a list
models = ['KNN', 'SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest']
# models = ['KNN', 'SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Bagging', 'Boosting']


def plt_accuracy_score():
    accuracies = [knn_accuracy, svm_accuracy, lr_accuracy, dt_accuracy, rf_accuracy]
    # accuracies = [knn_accuracy, svm_accuracy, lr_accuracy, dt_accuracy, rf_accuracy, bagging_accuracy, boosting_accuracy]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Models')
    plt.ylim(0, 1)  # Set the y-axis limits to [0, 1] for accuracy
    plt.show()


def plt_precision_score():
    # for precision
    # Assuming you have predictions (y_pred) and actual labels (y_true) for each model

    # Calculate precision score for KNN
    precision_knn_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_knn_1 = precision_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for SVM
    precision_svm_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_svm_1 = precision_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Random Forest
    precision_rf_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_rf_1 = precision_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Decision Tree
    precision_dt_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_dt_1 = precision_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Logistic Regression
    precision_lr_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_lr_1 = precision_score(y_test, y_pred, pos_label=1)

    # print(precision_lr_0)
    # print(precision_lr_1)

    precision_v_0 = [precision_lr_0, precision_svm_0, precision_rf_0, precision_dt_0, precision_lr_0]
    precision_v_1 = [precision_lr_1, precision_svm_1, precision_rf_1, precision_dt_1, precision_lr_1]

    x_axis = np.arange(len(models))

    plt.bar(x_axis - 0.2, precision_v_0, width=0.5, label='precision-0')
    plt.bar(x_axis + 0.2, precision_v_1, width=0.5, label='precision-1')

    # Xticks
    plt.xticks(x_axis, models)

    # labels
    plt.xlabel('Models', fontsize=11, fontweight='bold')
    plt.ylabel('precision', fontsize=11, fontweight='bold')
    plt.legend()
    plt.show()


def plt_recall_score():

    # for recall
    # Assuming you have predictions (y_pred) and actual labels (y_true) for each model

    # Calculate recall score for KNN
    recall_knn_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_knn_1 = recall_score(y_test, y_pred, pos_label=1)

    # Calculate recall score for SVM
    recall_svm_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_svm_1 = recall_score(y_test, y_pred, pos_label=1)

    # Calculate recall score for Random Forest
    recall_rf_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_rf_1 = recall_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Decision Tree
    recall_dt_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_dt_1 = recall_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Logistic Regression
    recall_lr_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_lr_1 = recall_score(y_test, y_pred, pos_label=1)

    # print(recall_lr_0)
    # print(recall_lr_1)

    recall_v_0 = [recall_lr_0, recall_svm_0, recall_rf_0, recall_dt_0, recall_lr_0]
    recall_v_1 = [recall_lr_1, recall_svm_1, recall_rf_1, recall_dt_1, recall_lr_1]

    x_axis = np.arange(len(models))

    plt.bar(x_axis - 0.2, recall_v_0, width=0.5, label='recall-0')
    plt.bar(x_axis + 0.2, recall_v_1, width=0.5, label='recall-1')

    # Xticks
    plt.xticks(x_axis, models)

    # labels
    plt.xlabel('Models', fontsize=11, fontweight='bold')
    plt.ylabel('recall', fontsize=11, fontweight='bold')
    plt.legend()
    plt.show()


def plt_f1_score():
    # for f1-score
    # Assuming you have predictions (y_pred) and actual labels (y_true) for each model

    # Calculate f1_score score for KNN
    f1_score_knn_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_score_knn_1 = f1_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for SVM
    f1_score_svm_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_score_svm_1 = f1_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Random Forest
    f1_score_rf_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_score_rf_1 = f1_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Decision Tree
    f1_score_dt_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_score_dt_1 = f1_score(y_test, y_pred, pos_label=1)

    # Calculate precision score for Logistic Regression
    f1_score_lr_0 = f1_score(y_test, y_pred, pos_label=0)
    f1_score_lr_1 = f1_score(y_test, y_pred, pos_label=1)

    # print(f1_score_lr_0)
    # print(f1_score_lr_1)

    f1_score_v_0 = [f1_score_lr_0, f1_score_svm_0, f1_score_rf_0, f1_score_dt_0, f1_score_lr_0]
    f1_score_v_1 = [f1_score_lr_1, f1_score_svm_1, f1_score_rf_1, f1_score_dt_1, f1_score_lr_1]

    x_axis = np.arange(len(models))

    plt.bar(x_axis - 0.2, f1_score_v_0, width=0.5, label='f1_score_0')
    plt.bar(x_axis + 0.2, f1_score_v_1, width=0.5, label='f1_score_1')

    # Xticks
    plt.xticks(x_axis, models)

    # labels
    plt.xlabel('Models', fontsize=11, fontweight='bold')
    plt.ylabel('f1_score', fontsize=11, fontweight='bold')
    plt.legend()
    plt.show()


def plt_roc():
    # Calculate ROC curve for KNN
    y_pred_knn = knn_model.predict(X_test)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn)
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    # Calculate ROC curve for SVM
    y_pred_svm = svm_model.predict(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)

    # Calculate ROC curve for Random Forest
    y_pred_rf = rf_model.predict(X_test)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Calculate ROC curve for Decision Tree
    y_pred_dt = dt_model.predict(X_test)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
    roc_auc_dt = auc(fpr_dt, tpr_dt)

    # Calculate ROC curve for Logistic Regression\
    y_pred_lr = lr_model.predict(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label=f'KNN (AUC = {roc_auc_knn:.2f})')
    plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
    plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot(fpr_dt, tpr_dt, color='orange', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
    plt.plot(fpr_lr, tpr_lr, color='purple', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def plt_conf_mat():
    # Confusion matrix of the diabetes dataset’s model
    y_pred_knn = knn_model.predict(X_test)

    y_pred_svm = svm_model.predict(X_test)

    y_pred_rf = rf_model.predict(X_test)

    y_pred_dt = dt_model.predict(X_test)

    y_pred_lr = lr_model.predict(X_test)

    y_pred_bag = voting_model.predict(X_test)

    y_pred_boost = stacking_model.predict(X_test)

    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - KNN')
    plt.show()

    # Confusion matrix for SVM
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - SVM')
    plt.show()

    # Confusion matrix for Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Random Forest')
    plt.show()

    # Confusion matrix for Decision Tree
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Decision Tree')
    plt.show()

    # Confusion matrix for Logistic Regression
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.show()

    # Confusion matrix for Bagging
    cm_bag = confusion_matrix(y_test, y_pred_bag)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Bagging')
    plt.show()

    # Confusion matrix for Boosting
    cm_boost = confusion_matrix(y_test, y_pred_boost)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Boosting')
    plt.show()


def plt_corr_mat():
    # Create the correlation matrix
    correlation_matrix = df.corr()

    # Print or visualize the correlation matrix
    print(correlation_matrix)

    # If you want to visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


# Uncomment any of these functions for visualisation

# plt_precision_score()
# plt_recall_score()
# plt_f1_score()
# plt_accuracy_score()
# plt_roc()
# plt_corr_mat()
# plt_conf_mat()

# WEBAPP
app = Flask(__name__)

# Save model using joblib
# joblib.dump(voting_model, 'model.joblib')


@app.route('/')
def home():
    return 'Welcome to the Diabetes Prediction Web App!'


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'dataset_file' not in request.files:
        return 'No file uploaded', 400

    # Get the uploaded file
    dataset_file = request.files['dataset_file']

    # Check if the file is a CSV file
    if dataset_file.filename.endswith('.csv'):
        render_template('index.html')

        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(dataset_file)

        # Convert DataFrame to NumPy array
        # data = df.values

        msg_data = {}
        for k in request.args.keys():
            val = request.args.get(k)
            msg_data[k] = val
        f = open("models/X_test.json", "r")

        X_test = json.load(f)
        f.close()
        all_cols = X_test

        input_df = pd.DataFrame(msg_data, columns=all_cols, index=[0])

        # save using pickle # Load your combined model
        model = joblib.load('models/model.joblib')
        pickle.dump(model, open('model.pkl', 'wb'))

        model = pickle.load(open(data, "rb"))
        arr_results = model.predict(input_df)
        diabetes_likelihood = ""
        if arr_results[0] == 0:
            diabetes_likelihood = "No"
        elif arr_results[0] == 1:
            diabetes_likelihood = "Yes"
        return diabetes_likelihood

    else:
        return 'Invalid file format. Please upload a CSV file.', 400


if __name__ == '__main__':
    app.run(debug=True)


