import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# # Train Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# # Train Model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# # Train Model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Train Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
rf_model.fit(X_train, y_train)

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

# pickle.dump(voting_model, open('model.pkl', 'wb'))
