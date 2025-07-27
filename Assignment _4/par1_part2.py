# part 1
# task 1

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = fetch_california_housing(as_frame=True)
X = data.data[['AveRooms']]
y = data.target

print(X.shape,y.shape)
# (20640, 1) (20640,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

ypred=model.predict(X_test)
print(ypred)
# [1.97653709 2.04156313 1.96003112 ... 2.21029436 2.06072485 1.96092798]

plt.scatter(X_test, y_test, color='blue',  label='Actual values')
plt.plot(X_test, ypred, color='red', label='Predicted values')
plt.title('California Housing Price Prediction')
plt.xlabel('Average  Rooms')
plt.ylabel('House price')
plt.legend()
plt.grid()
plt.show()

# task 2

from sklearn.metrics import mean_squared_error, r2_score

X_multi = data.data #all features
y_multi = data.target
print(X_multi.shape,y_multi.shape)
# (20640, 8) (20640,)

# Split the dataset into training and testing sets
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Train a multiple linear regression model
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

# Predicting values
y_pred_multi = model_multi.predict(X_test_multi)

# Evaluating the model
r_squared = r2_score(y_test_multi, y_pred_multi)
mse = mean_squared_error(y_test_multi, y_pred_multi)
rmse = np.sqrt(mse)
# Displaying the results
print(f'R-squared: {r_squared}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print('Coefficients:', model_multi.coef_)
# R-squared: 0.575787706032451
# Mean Squared Error: 0.5558915986952442
# Root Mean Squared Error: 0.7455813830127763
# Coefficients: [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01
#  -2.02962058e-06 -3.52631849e-03 -4.19792487e-01 -4.33708065e-01]

# task 3

from sklearn.preprocessing import StandardScaler

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_multi)
X_test_scaled = scaler.transform(X_test_multi)
# Train a multiple linear regression model on scaled data
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train_multi)

r_squared_before_scaling = r2_score(y_test_multi, y_pred_multi)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r_squared_after_scaling = r2_score(y_test_multi, y_pred_scaled)

print("\n--- Before Scaling ---")
print(f"R-squared: {r_squared_before_scaling:.4f}")
print("\n--- After Scaling ---")
print(f"R-squared: {r_squared_after_scaling:.4f}")
# --- Before Scaling ---
# R-squared: 0.5758

# --- After Scaling ---
# R-squared: 0.5758


# task 4

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

print(df)

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",)
plt.title('Feature Correlation Heatmap')
plt.show()

target_correlation = corr_matrix["Target"].drop("Target").sort_values(ascending=False)
print("\nStrongest correlations with Target:\n", target_correlation)

# Strongest correlations with Target:
#  MedInc        0.688075
# AveRooms      0.151948
# HouseAge      0.105623
# AveOccup     -0.023737
# Population   -0.024650
# Longitude    -0.045967
# AveBedrms    -0.046701
# Latitude     -0.144160
# Name: Target, dtype: float64


# part 2
# task 5

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)
print("ROC AUC Score:", roc_auc)

# Accuracy: 0.956140350877193
# Confusion Matrix:
#  [[39  4]
#  [ 1 70]]

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.97      0.91      0.94        43
#            1       0.95      0.99      0.97        71

#     accuracy                           0.96       114
#    macro avg       0.96      0.95      0.95       114
# weighted avg       0.96      0.96      0.96       114

# ROC AUC Score: 0.9977071732721913

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Breast Cancer Classification')
plt.legend()
plt.grid()
plt.show()


# task 6

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]
print(y_probs)

# [8.78417737e-01 3.04735433e-08 1.59969809e-03 9.98775726e-01
#  9.99858607e-01 2.00253498e-10 6.15347950e-11 1.92820681e-02
#  9.84264981e-01 9.94311003e-01 9.29270359e-01 8.09446888e-04
#  9.88907827e-01 1.76075741e-01 9.99212570e-01 1.74968272e-03
#  9.98803857e-01 9.99987939e-01 9.99999387e-01 7.22142596e-07
#  8.27778678e-01 9.92295256e-01 6.21088494e-09 9.99662491e-01
#  9.96530414e-01 9.99758501e-01 9.98974884e-01 9.98978180e-01
#  9.95311622e-01 7.41846222e-09 9.98535358e-01 9.99762181e-01
#  9.99107461e-01 9.86395780e-01 9.99883544e-01 9.99050996e-01
#  2.07437025e-03 9.98986277e-01 1.00852442e-05 7.18898594e-01
#  9.99782212e-01 6.22986310e-04 9.99539044e-01 9.96009202e-01
#  9.99756991e-01 9.82900274e-01 9.99967920e-01 9.99522850e-01
#  9.59469632e-01 9.98578390e-01 9.47005926e-05 2.91077079e-09
#  8.61050080e-01 9.99608713e-01 9.99399476e-01 9.86139467e-01
#  9.99866373e-01 3.78444071e-12 7.26690273e-01 9.99943047e-01
#  9.85647935e-01 7.80096190e-07 1.64719332e-09 9.19484277e-01
#  9.97331188e-01 8.96443476e-01 9.12439134e-06 7.34825870e-10
#  9.99357437e-01 9.85727734e-01 9.41558531e-03 4.14902228e-05
#  9.97635031e-01 6.86166239e-02 9.99902514e-01 9.97809013e-01
#  9.29933641e-01 5.61718297e-01 9.99851371e-01 9.98745165e-01
#  5.78491230e-04 9.99481448e-01 5.35081611e-01 4.33610514e-10
#  1.03246371e-03 1.27086441e-02 5.37222217e-04 3.21260663e-07
#  9.97393805e-01 9.97026414e-01 9.92644119e-01 9.00259568e-01
#  8.50315591e-01 9.99038484e-01 9.99734513e-01 9.99882894e-01
#  1.70963034e-07 3.13708069e-07 9.99970949e-01 7.66165052e-05
#  8.36342464e-05 9.99997082e-01 2.08546970e-08 1.96390787e-03
#  9.71775420e-01 9.56422777e-01 9.90317642e-01 1.05679697e-14
#  9.78023363e-01 8.77523834e-01 4.02287484e-04 9.99245062e-01
#  8.25445696e-02 7.79848505e-25]

thresholds = [0.3, 0.5, 0.7]

for t in thresholds:
    y_pred_thresh = (y_probs >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    print(f"\nThreshold: {t}")
    print("Confusion Matrix:\n", cm)
    print("F1 Score:", round(f1, 3))

fpr, tpr, thr = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# Threshold: 0.3
# Confusion Matrix:
#  [[39  4]
#  [ 1 70]]
# F1 Score: 0.966

# Threshold: 0.5
# Confusion Matrix:
#  [[39  4]
#  [ 1 70]]
# F1 Score: 0.966

# Threshold: 0.7
# Confusion Matrix:
#  [[41  2]
#  [ 1 70]]
# F1 Score: 0.979

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Threshold Tuning')
plt.legend()
plt.show()