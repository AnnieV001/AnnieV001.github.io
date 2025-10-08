---
layout: post
title: "Titanic Logistic Regression: From Notebook to Blog"
date: 2025-10-08
tags: [python, pandas, scikit-learn, logistic-regression, tutorial, titanic]
# Optional fields for Bay / Jekyll:
# author: "Annie Vo"
# description: "Building a Titanic survival classifier in Python—data prep, logistic regression, evaluation, and feature importance."
# image: /assets/img/your-cover.jpg
# permalink: /blog/titanic-logistic-regression-notebook-to-blog/
---
# Overview

This post converts my Jupyter notebook into a readable walkthrough showing how I trained a **logistic regression** classifier on the classic **Titanic** dataset. We’ll cover data preparation, avoiding leakage, model training, evaluation, and feature importance, then run the model on the test set and try a small feature engineering exercise.

> **Dataset:** Kaggle Titanic (train/test).  
> **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit‑learn.

---

## 1) Imports & Data Load

```python
# import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Create DataFrame
data = pd.read_csv('train.csv')

# Display the DataFrame
data.head()

# creating a backup copy of the data 
data_original = data.copy()
```
---

## 2) Imputing `Age` (without leakage)

**Why not use `Survived` to impute `Age`?**  
Using the target (`Survived`) for imputing predictors causes **data leakage**—information about the label leaks into the features, inflating performance and hurting generalization.

We impute `Age` with the **mean `Age` within each (`Sex`, `Pclass`) group**:

```python
# Populating null Age values with the average age by Sex and Pclass
data['Age'] = (
    data
    .groupby(['Sex', 'Pclass'], group_keys=False)['Age']
    .apply(lambda s: s.fillna(s.mean()))
)
```

### (Optional) Visual check
```python
fig, axes = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
sns.histplot(data_original['Age'], ax=axes[0], kde=True)
axes[0].set_title("Before imputation")

sns.histplot(data['Age'], kde=True, ax=axes[1])
axes[1].set_title("After imputation")
plt.show()
```

---

## 3) Categorical encoding

We convert categories to numeric with one‑hot encoding:

```python
# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
```

**Q:** *What did `pd.get_dummies` do?*  
**A:** It converts categorical variables to dummy/indicator variables.

---

## 4) Feature/Target split

```python
# Define features we will use for the model
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]

# define the target variable 
y = data['Survived']
```

**Why split?**  
So the model knows what to predict (target) and which inputs to use (features).

---

## 5) Train/Test split

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Why split the data?**  
To evaluate generalization on unseen data.

---

## 6) Train a Logistic Regression model

```python
# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

> **About `max_iter`:** It sets the maximum optimization steps. If the solver
> doesn’t converge within this limit, increase it or consider scaling features / changing solvers.

**Q:** *Why use logistic regression here?*  
**A:** Because the task is a **binary classification** (survived vs not). Logistic regression outputs probabilities suitable for two-class problems and is simple, strong, and interpretable.

---

## 7) Evaluate

```python
# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:\n', conf_matrix)
```

Example output from my run:
```
Accuracy: 0.8556
Confusion Matrix:
[[46  8]
 [ 5 31]]
```

> **Confusion Matrix** summarizes correct/incorrect predictions.  
> **Accuracy** is (correct predictions) / (all predictions).

---

## 8) Feature importance (via coefficients)

```python
# Calculate feature importance
feature_importance = model.coef_[0]

# Wrap in a DataFrame and plot
importance_df = (
    pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    .sort_values(by='Importance', ascending=False)
)
importance_df
```

```python
plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

**Reading coefficients**  
- **Positive**: higher values push prediction toward **survived = 1**.  
- **Negative**: higher values push toward **survived = 0**.

---

## 9) Apply the pipeline to **test** data

We must repeat the same preprocessing steps used for training.

```python
# Import new test data
test_data = pd.read_csv('test.csv')

# Impute Age the same way
test_data['Age'] = (
    test_data
    .groupby(['Sex', 'Pclass'], group_keys=False)['Age']
    .apply(lambda s: s.fillna(s.mean()))
)

# Check nulls
test_data.isnull().sum()

# Impute missing Fare with group mean by (Sex, Pclass)
test_data['Fare'] = (
    test_data
    .groupby(['Sex', 'Pclass'], group_keys=False)['Fare']
    .apply(lambda s: s.fillna(s.mean()))
)

# One‑hot encode like train
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Align columns with training features
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Predict on the new test data
test_predictions = model.predict(test_data)

# Attach predictions
test_data['Survived_predicated'] = test_predictions

# Class counts
test_data['Survived_predicated'].value_counts()
```

Example class balance from my run:
```
0    260
1    158
Name: Survived_predicated, dtype: int64
```

---

## 10) Mini exercise: `FamilySize` feature

Let’s engineer a simple `FamilySize = SibSp + Parch` and rebuild the model.

```python
# Feature engineering
data['FamilySize'] = data['SibSp'] + data['Parch']

features = ['Pclass', 'Age', 'FamilySize', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
target = 'Survived'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:\n', conf_matrix)

# Coefficients
feature_importance = model.coef_[0]
importance_df = (
    pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    .sort_values(by='Importance', ascending=False)
)
importance_df
```

Example output from my run:
```
Accuracy: 0.8667
Confusion Matrix:
[[47  7]
 [ 5 31]]
```

---

## Wrap‑up

- Avoid leakage: never use the target (`Survived`) to impute or engineer features.  
- Keep preprocessing consistent across train and test.  
- Logistic regression is a solid, interpretable baseline for binary classification.  
- Inspect coefficients to understand which features matter most.

If you try other features (titles from names, cabin deck, ticket groupings) and/or scaling, you can often squeeze out a bit more performance.
