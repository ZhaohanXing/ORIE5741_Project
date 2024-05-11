import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import statsmodels.api as sm

# Import the dataset
data = pd.read_csv('D:/T2DM_MissingFilled_FINALVERSION.csv')

# Recoding the 'DM' variable for a binary classification (1 for diabetes, 0 for no diabetes)
data['DM'] = data['DM'].replace({1: 1, 2: 0, 3: 0, 9: 0})

# Perform one-hot encoding on categorical variables
# Remove 'PA' from the list of columns to be encoded
categorical_cols = ['Race', 'Menopause']
data = pd.get_dummies(data, columns=categorical_cols)

# Remove unwanted features from the data
columns_to_drop = ['PA', 'Alcohol', 'Menopause', 'Systolic']
for col in columns_to_drop:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)


# Prepare features and target variable
X = data.drop(['ID', 'DM'], axis=1)
y = data['DM']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model with class weight balanced
lr_model = LogisticRegression(solver='liblinear', class_weight='balanced')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Display results for Logistic Regression
print('Logistic Regression Model with Balanced Class Weights:')
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy of Logistic Regression: {accuracy_lr:.2f}")

# Specificity for Logistic Regression
tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, y_pred_lr).ravel()
specificity_lr = tn_lr / (tn_lr + fp_lr)
print(f'Logistic Regression Specificity: {specificity_lr:.2f}')



# Plotting ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of the Logistic Regression Model')
plt.legend(loc="lower right")
plt.show()


# KNN Model - Cross-Validation for finding the best 'k'
k_range = range(1, 100)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = k_scores.index(max(k_scores)) + 1
print("Best K value:", best_k)

k = best_k
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Display results for KNN
print(f'KNN Model with {k} Neighbors:')
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of KNN: {accuracy_knn:.2f}")

# Specificity for KNN
tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(y_test, y_pred_knn).ravel()
specificity_knn = tn_knn / (tn_knn + fp_knn)
print(f'KNN Specificity: {specificity_knn:.2f}')


# Predict probabilities for ROC curve
y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]  # probabilities for the positive class

# Plotting ROC Curve
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_pred_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure()
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for KNN')
plt.legend(loc="lower right")
plt.show()


# Visualization - K-Value vs. Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('Number of Neighbors k')
plt.ylabel('Cross-Validated Accuracy')
plt.title('K-Value vs. Accuracy')
plt.show()

# Decision Boundary Visualization
pca = PCA(n_components=2)
X_train2D = pca.fit_transform(X_train)
X_test2D = pca.transform(X_test)

knn_model_2D = KNeighborsClassifier(n_neighbors=k)
knn_model_2D.fit(X_train2D, y_train)

plt.figure(figsize=(10, 6))
plot_decision_regions(X_train2D, y_train.to_numpy(), clf=knn_model_2D, legend=2)
plt.title(f'KNN Decision Boundaries with k={k}')
plt.show()

# Confusion Matrix Visualization for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for KNN')
plt.show()

