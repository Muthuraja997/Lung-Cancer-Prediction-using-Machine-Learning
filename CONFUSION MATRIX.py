import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- ANN ---
y_pred_ann = (predictions > 0.5).astype(int)
cm_ann = confusion_matrix(y_test, y_pred_ann)
print("Confusion Matrix (ANN):\n", cm_ann)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (ANN)')
plt.show()

# --- Random Forest ---
cm_rf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Random Forest):\n", cm_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# --- Decision Tree ---
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix (Decision Tree):\n", cm_dt)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Decision Tree)')
plt.show()

# --- SVM ---
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix (SVM):\n", cm_svm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (SVM)')
plt.show()

# --- KNN ---
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix (KNN):\n", cm_knn)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (KNN)')
plt.show()

# --- Logistic Regression ---
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
print("Confusion Matrix (Logistic Regression):\n", cm_logistic)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()
