from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC

# Create SVM model
svm_model = SVC(kernel='linear', probability=True)  # You can choose different kernels

# Fit the model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_model.predict(X_test)

# Calculate and print the accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm * 100, "%")
