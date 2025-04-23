from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Import scaler

from sklearn.model_selection import train_test_split
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler fitted on training data

# Create and fit the logistic regression model with increased iterations
logistic_model = LogisticRegression(max_iter=1000)  # Increase max_iter
logistic_model.fit(X_train_scaled, y_train)  # Fit on scaled data

# Predict on the scaled test set
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Calculate and print the accuracy
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("Logistic Regression Accuracy:", accuracy_logistic * 100, "%")