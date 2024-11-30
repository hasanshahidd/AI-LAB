import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

# Load data and handle missing values in critical columns
data = pd.read_csv("C:\\Users\\Admin\\Desktop\\data task.csv")
data = data.dropna(subset=['condition', 'bedrooms', 'bathrooms', 'sqft_living'])

# Features and target
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']]
y = data['condition']

# Impute missing values with the mean
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
X_train_scaled, X_test_scaled = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

# Train and evaluate SVM classifier
svm = SVC(kernel='linear', random_state=42).fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

# Print results
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
