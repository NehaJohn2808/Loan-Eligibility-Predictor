import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("loan_data.csv")

# Convert categorical values to numbers
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

# Remove missing values
data = data.dropna()

# Features and Target
X = data[['Gender', 'Married', 'ApplicantIncome']]
y = data['Loan_Status']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# New customer data
new_customer = pd.DataFrame({
    'Gender': [1],          # Male = 1, Female = 0
    'Married': [1],         # Yes = 1, No = 0
    'ApplicantIncome': [5000]
})

# Predict loan eligibility
result = model.predict(new_customer)

if result[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")