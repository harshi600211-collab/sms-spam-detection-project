
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the SMS Spam Collection dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Support common dataset column names
if {"v1", "v2"}.issubset(data.columns):
    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]
elif {"label", "message"}.issubset(data.columns):
    data = data[["label", "message"]]
else:
    raise ValueError(
        "Dataset must contain v1/v2 or label/message columns."
    )

data = data.dropna()
data["label"] = data["label"].str.lower().str.strip()
data = data[data["label"].isin(["ham", "spam"])]

# Convert labels into numbers: ham = 0, spam = 1
X = data["message"]
y = data["label"].map({"ham": 0, "spam": 1})

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a text-classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train and evaluate the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_test, predictions, target_names=["Ham", "Spam"]
))

# Test a sample message
message = input("\nEnter an SMS message: ")
prediction = model.predict([message])[0]

if prediction == 1:
    print("Result: SPAM message")
else:
    print("Result: HAM (not spam)")
