import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Loads Dataset
df = pd.read_csv("phishing_email.csv")

TEXT_COL = "text_combined"
LABEL_COL = "label"

X = df[TEXT_COL]
y = df[LABEL_COL]

# Train-tests split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorizes text
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Prints out metrics and results
y_pred = model.predict(X_test_vec)
print("Baseline Model Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))