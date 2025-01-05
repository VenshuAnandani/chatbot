import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the intents JSON file
with open("intents.json", "r") as file:
    data = json.load(file)

# Access the list of intents
intents = data["intents"]

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Preprocess the data
ints = []  # List of tags (labels)
patterns = []  # List of patterns (input examples)
for intent in intents:
    for pattern in intent['patterns']:
        ints.append(intent['intent'])
        patterns.append(pattern)

# Training the model
x = vectorizer.fit_transform(patterns)  # Transform input patterns into vectors
y = ints  # Target labels
clf.fit(vectorizer.fit_transform(patterns), ints)

# Save the trained vectorizer and classifier
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(clf, 'clf.joblib')
print("Model and vectorizer saved successfully.")
