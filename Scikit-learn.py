# Spam Detection with Scikit-learn

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 2: Load Dataset
# You can download the dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Assuming the file name is 'spam.csv'
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Preprocess Data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 5: Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Predict custom message
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)
    return "Spam" if pred[0] == 1 else "Ham"

# Test
print("\nSample Test: 'Congratulations! You have won a $1000 gift card.' →", predict_message("Congratulations! You have won a $1000 gift card."))
