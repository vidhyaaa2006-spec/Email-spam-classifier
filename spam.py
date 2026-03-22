import pandas as pd

# Step 1: Load dataset
try:
    data = pd.read_csv("spam.csv", encoding='latin-1')
except:
    # If dataset is tab-separated (SMSSpamCollection)
    data = pd.read_csv("spam.csv", sep='\t', names=['label', 'message'])

# Step 2: Clean dataset
if 'v1' in data.columns:
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

# Step 3: Convert labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 4: Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Step 5: Convert text → numbers
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Accuracy
accuracy = model.score(X_test_vec, y_test)
print(f"\n✅ Model Accuracy: {accuracy:.2f}\n")

# Step 8: Continuous input (FINAL FEATURE 🔥)
while True:
    msg = input("Enter a message (type 'exit' to stop): ")

    if msg.lower() == 'exit':
        print("👋 Exiting program...")
        break

    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)

    if result[0] == 1:
        print("📩 Spam ❌\n")
    else:
        print("📩 Not Spam ✅\n")