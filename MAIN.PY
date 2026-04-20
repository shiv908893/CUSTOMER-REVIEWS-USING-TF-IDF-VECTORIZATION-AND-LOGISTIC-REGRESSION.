# Import necessary libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
file_path = 'anonymous-survey-responses.csv'
df = pd.read_csv("C:\\Users\\Shailesh Shukla\\OneDrive\\Desktop\\anonymous-survey-responses.csv")

# Define our columns
text_column = "What's your interest in data science?"
label_column = "Just for fun, do you prefer dogs or cat?"

print("Dataset Preview:")
print(df[[text_column, label_column]].head())

# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
def clean_text(text):
    """Basic text cleaning: lowercase and remove special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def clean_label(label):
    """Clean the target labels (removing emojis/special characters)."""
    if not isinstance(label, str):
        return "Unknown"
    return re.sub(r'[^\w\s]', '', label).strip()

# Apply cleaning
df['cleaned_text'] = df[text_column].apply(clean_text)
df['cleaned_label'] = df[label_column].apply(clean_label)

# Encode labels
le = LabelEncoder()
df['target'] = le.fit_transform(df['cleaned_label'])

print("\nUnique labels identified:", le.classes_)

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# ---------------------------------------------------------
# Stratify=df['target'] ensures that the test set contains samples of every class
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['target'],
    test_size=0.2,
    random_state=42,
    stratify=df['target']
)

# ---------------------------------------------------------
# 4. TF-IDF VECTORIZATION
# ---------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---------------------------------------------------------
# 5. MODELING (LOGISTIC REGRESSION)
# ---------------------------------------------------------
# FIX: Removed 'multi_class' to support newer sklearn versions.
# Added 'class_weight' to help the model handle imbalanced classes.
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# ---------------------------------------------------------
# 6. EVALUATION
# ---------------------------------------------------------
print("\n--- MODEL EVALUATION ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
# FIX: Added zero_division=0 to suppress metric warnings
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# ---------------------------------------------------------
# 7. CONFUSION MATRIX VISUALIZATION
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix: Text-based Preference Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
