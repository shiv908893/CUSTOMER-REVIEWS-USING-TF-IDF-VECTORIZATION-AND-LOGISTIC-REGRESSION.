#PERFORM SENTIMENT ANALYSIS ON A DATASET OF  CUSTOMER-REVIEWS-USING-TF-IDF-VECTORIZATION-AND-LOGISTIC-REGRESSION.

*COMPANY*: CODE IT SOLUTIONS

*NAME*: SHAILESH MOHAN SHUKLA

*INTERN* ID: CSIT7590

*DOMAIN*: MACHINE LEARNING

*MENTOR*: NEELA SANTOSH

*DURATION*: 4 WEEKS

*DESCRIPTION*:

To perform Sentiment Analysis (or text classification) on your dataset, we will follow a standard machine learning pipeline: Data Preprocessing, Feature Extraction (TF-IDF), Model Training (Logistic Regression), and Evaluation.
Since the uploaded file (anonymous-survey-responses.csv) contains survey answers rather than traditional "Positive/Negative" reviews, I have adapted the code to treat the "What's your interest in data science?" column as the text input and the "Just for fun, do you prefer dogs or cat?" column as the target category. This demonstrates the exact technical workflow you requested for sentiment analysis.
Performance Summary
•	Accuracy: ~55% (The model tends to predict the majority class "Dogs" because the input text is categorical and limited in this specific dataset).
•	Technique: TF-IDF Vectorization followed by a Logistic Regression classifier.
Explanation of Steps:
1.	Text Cleaning: We converted all text to lowercase and removed special symbols. This ensures that "Dogs" and "dogs" are treated identically.
2.	TF-IDF (Term Frequency-Inverse Document Frequency): This converts words into numbers by calculating how important a word is to a document in a collection. It helps the model ignore common filler words and focus on descriptive terms.
3.	Logistic Regression: A robust linear model used here for multi-class classification. It calculates the probability of each class and assigns the most likely one.
4.	Label Encoding: Since machine learning models require numeric targets, we mapped the category strings (Both, Cats, Dogs, Neither) into integers.

<img width="595" height="432" alt="Image" src="https://github.com/user-attachments/assets/af30cca5-3a24-42a8-bec6-3f6f02e4fa89" />
<img width="926" height="509" alt="Image" src="https://github.com/user-attachments/assets/a68cfad0-f991-47f8-8db4-0fa095c58e2e" />
