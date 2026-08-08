# 🎫 Auto Email / Ticket Categorizer

An NLP-based supervised machine learning project that automatically categorizes incoming support tickets into Billing, Technical, HR, or General departments.

## 🚀 Features

- Text cleaning and preprocessing
- Subject and body text extraction
- TF-IDF feature extraction
- Logistic Regression classifier
- Hyperparameter tuning using GridSearchCV
- Train/test evaluation
- Accuracy, Precision, Recall and F1-score
- Confusion Matrix
- Confidence score
- Human-review fallback for low-confidence tickets
- Urgent/Normal priority detection
- Real-time ticket classification
- Streamlit web interface

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- GridSearchCV
- Streamlit

## 📂 Project Structure

```text
ticket-classifier/
│
├── ticket_classifier.py
├── app.py
├── requirements.txt
├── README.md

🧠 Machine Learning Approach

This project uses Supervised Learning because the training dataset contains predefined department labels.

Support Ticket
      ↓
Text Cleaning
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Category Prediction
      ↓
Confidence + Priority
      ↓
Auto Assignment / Human Review
📌 Categories
Category	Example
BILLING	Payment charged twice
TECHNICAL	API error
HR	Leave balance query
GENERAL	General product question
⚙️ Installation

Install the required Python libraries:

pip install -r requirements.txt
▶️ Run Python Model
python ticket_classifier.py
🌐 Run Streamlit Application
streamlit run app.py

The Streamlit application allows users to enter a new support ticket and instantly receive:

Predicted category
Confidence percentage
Priority
Auto-assignment status
Human-review status
📊 Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Hyperparameters are optimized using GridSearchCV to improve model performance.

🔍 Edge Case Handling

The system validates incoming tickets before prediction.

Invalid or very short tickets are rejected.

If model confidence is below the defined threshold, the ticket is marked as:

NEEDS HUMAN REVIEW

This prevents uncertain predictions from being automatically routed.

🚨 Priority Detection

The system also detects ticket priority using simple keyword rules.

Examples of urgent keywords:

urgent
critical
down
not working
failed
failure
crash
blocked

Tickets containing these keywords are marked as:

URGENT

Otherwise:

NORMAL
🎯 Example

Input:

My payment was charged twice

Output:

Category: BILLING
Confidence: 85%
Priority: NORMAL
Status: AUTO ASSIGNED

Another example:

The production server is down

Output:

Category: TECHNICAL
Confidence: 90%
Priority: URGENT
Status: AUTO ASSIGNED
💡 Future Improvements

With more real-world ticket data, the model could be improved by adding more training examples, handling class imbalance, tuning NLP parameters further, and continuously retraining the model using human-reviewed tickets.

👨‍💻 Project Summary

This project demonstrates an end-to-end NLP classification workflow for automated support ticket routing using supervised machine learning.
