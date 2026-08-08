import re
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# DATASET
# ============================================================

data = {
    "subject": [
        "API Error",
        "Application Crash",
        "Server Problem",
        "Website Down",

        "Invoice Issue",
        "Duplicate Payment",
        "Wrong Invoice",
        "Refund Issue",

        "Leave Request",
        "Salary Question",
        "Employee Details",
        "HR Request",

        "Account Help",
        "Profile Update",
        "General Question",
        "Account Settings",

        "API Timeout",
        "Login Error",
        "Server Connection",
        "Application Error",

        "Payment Failed",
        "Billing Problem",
        "Missing Receipt",
        "Refund Pending",

        "Vacation Request",
        "Payroll Question",
        "Leave Balance",
        "HR Support",

        "Username Change",
        "Profile Help",
        "Account Update",
        "General Help"
    ],

    "body": [
        "API returns 500 error",
        "Application crashes during login",
        "Server connection failed",
        "Website is not loading",

        "Invoice not received",
        "Payment charged twice",
        "Wrong invoice amount",
        "Refund has not been processed",

        "How do I apply for leave",
        "I have a salary related question",
        "How can I update employee details",
        "I need help with my HR request",

        "I need help with my account",
        "How can I update my profile",
        "I have a general question",
        "Where can I find my account settings",

        "API timeout error",
        "Login page is not working",
        "Unable to connect to server",
        "Application is showing an error",

        "My payment failed",
        "I have a billing issue",
        "Payment receipt is missing",
        "My refund is still pending",

        "I want to apply for vacation",
        "I have a payroll question",
        "I want to check my leave balance",
        "I need HR support",

        "I want to change my username",
        "I need help with my profile",
        "How do I update my account",
        "I have a general account question"
    ],

    "category": [
        "TECHNICAL", "TECHNICAL", "TECHNICAL", "TECHNICAL",
        "BILLING", "BILLING", "BILLING", "BILLING",
        "HR", "HR", "HR", "HR",
        "GENERAL", "GENERAL", "GENERAL", "GENERAL",

        "TECHNICAL", "TECHNICAL", "TECHNICAL", "TECHNICAL",
        "BILLING", "BILLING", "BILLING", "BILLING",
        "HR", "HR", "HR", "HR",
        "GENERAL", "GENERAL", "GENERAL", "GENERAL"
    ]
}


df = pd.DataFrame(data)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["text"] = (
    df["subject"] + " " + df["body"]
)

df["text"] = df["text"].apply(
    clean_text
)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    X = df["text"]
    y = df["category"]


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.25,

        random_state=42,

        stratify=y
    )


    pipeline = Pipeline([

        (
            "tfidf",

            TfidfVectorizer(
                stop_words="english",
                lowercase=True
            )
        ),

        (
            "classifier",

            LogisticRegression(
                max_iter=2000
            )
        )
    ])


    param_grid = {

        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],

        "classifier__C": [
            0.1,
            1,
            10,
            100
        ]
    }


    grid_search = GridSearchCV(

        pipeline,

        param_grid,

        cv=3,

        scoring="accuracy",

        n_jobs=-1
    )


    grid_search.fit(
        X_train,
        y_train
    )


    model = grid_search.best_estimator_


    # Evaluation

    y_pred = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )


    labels = sorted(
        y.unique()
    )


    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )


    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_
    }


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_ticket(ticket_text):

    if not isinstance(
        ticket_text,
        str
    ):

        return False, "Invalid input."

    ticket_text = ticket_text.strip()


    if not ticket_text:

        return False, "Ticket cannot be empty."


    if len(ticket_text) < 5:

        return False, "Ticket is too short."


    if len(ticket_text) > 1000:

        return False, "Ticket is too long."


    return True, ""


# ============================================================
# PRIORITY DETECTION
# ============================================================

def detect_priority(ticket_text):

    urgent_keywords = [

        "urgent",
        "emergency",
        "critical",
        "down",
        "not working",
        "failed",
        "failure",
        "crash",
        "crashed",
        "blocked"
    ]


    text = ticket_text.lower()


    for keyword in urgent_keywords:

        if keyword in text:

            return "URGENT"


    return "NORMAL"


# ============================================================
# CLASSIFY TICKET
# ============================================================

def classify_ticket(
    model,
    ticket_text,
    threshold=0.60
):

    valid, message = validate_ticket(
        ticket_text
    )


    if not valid:

        return {
            "category": None,
            "confidence": 0,
            "priority": None,
            "status": "INVALID INPUT",
            "message": message
        }


    cleaned_text = clean_text(
        ticket_text
    )


    prediction = model.predict(
        [cleaned_text]
    )[0]


    probabilities = model.predict_proba(
        [cleaned_text]
    )[0]


    confidence = probabilities.max()


    if confidence >= threshold:

        status = "AUTO ASSIGNED"

    else:

        status = "NEEDS HUMAN REVIEW"


    return {

        "category": prediction,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "priority":
            detect_priority(
                ticket_text
            ),

        "status": status,

        "probabilities": {
            category: round(
                probability * 100,
                2
            )

            for category, probability
            in zip(
                model.classes_,
                probabilities
            )
        }
    }


# ============================================================
# TEST WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    result = train_model()

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nCategory Distribution:")
    print(df["category"].value_counts())


    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    print(
        "Best Parameters:",
        result["best_params"]
    )

    print(
        "Cross Validation Score:",
        round(
            result["cv_score"],
            4
        )
    )


    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        "Accuracy:",
        round(
            result["accuracy"],
            4
        )
    )

    print("\nClassification Report:")

    print(
        pd.DataFrame(
            result["report"]
        ).transpose()
    )


    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            result["confusion_matrix"],
            index=result["labels"],
            columns=result["labels"]
        )
    )


    print("\n" + "=" * 60)
    print("NEW TICKET TESTING")
    print("=" * 60)

    test_tickets = [

        "My payment was charged twice",

        "The API is down",

        "I want to apply for leave",

        "How can I update my profile",

        "I have not received my invoice",

        "My application is not working"
    ]


    for ticket in test_tickets:

        output = classify_ticket(
            result["model"],
            ticket
        )


        print("\nTicket:", ticket)

        print(
            "Category:",
            output["category"]
        )

        print(
            "Confidence:",
            output["confidence"],
            "%"
        )

        print(
            "Priority:",
            output["priority"]
        )

        print(
            "Status:",
            output["status"]
        )