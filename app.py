import streamlit as st

from ticket_classifier import (
    train_model,
    classify_ticket
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return train_model()


result = load_model()

model = result["model"]


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎫 AI Customer Support Ticket Classifier"
)

st.write(
    "Automatically classify customer support tickets "
    "into Billing, Technical, HR, or General."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Test Accuracy",
        f"{result['accuracy'] * 100:.2f}%"
    )


with col2:

    st.metric(
        "CV Accuracy",
        f"{result['cv_score'] * 100:.2f}%"
    )


with col3:

    st.metric(
        "Training Records",
        len(result["labels"]) * 8
    )


# ============================================================
# SAMPLE TICKETS
# ============================================================

st.divider()

st.subheader(
    "📩 New Customer Ticket"
)


sample_tickets = {

    "Select a sample": "",

    "Billing - Duplicate Payment":
        "My payment was charged twice",

    "Technical - API Down":
        "The production API is down",

    "HR - Leave Request":
        "I want to apply for annual leave",

    "General - Profile":
        "How can I update my profile",

    "Billing - Invoice":
        "I have not received my invoice",

    "Technical - Application":
        "My application is not working",

    "HR - Salary":
        "I have a question about my salary",

    "General - Account":
        "I have a question about my account"
}


selected_sample = st.selectbox(

    "Choose a sample ticket",

    list(sample_tickets.keys())
)


ticket_text = st.text_area(

    "Enter ticket text",

    value=sample_tickets[selected_sample],

    height=150,

    placeholder="Enter customer support ticket..."
)


# ============================================================
# CLASSIFY BUTTON
# ============================================================

if st.button(

    "🔍 Classify Ticket",

    type="primary",

    use_container_width=True
):

    output = classify_ticket(
        model,
        ticket_text,
        threshold=0.60
    )


    # --------------------------------------------------------
    # VALIDATION ERROR
    # --------------------------------------------------------

    if output["status"] == "INVALID INPUT":

        st.error(
            output["message"]
        )


    else:

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Category",
                output["category"]
            )


        with col2:

            st.metric(
                "Confidence",
                f"{output['confidence']}%"
            )


        with col3:

            st.metric(
                "Priority",
                output["priority"]
            )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if output["status"] == "AUTO ASSIGNED":

            st.success(
                "✅ AUTO ASSIGNED"
            )

        else:

            st.warning(
                "⚠️ NEEDS HUMAN REVIEW"
            )


        # ----------------------------------------------------
        # CONFIDENCE BAR
        # ----------------------------------------------------

        st.subheader(
            "Confidence"
        )


        st.progress(
            output["confidence"] / 100
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Category Probabilities"
        )


        probability_data = [

            {
                "Category": category,
                "Probability": f"{probability}%"
            }

            for category, probability
            in output["probabilities"].items()
        ]


        st.dataframe(
            probability_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL DETAILS
# ============================================================

st.divider()

with st.expander(
    "⚙️ Model Details"
):

    st.write(
        "**Feature Extraction:** TF-IDF"
    )

    st.write(
        "**Classifier:** Logistic Regression"
    )

    st.write(
        "**Hyperparameter Tuning:** GridSearchCV"
    )

    st.write(
        "**Confidence Threshold:** 60%"
    )

    st.write(
        "**Best Parameters:**"
    )

    st.json(
        result["best_params"]
    )


# ============================================================
# EVALUATION
# ============================================================

with st.expander(
    "📊 Model Evaluation"
):

    metrics_df = (
        __import__("pandas")
        .DataFrame(result["report"])
        .transpose()
        .round(3)
    )


    st.dataframe(
        metrics_df,
        use_container_width=True
    )


    st.subheader(
        "Confusion Matrix"
    )


    cm_df = (
        __import__("pandas")
        .DataFrame(
            result["confusion_matrix"],
            index=result["labels"],
            columns=result["labels"]
        )
    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Customer Support Ticket Classifier "
    "| TF-IDF + Logistic Regression"
)