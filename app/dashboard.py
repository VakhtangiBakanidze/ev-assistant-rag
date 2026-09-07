from pathlib import Path

import pandas as pd
import streamlit as st


FEEDBACK_PATH = Path("data/feedback/feedback.csv")


st.set_page_config(
    page_title="EV Assistant Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 EV Assistant Monitoring Dashboard")

st.write(
    "This dashboard shows user questions, retrieval choices, "
    "and feedback collected from the EV Assistant."
)


if not FEEDBACK_PATH.exists():
    st.info(
        "No feedback has been collected yet. "
        "Use the EV Assistant and submit feedback first."
    )
    st.stop()


feedback_df = pd.read_csv(FEEDBACK_PATH)

feedback_df["timestamp"] = pd.to_datetime(
    feedback_df["timestamp"],
    errors="coerce",
)

feedback_df["date"] = feedback_df["timestamp"].dt.date

feedback_df["answer_length"] = (
    feedback_df["answer"]
    .fillna("")
    .str.len()
)


# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

total_questions = len(feedback_df)

helpful_count = (
    feedback_df["rating"] == "helpful"
).sum()

not_helpful_count = (
    feedback_df["rating"] == "not_helpful"
).sum()

if total_questions > 0:
    helpful_percentage = helpful_count / total_questions * 100
else:
    helpful_percentage = 0


metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

metric_col_1.metric(
    "Total feedback entries",
    total_questions,
)

metric_col_2.metric(
    "Helpful answers",
    helpful_count,
)

metric_col_3.metric(
    "Helpful percentage",
    f"{helpful_percentage:.1f}%",
)


# --------------------------------------------------
# Chart 1: Feedback ratings
# --------------------------------------------------

st.subheader("1. Feedback ratings")

rating_counts = feedback_df["rating"].value_counts()

st.bar_chart(rating_counts)


# --------------------------------------------------
# Chart 2: Questions over time
# --------------------------------------------------

st.subheader("2. Questions over time")

questions_by_date = (
    feedback_df
    .groupby("date")
    .size()
    .reset_index(name="questions")
    .set_index("date")
)

st.line_chart(questions_by_date)


# --------------------------------------------------
# Chart 3: Retrieval method usage
# --------------------------------------------------

st.subheader("3. Retrieval methods used")

retrieval_method_counts = (
    feedback_df["retrieval_method"]
    .value_counts()
)

st.bar_chart(retrieval_method_counts)


# --------------------------------------------------
# Chart 4: Most frequently retrieved vehicles
# --------------------------------------------------

st.subheader("4. Most frequently retrieved vehicles")

vehicle_names = (
    feedback_df["source_vehicle_names"]
    .fillna("")
    .str.split(" \\| ")
    .explode()
)

vehicle_names = vehicle_names[
    vehicle_names != ""
]

top_vehicles = vehicle_names.value_counts().head(10)

st.bar_chart(top_vehicles)


# --------------------------------------------------
# Chart 5: Answer-length distribution
# --------------------------------------------------

st.subheader("5. Answer length distribution")

# Create simple answer-length groups
answer_length_bins = [0, 100, 200, 300, 400, 500, 1000, float("inf")]

answer_length_labels = [
    "0-100 characters",
    "101-200 characters",
    "201-300 characters",
    "301-400 characters",
    "401-500 characters",
    "501-1000 characters",
    "More than 1000 characters",
]

feedback_df["answer_length_group"] = pd.cut(
    feedback_df["answer_length"],
    bins=answer_length_bins,
    labels=answer_length_labels,
    include_lowest=True,
)

answer_length_counts = (
    feedback_df["answer_length_group"]
    .value_counts()
    .reindex(answer_length_labels, fill_value=0)
)

# Convert the labels to a normal table
answer_length_chart = answer_length_counts.reset_index()

answer_length_chart.columns = [
    "answer_length_group",
    "number_of_answers",
]

st.bar_chart(
    answer_length_chart,
    x="answer_length_group",
    y="number_of_answers",
)