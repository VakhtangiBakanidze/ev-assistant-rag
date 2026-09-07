import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Basic project settings
# --------------------------------------------------

load_dotenv()

ELASTICSEARCH_URL = os.getenv(
    "ELASTICSEARCH_URL",
    "http://localhost:9200",
)

INDEX_NAME = os.getenv(
    "ELASTICSEARCH_INDEX",
    "ev-vehicles",
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

FEEDBACK_PATH = Path("data/feedback/feedback.csv")

SOURCE_FIELDS = [
    "id",
    "year",
    "make",
    "model",
    "vehicle_name",
    "vehicle_class",
    "drive",
    "electric_range_miles",
    "city_mpge",
    "highway_mpge",
    "combined_mpge",
    "charge_120v_hours",
    "charge_240v_hours",
    "ev_motor",
    "annual_fuel_cost_usd",
    "document_text",
]


# --------------------------------------------------
# Load Elasticsearch, embedding model, and OpenAI
# --------------------------------------------------

@st.cache_resource
def load_resources():
    """
    Load resources once and reuse them.

    Streamlit reruns the file whenever a user clicks a button.
    The cache prevents Elasticsearch and the embedding model
    from being loaded repeatedly.
    """
    es_client = Elasticsearch(ELASTICSEARCH_URL)

    if not es_client.ping():
        raise ConnectionError(
            "Cannot connect to Elasticsearch. "
            "Run: docker compose up -d"
        )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Add it to your .env file."
        )

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return es_client, embedding_model, openai_client


# --------------------------------------------------
# Retrieval functions
# --------------------------------------------------

def text_search(es_client, question, number_of_results=5):
    """Search Elasticsearch using normal text search."""

    response = es_client.search(
        index=INDEX_NAME,
        size=number_of_results,
        source=SOURCE_FIELDS,
        query={
            "multi_match": {
                "query": question,
                "fields": [
                    "vehicle_name^4",
                    "make^3",
                    "model^3",
                    "vehicle_class^2",
                    "document_text",
                ],
                "fuzziness": "AUTO",
            }
        },
    )

    return response["hits"]["hits"]


def vector_search(
    es_client,
    embedding_model,
    question,
    number_of_results=5,
):
    """Search Elasticsearch using vector similarity."""

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    response = es_client.search(
        index=INDEX_NAME,
        size=number_of_results,
        source=SOURCE_FIELDS,
        knn={
            "field": "embedding",
            "query_vector": question_embedding,
            "k": number_of_results,
            "num_candidates": 100,
        },
    )

    return response["hits"]["hits"]


def hybrid_search(
    es_client,
    embedding_model,
    question,
    number_of_results=5,
):
    """
    Combine text and vector search using Reciprocal Rank Fusion.
    """

    text_results = text_search(
        es_client,
        question,
        number_of_results=20,
    )

    vector_results = vector_search(
        es_client,
        embedding_model,
        question,
        number_of_results=20,
    )

    combined_results = {}
    rrf_constant = 60

    for rank, result in enumerate(text_results, start=1):
        document_id = result["_id"]

        if document_id not in combined_results:
            combined_results[document_id] = {
                "_source": result["_source"],
                "_score": 0,
            }

        combined_results[document_id]["_score"] += 1 / (
            rrf_constant + rank
        )

    for rank, result in enumerate(vector_results, start=1):
        document_id = result["_id"]

        if document_id not in combined_results:
            combined_results[document_id] = {
                "_source": result["_source"],
                "_score": 0,
            }

        combined_results[document_id]["_score"] += 1 / (
            rrf_constant + rank
        )

    ranked_results = sorted(
        combined_results.values(),
        key=lambda result: result["_score"],
        reverse=True,
    )

    return ranked_results[:number_of_results]


# --------------------------------------------------
# RAG functions
# --------------------------------------------------

def build_context(search_results):
    """Convert retrieved BEV records into LLM context."""

    context_parts = []

    for rank, result in enumerate(search_results, start=1):
        document = result["_source"]

        context_parts.append(
            f"""
Source {rank}
Vehicle: {document.get("vehicle_name")}
FuelEconomy.gov vehicle ID: {document.get("id")}

{document.get("document_text")}
""".strip()
        )

    return "\n\n".join(context_parts)


def create_prompt(question, context):
    """Create a grounded prompt for OpenAI."""

    return f"""
You are EV Assistant, an assistant for battery electric vehicles.

Use only the FuelEconomy.gov BEV records provided below.

Rules:
- Only answer questions about battery electric vehicles.
- Do not invent prices, tax incentives, charging network information,
  battery degradation facts, safety ratings, or vehicle features that
  are not included in the retrieved records.
- If the answer is not in the records, say:
  "I do not have enough information in the FuelEconomy.gov records provided."
- Include units for numbers, such as miles, MPGe, or hours.
- Cite the source used, for example [Source 1].
- Keep the answer concise and clear.

Question:
{question}

FuelEconomy.gov BEV records:
{context}
""".strip()


def generate_answer(openai_client, question, context):
    """Ask OpenAI to answer using retrieved context."""

    prompt = create_prompt(question, context)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful EV assistant. "
                    "Use only the supplied vehicle records."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content


def ask_ev_assistant(
    es_client,
    embedding_model,
    openai_client,
    question,
    retrieval_method,
):
    """Run the full RAG flow."""

    if retrieval_method == "Text search":
        search_results = text_search(
            es_client,
            question,
        )

    elif retrieval_method == "Vector search":
        search_results = vector_search(
            es_client,
            embedding_model,
            question,
        )

    else:
        search_results = hybrid_search(
            es_client,
            embedding_model,
            question,
        )

    context = build_context(search_results)

    answer = generate_answer(
        openai_client,
        question,
        context,
    )

    return answer, search_results


# --------------------------------------------------
# Feedback functions
# --------------------------------------------------

def save_feedback(
    question,
    answer,
    retrieval_method,
    search_results,
    rating,
    comment,
):
    """Save user feedback to a local CSV file."""

    FEEDBACK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_ids = [
        str(result["_source"].get("id"))
        for result in search_results
    ]

    source_vehicle_names = [
        str(result["_source"].get("vehicle_name"))
        for result in search_results
    ]

    feedback_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "retrieval_method": retrieval_method,
                "rating": rating,
                "comment": comment,
                "source_ids": " | ".join(source_ids),
                "source_vehicle_names": " | ".join(source_vehicle_names),
            }
        ]
    )

    file_exists = FEEDBACK_PATH.exists()

    feedback_row.to_csv(
        FEEDBACK_PATH,
        mode="a",
        header=not file_exists,
        index=False,
    )


# --------------------------------------------------
# Streamlit page layout
# --------------------------------------------------

st.set_page_config(
    page_title="EV Assistant",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EV Assistant")
st.write(
    "Ask questions about battery electric vehicles using "
    "official FuelEconomy.gov data."
)

st.caption(
    "The assistant covers battery electric vehicles (BEVs) only. "
    "It does not include plug-in hybrids."
)

try:
    es_client, embedding_model, openai_client = load_resources()

except Exception as error:
    st.error("The application could not start.")
    st.exception(error)
    st.stop()


# Create session state values the first time the app starts
if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = None

if "question" not in st.session_state:
    st.session_state.question = None

if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = None

if "feedback_saved" not in st.session_state:
    st.session_state.feedback_saved = False


# User input
question = st.text_input(
    "Ask a question about a battery electric vehicle",
    placeholder=(
        "Example: Which battery electric vehicles have "
        "the longest driving range?"
    ),
)

retrieval_method = st.selectbox(
    "Retrieval method",
    [
        "Hybrid search",
        "Text search",
        "Vector search",
    ],
    index=0,
)

ask_button = st.button("Ask EV Assistant", type="primary")


# Ask question
if ask_button:

    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        with st.spinner("Searching BEV records and generating an answer..."):

            answer, sources = ask_ev_assistant(
                es_client=es_client,
                embedding_model=embedding_model,
                openai_client=openai_client,
                question=question,
                retrieval_method=retrieval_method,
            )

        st.session_state.answer = answer
        st.session_state.sources = sources
        st.session_state.question = question
        st.session_state.retrieval_method = retrieval_method
        st.session_state.feedback_saved = False


# Show answer and sources
if st.session_state.answer:

    st.subheader("Answer")
    st.write(st.session_state.answer)

    st.subheader("Retrieved Sources")

    for rank, result in enumerate(
        st.session_state.sources,
        start=1,
    ):
        source = result["_source"]

        source_title = (
            f"Source {rank}: "
            f"{source.get('vehicle_name')}"
        )

        with st.expander(source_title):
            st.write(
                f"**FuelEconomy.gov vehicle ID:** "
                f"{source.get('id')}"
            )

            st.write(
                f"**Electric range:** "
                f"{source.get('electric_range_miles')} miles"
            )

            st.write(
                f"**Combined efficiency:** "
                f"{source.get('combined_mpge')} MPGe"
            )

            st.write(
                f"**240V charging time:** "
                f"{source.get('charge_240v_hours')} hours"
            )

            st.write(
                f"**Vehicle class:** "
                f"{source.get('vehicle_class')}"
            )

            st.write("**Full source record:**")
            st.code(source.get("document_text"))


    st.subheader("Was this answer helpful?")

    feedback_comment = st.text_area(
        "Optional feedback comment",
        placeholder=(
            "Example: The answer was correct, "
            "but I wanted more vehicle comparisons."
        ),
    )

    feedback_col_1, feedback_col_2 = st.columns(2)

    with feedback_col_1:
        helpful_button = st.button(
            "👍 Helpful",
            disabled=st.session_state.feedback_saved,
        )

    with feedback_col_2:
        not_helpful_button = st.button(
            "👎 Not helpful",
            disabled=st.session_state.feedback_saved,
        )

    if helpful_button:
        save_feedback(
            question=st.session_state.question,
            answer=st.session_state.answer,
            retrieval_method=st.session_state.retrieval_method,
            search_results=st.session_state.sources,
            rating="helpful",
            comment=feedback_comment,
        )

        st.session_state.feedback_saved = True
        st.success("Thank you. Your feedback was saved.")

    if not_helpful_button:
        save_feedback(
            question=st.session_state.question,
            answer=st.session_state.answer,
            retrieval_method=st.session_state.retrieval_method,
            search_results=st.session_state.sources,
            rating="not_helpful",
            comment=feedback_comment,
        )

        st.session_state.feedback_saved = True
        st.success("Thank you. Your feedback was saved.")


        