FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Needed by some machine-learning packages such as PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv inside the container
RUN pip install --no-cache-dir uv

# Copy dependency files first.
# Docker can reuse this layer if app code changes but dependencies do not.
COPY pyproject.toml uv.lock ./

# Install locked project dependencies.
RUN uv sync --frozen --no-dev --no-install-project

# Copy the Streamlit application files
COPY app ./app

# Download the sentence-transformer model during the Docker build.
# This means the app starts faster later.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]