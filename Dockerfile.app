# =============================================================================
# CLV Engine — Streamlit App Container
# =============================================================================
# Self-contained image for the dashboard. Bundles model artifacts and
# scored data so it can run anywhere without volume mounts.
#
# Local:           docker compose up app → http://localhost:8501
# HuggingFace:     push this Dockerfile + code to a Spaces repo
#
# Phase 6 will add: app/streamlit_app.py, app/pages/, app/components/
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (needed for imports in app)
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy the Streamlit app
COPY app/ app/

# Bundle model artifacts and scored data into the image
# These are the trained models and pre-scored customer table
# that the dashboard reads at startup
COPY models/ models/
COPY data/processed/ data/processed/
COPY data/predictions/ data/predictions/

# Streamlit configuration
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = "#2563eb"\n\
backgroundColor = "#ffffff"\n\
secondaryBackgroundColor = "#f8fafc"\n\
textColor = "#1e293b"\n\
' > /root/.streamlit/config.toml

EXPOSE 8501

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py"]
