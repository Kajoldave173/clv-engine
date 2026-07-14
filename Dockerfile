# =============================================================================
# CLV Engine — Pipeline Container
# =============================================================================
# Runs any pipeline CLI command:
#   docker compose run pipeline train
#   docker compose run pipeline tune
#   docker compose run pipeline score
#   docker compose run pipeline validate
#
# Data and model artifacts are mounted as volumes (not baked into the image)
# so they persist between runs and can be shared with the app container.
# =============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed by some ML packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .

# Install the package in editable mode so 'src' is importable
RUN pip install --no-cache-dir -e .

# Create directories for mounted volumes
RUN mkdir -p data/raw data/interim data/processed data/predictions \
             models reports/figures

# Default entrypoint: the Typer CLI
ENTRYPOINT ["python", "-m", "src.cli"]

# Default command (show help if no command specified)
CMD ["--help"]
