.PHONY: data validate features train evaluate segment score all app clean

# Download and clean raw data
data:
	python -m src.cli data --config configs/params.yaml

# Run data validation checks
validate:
	python -m src.cli validate --config configs/params.yaml

# Build feature matrices
features:
	python -m src.cli features --config configs/params.yaml

# Train all models (probabilistic + ML + churn)
train:
	python -m src.cli train --config configs/params.yaml

# Evaluate models against holdout
evaluate:
	python -m src.cli evaluate --config configs/params.yaml

# Run segmentation
segment:
	python -m src.cli segment --config configs/params.yaml

# Batch score all customers
score:
	python -m src.cli score --config configs/params.yaml

# Full pipeline end-to-end
all: data validate features train evaluate segment score

# Launch Streamlit app
app:
	streamlit run app/streamlit_app.py

# Clean generated artifacts
clean:
	rm -rf data/interim/* data/processed/* data/predictions/* models/* reports/figures/*
