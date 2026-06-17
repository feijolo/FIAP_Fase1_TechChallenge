FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY scripts/ scripts/
COPY notebooks/ notebooks/
COPY reports/ reports/
COPY models/ models/
COPY features/ features/
COPY README.md .

# Default command: run the tabular pipeline
CMD ["python", "scripts/run_tabular_pipeline.py"]
