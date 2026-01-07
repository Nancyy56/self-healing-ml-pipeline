# 1. Base Python image
FROM python:3.10-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy full project into container
COPY . .

# 7. Set Python path
ENV PYTHONPATH=/app

# 8. Run the Prefect flow
CMD ["python", "-m", "src.orchestration.flows"]
