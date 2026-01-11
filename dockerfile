FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY app.py .
COPY artifacts/ ./artifacts/
COPY utils/ ./utils/

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]