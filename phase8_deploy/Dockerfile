FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI backend
CMD ["uvicorn", "phase6_api.main:app", "--host", "0.0.0.0", "--port", "8000"]