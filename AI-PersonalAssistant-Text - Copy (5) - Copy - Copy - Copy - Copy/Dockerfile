# Use official Python base image
ARG PYTHON_VERSION=3.13.9
FROM python:${PYTHON_VERSION}-slim-bookworm

# Set working directory
WORKDIR /app

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy all project files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "src.main:app"]