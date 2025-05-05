# Dockerfile
FROM python:3.11-slim

# Add this near the top of your Dockerfile
ENV PYTHONUNBUFFERED=1
#ENV STREAMLIT_GLOBAL_DEVELOPMENT_MODE=true

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/vector_store /app/data/pdfs /app/data/texts /app/logs && \
	chmod -R 777 /app/data /app/logs

# Copy application
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
