# Use Python 3.11 slim image as a base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire application to the container's working directory
COPY ./app /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Expose the port the app will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
