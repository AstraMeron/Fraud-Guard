# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes models/, src/, and serve_model.py
COPY . .

# Expose the port Flask is running on
EXPOSE 5000

# Command to run the API
CMD ["python", "serve_model.py"]