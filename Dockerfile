FROM python:3.11.1-slim

WORKDIR /

# Copy and install requirements
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY src/handler.py .

# Command to run when the container starts
CMD ["python", "-u", "/handler.py"]