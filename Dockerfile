FROM dhi.io/pytorch:2.10-cuda12.8-cudnn9

WORKDIR /

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY src/handler.py .

# Command to run when the container starts
CMD ["python", "-u", "/handler.py"]