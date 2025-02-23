# Start from Runpod's FlashBoot-enabled PyTorch image (CUDA 11.8 + Python 3.10 + Ubuntu 20.04)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8-0-flash-boot-ubuntu20.04

# Let Runpod know we want FlashBoot enabled
ENV FLASHBOOT_ENABLED=1

# Create a directory for your app
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt

# Copy your application code into the container
COPY . /app

# Expose the port Flask will run on
EXPOSE 5000

# Run the Flask app. The heavy model load occurs when app.py starts up.
CMD ["python3", "app.py"]
