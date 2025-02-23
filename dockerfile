# Start from Runpod's FlashBoot-enabled PyTorch image (CUDA 11.8 + Python 3.10 + Ubuntu 20.04)
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Enable FlashBoot (if desired)
ENV FLASHBOOT_ENABLED=1

# Create app directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/requirements.txt

# Install Git LFS (needed to clone large files on Hugging Face)
RUN apt-get update && apt-get install -y git-lfs && \
    git lfs install

# Upgrade pip and install Python deps
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --ignore-installed --no-cache-dir -r /app/requirements.txt

# ---- Download the Flex model from Hugging Face ----
# This will add ~22GB to your image layer.
RUN git clone https://huggingface.co/ostris/Flex.1-alpha /app/models/Flex.1-alpha && \
    cd /app/models/Flex.1-alpha && git lfs pull

# Copy the rest of your application code
COPY . /app

# Expose the port if you're running a Flask app
EXPOSE 5000

# Start your server (e.g. a Flask server in app.py)
CMD ["python3", "app.py"]
