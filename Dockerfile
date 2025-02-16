# Use a base image with Python
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
        git \
        gcc \
        make \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1-mesa-glx && \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Working directory 
WORKDIR /src/app

#  requirements file
COPY ./requirements.txt /src/app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

# Clone the decord repository  or install the package
RUN git clone --recursive https://github.com/dmlc/decord

# Ensure ffmpeg and ffprobe are executable
RUN chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# Copy  project folder to the container
COPY . /src/app

# Expose the application port
EXPOSE 5050

# Command to run the application
CMD ["python", "test_api.py"]
