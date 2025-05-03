FROM python:3.10-slim-buster

# Install system dependencies, Praat, CMake, and additional build tools => necessary because Praat refuses to get installed :D
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3-dev \
        libpython3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
        libsndfile1 \
        praat \
        wget \
        git \
        libfftw3-dev \
        libatlas-base-dev \
        curl \
        ca-certificates \
        g++ && \
    # Clean up to reduce image size
    rm -rf /var/lib/apt/lists/*

RUN cmake --version

# Set working directory
WORKDIR /app
COPY . /app

RUN python3 -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install praat-parselmouth

CMD ["python3", "timing.py"]
