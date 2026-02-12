FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    sox \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set Huggingface cache
#ENV HF_HOME=/cache
ENV PIP_CACHE_DIR=/cache/pip
WORKDIR /app

# --- Dependency Stage ---
FROM base AS dependencies

# Upgrade build tools
RUN pip install --no-cache-dir -U pip setuptools wheel tomli

# Install torch with CUDA 12.1 support (Large layer, cached separately)
#RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy only pyproject.toml to install dependencies
COPY pyproject.toml .

# Extract and install dependencies from pyproject.toml
RUN python3 -c "import tomli; d = tomli.load(open('pyproject.toml', 'rb')); print('\n'.join(d['project']['dependencies']))" > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Install flash-attention (Slow to build/install, cached here)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# --- Final Stage ---
FROM dependencies AS runtime

# Copy the rest of the application
COPY . .

# Install the project itself
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
