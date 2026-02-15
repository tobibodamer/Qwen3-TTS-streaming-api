# Build stage - uses devel image for compilation
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set pip cache
ENV PIP_CACHE_DIR=/cache/pip
WORKDIR /app

# Upgrade build tools
RUN pip install --no-cache-dir -U pip setuptools wheel tomli

# Copy only pyproject.toml to install dependencies (cached layer)
COPY pyproject.toml .

# Extract and install dependencies from pyproject.toml (cached layer)
RUN python -c "import tomli; d = tomli.load(open('pyproject.toml', 'rb')); print('\n'.join(d['project']['dependencies']))" > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Install flash-attention (cached layer - slow build)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Copy the rest of the application (this layer changes frequently)
COPY . .

# Install the project itself (fast, only reruns when code changes)
RUN pip install --no-cache-dir .

# Runtime stage - uses smaller runtime image
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies INCLUDING build tools for Triton
RUN apt-get update && apt-get install -y \
    python3.10 \
    sox \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Huggingface cache
ENV PIP_CACHE_DIR=/cache/pip
WORKDIR /app

# Copy Python packages from builder (includes all dependencies)
COPY --from=builder /opt/conda /opt/conda

# Copy application code from builder (includes installed project)
COPY --from=builder /app /app

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]