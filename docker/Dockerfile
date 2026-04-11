# Benchmark Runner Docker Image
# Includes all memory store libraries and dependencies

FROM python:3.12-slim-bookworm

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    # For Neo4j Python driver
    libssl-dev \
    # For numpy/scipy if needed
    libopenblas-dev \
    # For ChromaDB
    libsqlite3-dev \
    # General utilities
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
COPY docker/requirements-docker.txt .

# Install Python dependencies
# Split into layers for better caching
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the application code
COPY memory_stores/ ./memory_stores/
COPY context_managers/ ./context_managers/
COPY benchmark/ ./benchmark/
COPY config/ ./config/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY *.py ./
COPY *.md ./
COPY *.sh ./

# Create directories for data
RUN mkdir -p /app/data /app/results /app/configs

# Verify installations
RUN python -c "
import sys
print(f'Python: {sys.version}')

# Check all memory stores
stores = []
try:
    from mem0 import Memory
    stores.append('✓ Mem0')
except ImportError as e:
    stores.append(f'✗ Mem0: {e}')

try:
    from zep_python import Zep
    stores.append('✓ Zep')
except ImportError as e:
    stores.append(f'✗ Zep: {e}')

try:
    from graphiti_core import Graphiti
    stores.append('✓ Graphiti')
except ImportError as e:
    stores.append(f'✗ Graphiti: {e}')

try:
    import letta
    stores.append('✓ Letta')
except ImportError as e:
    stores.append(f'✗ Letta: {e}')

try:
    import mempalace
    stores.append('✓ MemPalace')
except ImportError as e:
    stores.append(f'✗ MemPalace: {e}')

print('\\nMemory Stores:')
for s in stores:
    print(f'  {s}')
"

# Default command
CMD ["python", "-m", "scripts.coordinator", "--help"]
