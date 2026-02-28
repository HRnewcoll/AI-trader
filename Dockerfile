FROM python:3.12-slim

WORKDIR /app

# System dependencies for TA-Lib, Rust toolchain
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install Rust (for rust_core)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build Rust extensions
RUN cd rust_core && cargo build --release 2>/dev/null || echo "Rust build skipped"

# Create directories
RUN mkdir -p logs artifacts/models artifacts/onnx artifacts/chromadb

EXPOSE 8501 9090

CMD ["python", "main_orchestrator.py"]
