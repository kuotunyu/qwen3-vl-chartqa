FROM python:3.12-slim AS llama-build

ARG LLAMA_CPP_COMMIT=79bba02a6741de194912d370015866414faa83ad

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && git -C /opt/llama.cpp checkout "${LLAMA_CPP_COMMIT}" \
    && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
       -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_SHARED_LIBS=OFF \
       -DGGML_NATIVE=OFF \
       -DLLAMA_CURL=OFF \
       -DLLAMA_BUILD_TESTS=OFF \
       -DLLAMA_BUILD_EXAMPLES=OFF \
       -DLLAMA_BUILD_TOOLS=ON \
       -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /opt/llama.cpp/build --config Release -j2 --target llama-server

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

COPY --from=llama-build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH} \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /home/user/app
COPY --chown=user:user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user:user app.py ./app.py
COPY --chown=user:user runtime.py ./runtime.py

EXPOSE 7860
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3)"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "app.py"]
