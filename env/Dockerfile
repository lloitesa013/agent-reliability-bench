# Week-2 data-engine environment for agent-reliability-bench.
# Base = the cached NGC PyTorch stack native to this Blackwell (B300) box.
# We ONLY add inference/eval Python deps on top; we do NOT touch the NGC torch build.
FROM nvcr.io/nvidia/pytorch:26.03-py3

# hf_transfer = fast, resumable HF downloads. Pin transformers/sentence-transformers
# but let pip keep the existing NGC torch (it already satisfies their requirements).
RUN pip install --no-cache-dir \
        "transformers>=4.44" \
        "accelerate>=0.34" \
        "sentence-transformers>=3.0" \
        "hf_transfer>=0.1.6" \
    && python -c "import torch; print('torch kept:', torch.__version__)"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /workspace
