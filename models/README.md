# Model Downloads

Download the required models into this directory before running the examples.

## Chat Model (TinyLlama 1.1B)

Used by the chat and streaming examples:

```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf \
    --local-dir models/
```

## Function Calling Model (Mistral 7B Instruct)

Used by the function calling example:

```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.3-GGUF \
    mistral-7b-instruct-v0.3.Q4_K_M.gguf \
    --local-dir models/
```

## Embedding Model (e5-small-v2)

Used by the RAG embedding example:

```bash
huggingface-cli download intfloat/e5-small-v2 \
    --local-dir models/e5-small-v2
```

## Requirements

Install the Hugging Face CLI if you don't have it:

```bash
pip install huggingface-hub
```
