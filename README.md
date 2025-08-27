# MySQL AI Expert (GPU)

A Retrieval-Augmented Generation (RAG) chatbot that turns your MySQL PDFs into a knowledge base and answers questions as a MySQL 8.0 expert. Optimized for CUDA GPUs with 4-bit quantized Llama-3 8B (Ko) + LoRA adapter and Chroma vector DB.

## Key improvements in this version
- Much faster PDF ingestion using PyMuPDF (fitz) and incremental Chroma persistence.
- Processes all PDFs under `data/` (not just first one).
- Normalized embeddings and improved chunking for better retrieval quality.
- MMR retriever and controlled generation params to prevent repetitive answers and prompt echo.

## Requirements
- Python 3.13.1
- pip 25.1.1
- CUDA 12.6 GPU recommended
- Install PyTorch trio separately for CUDA 12.6:
  - pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
- Then install the rest:
  - pip install -r requirements.txt

requirements.txt already includes:
- transformers, accelerate, bitsandbytes, peft, trl
- langchain, langchain-community, langchain-chroma, langchain-huggingface, sentence-transformers, chromadb
- fastapi, uvicorn, streamlit, requests, tqdm
- pymupdf (for fast PDF parsing)

## Prepare data
- Put your MySQL PDFs into the `data/` folder.

## Build the vector DB (RAG)
Run:
- python ingest.py

This will:
- Use PyMuPDF to parse all PDFs in `data/`
- Split into ~1000-char chunks with 150 overlap
- Create/update the `db/` Chroma store incrementally (persists after each file)

## Optional: Fine-tune the model
- Prepare `data/finetune_data.jsonl` (SFT format)
- Run: python finetune.py
- The LoRA adapter is saved to `mysql-expert-llama-3-8b/`

## Run the backend API
- python -m uvicorn main:app --host 0.0.0.0 --port 8000

Endpoint:
- POST /ask with JSON {"query": "your question"}

## Simple web UI (Streamlit)
- streamlit run app.py

## Example
```
POST http://127.0.0.1:8000/ask
{"query": "What are the advantages of the InnoDB storage engine?"}
```
Expected behavior:
- The answer should be concise, based on retrieved context, and avoid repetitive phrases.
- If no relevant context is found, server returns a helpful message asking for more specific question or data.

## Notes on performance
- If you have OOM during embeddings, reduce ENCODE_BATCH_SIZE in ingest.py.
- If retrieval results feel redundant, adjust retriever k/fetch_k or chunk size/overlap.
- Keep `normalize_embeddings=True` for sentence-transformers when using cosine similarity.

## Troubleshooting
- If `PyMuPDF` not found: `pip install pymupdf`
- If GPU not detected: run `python checktorch.py` to verify CUDA availability.

## License
See LICENSE.


## Korean/English optimization
- Default embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for strong Korean and multilingual retrieval.
- You can switch to an English-only model (e.g., all-MiniLM-L6-v2) if your corpus is only English. Update `embedding_model_name` in `main.py` and `model_name` in `ingest.py` accordingly.

## Fine-tuning data format (KR/EN)
Place a JSONL file at `data/finetune_data.jsonl` where each line follows the OpenAI-style chat message schema:

Example (English):
{"messages": [
  {"role": "system", "content": "You are a MySQL 8.0 database expert. Please answer questions with clear technical reasoning."},
  {"role": "user", "content": "What are the advantages of the InnoDB storage engine?"},
  {"role": "assistant", "content": "InnoDB supports ACID transactions, row-level locking, and foreign keys, which improve concurrency and integrity."}
]}

Example (Korean):
{"messages": [
  {"role": "system", "content": "당신은 MySQL 8.0 데이터베이스 전문가입니다. 기술적 근거를 들어 명확히 답변하세요."},
  {"role": "user", "content": "InnoDB 스토리지 엔진의 장점은 무엇인가요?"},
  {"role": "assistant", "content": "InnoDB는 트랜잭션을 지원하고, 행 단위 잠금을 사용하며, 외래 키 제약을 지원합니다. 이는 동시성과 데이터 무결성에 유리합니다."}
]}

`finetune.py` uses the model's chat template (`tokenizer.apply_chat_template`) so both KR/EN data are supported out of the box.

## Ingestion robustness
- Adaptive sub-batching automatically reduces batch size if the underlying store rejects large batches.
- Safe persistence: if the wrapper lacks `persist()`, the code tries the underlying client or no-ops, avoiding crashes.
