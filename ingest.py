import time
from datetime import datetime
from tqdm import tqdm
import os
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# --- 설정값 ---
# 임베딩 배치 크기(encode용). 8GB VRAM: 512~1024 권장, OOM 시 한 단계 낮추기
ENCODE_BATCH_SIZE = 768
# 벡터DB 쓰기 배치 크기(너무 크게 잡으면 메모리 피크 증가)
WRITE_BATCH_SIZE = 8000

# 청크 조정
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 80

# 성능 힌트: TF32/고정밀 매트멀 허용
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# 1. 임베딩 모델 설정
# model_name = "jhgan/ko-sroberta-multitask"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': False}
)

# 2. 벡터 DB 저장 경로
persist_directory = "db"

# --- 처리할 PDF 파일 경로 지정 ---
data_folder = "data"
pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
if not pdf_files:
    print(f"오류: '{data_folder}' 폴더에 PDF 파일이 없습니다.")
    raise SystemExit(1)
pdf_path = os.path.join(data_folder, pdf_files[0])


def format_eta(processed, total, start_ts):
    if processed == 0:
        return "계산 중..."
    elapsed = time.time() - start_ts
    rate = processed / max(elapsed, 1e-6)  # items/sec
    remaining = total - processed
    eta_sec = remaining / max(rate, 1e-6)
    return f"{eta_sec/60:.1f}분 남음 (속도: {rate:.1f}개/초)"

def main():
    start_time = time.time()
    print(f"✅ 작업 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  -> 처리 대상 파일: {pdf_path}")
    print(f"  -> 임베딩 장치(device): {device}")
    print(f"  -> 인코딩 배치: {ENCODE_BATCH_SIZE}, DB 쓰기 배치: {WRITE_BATCH_SIZE}")
    print(f"  -> 청크 설정: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,  # SentenceTransformer 모델을 embedding_function으로 전달
    )

    # 1) PDF 로드
    print("\n[1/5] PDF 페이지 로드 중...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  -> 총 {len(pages)} 페이지 로드 완료.")

    # 2) 텍스트 분할
    print("\n[2/5] 텍스트 청크로 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(pages)
    # 공백/매우 짧은 텍스트 제거
    chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) >= 20]
    total = len(chunks)
    print(f"  -> 총 {total}개의 유효 청크 생성.")

    # 3) 벡터 DB에 직접 문서 저장
    print("\n[3/5] 문서 임베딩 및 DB 저장 시작...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    # 4) 디스크 반영
    print("\n[4/5] 벡터 DB 자동 저장됨...")
    print("  -> 저장 완료.")

    # 5) 요약
    elapsed_time = time.time() - start_time
    print("\n[5/5] 요약")
    print(f"  -> 처리된 청크: {total}")
    print(f"  -> 평균 속도: {total / max(elapsed_time, 1e-6):.1f}개/초")
    print(f"  -> 예상 대비 잔여 시간: 0.0분 (완료)")
    print(f"\n✅ 작업 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")

if __name__ == "__main__":
    main()