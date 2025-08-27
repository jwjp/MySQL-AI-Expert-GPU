import time
from datetime import datetime
from tqdm import tqdm
import os
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 설정값 ---
# 임베딩 배치 크기(encode용). 8GB VRAM: 512~1024 권장, OOM 시 한 단계 낮추기
ENCODE_BATCH_SIZE = 768
# 벡터DB 쓰기 배치 크기(너무 크게 잡으면 메모리 피크 증가)
WRITE_BATCH_SIZE = 5000

# 청크 조정 (MiniLM 권장: 800~1200자)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# 성능 힌트: TF32/고정밀 매트멀 허용
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# 1. 임베딩 모델 설정
# 한국어/다국어 최적화 임베딩 모델
# 참고: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': ENCODE_BATCH_SIZE}
)

# 2. 벡터 DB 저장 경로
persist_directory = "db"

# --- 처리할 PDF 파일 경로 지정 ---
data_folder = "data"
pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
if not pdf_files:
    print(f"오류: '{data_folder}' 폴더에 PDF 파일이 없습니다.")
    raise SystemExit(1)


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
    print(f"  -> 처리 대상 파일 수: {len(pdf_files)}")
    print(f"  -> 임베딩 장치(device): {device}")
    print(f"  -> 인코딩 배치: {ENCODE_BATCH_SIZE}, DB 쓰기 배치: {WRITE_BATCH_SIZE}")
    print(f"  -> 청크 설정: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,  # SentenceTransformer 모델을 embedding_function으로 전달
    )

    total_chunks = 0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for fname in pdf_files:
        pdf_path = os.path.join(data_folder, fname)
        try:
            # 1) PDF 로드 (PyMuPDF 기반으로 빠름)
            print(f"\n[1/3] '{fname}' 페이지 로드 중...")
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            print(f"  -> 총 {len(pages)} 페이지 로드 완료.")

            # 2) 텍스트 분할
            print("[2/3] 텍스트 청크로 분할 중...")
            chunks = text_splitter.split_documents(pages)
            chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) >= 20]
            print(f"  -> {len(chunks)}개의 유효 청크.")

            # 3) 벡터 DB 저장 (증분 추가 + 주기적 persist)
            print("[3/3] 문서 임베딩 및 DB 저장 중...")
            for i in range(0, len(chunks), WRITE_BATCH_SIZE):
                batch = chunks[i:i+WRITE_BATCH_SIZE]
                if not batch:
                    continue
                # 적응형 서브 배칭: 거부 시 절반으로 줄이며 재시도
                start = 0
                step = len(batch)
                while start < len(batch):
                    sub = batch[start:start+step]
                    try:
                        vectordb.add_documents(sub)
                        start += step
                    except ValueError as ve:
                        # 예: "Batch size ... is greater than max batch size ..."
                        if step <= 1:
                            print(f"    - 경고: 단일 문서 추가 실패: {ve}")
                            # 실패 문서 스킵
                            start += 1
                            step = max(1, step)
                            continue
                        step = max(1, step // 2)
                        print(f"    - 배치 축소 재시도: {step}개")
                    except Exception as e:
                        print(f"    - 배치 추가 중 기타 오류: {e}")
                        # 해당 서브배치 스킵하고 다음으로 이동
                        start += step
            # 버전별 안전 persist 처리
            persisted = False
            if hasattr(vectordb, "persist"):
                try:
                    vectordb.persist()
                    persisted = True
                except Exception as e:
                    print(f"    - persist() 호출 실패, 대체 경로 시도: {e}")
            if not persisted:
                client = getattr(vectordb, "_client", None)
                if client and hasattr(client, "persist"):
                    try:
                        client.persist()
                        persisted = True
                    except Exception as e:
                        print(f"    - _client.persist() 호출 실패: {e}")
            if not persisted:
                print("    - 주의: 사용 중인 Chroma 래퍼에 persist 메서드가 없습니다. (무시) ")
            total_chunks += len(chunks)
            print(f"  -> 누적 청크 수: {total_chunks}")
        except Exception as e:
            print(f"  !! '{fname}' 처리 중 오류 발생: {e}")
            continue

    # 요약
    elapsed_time = time.time() - start_time
    print("\n요약")
    print(f"  -> 총 처리된 청크: {total_chunks}")
    print(f"  -> 평균 속도: {total_chunks / max(elapsed_time, 1e-6):.1f}개/초")
    print(f"\n✅ 작업 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")

if __name__ == "__main__":
    main()