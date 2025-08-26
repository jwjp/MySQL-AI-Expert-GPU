from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 임베딩 모델 설정 (텍스트를 벡터로 변환)
# 한국어 자연어 이해에 특화된 모델 사용
model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cuda'} # GPU 사용
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 2. 벡터 DB 저장 경로
persist_directory = "db"

def main():
    print("PDF 문서 로드를 시작합니다...")
    # 'data' 폴더 내 모든 PDF 로드
    loader = DirectoryLoader("data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"{len(documents)}개의 문서를 로드했습니다.")

    print("문서를 청크(chunk) 단위로 분할합니다...")
    # 문서를 1000자 단위로 나누고, 200자씩 겹치게 설정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"총 {len(texts)}개의 텍스트 청크로 분할되었습니다.")

    print("텍스트를 임베딩하여 벡터 DB에 저장합니다...")
    # 분할된 텍스트와 임베딩 모델을 사용하여 ChromaDB에 저장
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"벡터 DB가 '{persist_directory}' 경로에 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    main()
