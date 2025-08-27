import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# --- 1. 모델 및 RAG 초기 설정 ---
# 파인튜닝된 모델 어댑터 경로
adapter_path = "./mysql-expert-llama-3-8b"
base_model_id = "beomi/Llama-3-Open-Ko-8B"

# 4비트 양자화 설정 (추론 시에도 메모리 절약)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 기반 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

# 파인튜닝된 LoRA 어댑터 결합
model = PeftModel.from_pretrained(base_model, adapter_path)
print("파인튜닝 모델 로드 완료!")

# RAG를 위한 임베딩 모델 및 벡터 DB 로드
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vectordb = Chroma(persist_directory="db", embedding_function=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # 가장 유사한 3개 청크 검색
print("벡터 DB 로드 완료!")

# LLM에 전달할 프롬프트 템플릿 정의
prompt_template = """
### 시스템:
당신은 MySQL 8.0 데이터베이스 전문가입니다. 아래에 제공된 '참고 자료'를 바탕으로 사용자의 '질문'에 대해 명확하고 기술적인 근거를 들어 답변해주세요. 참고 자료에 없는 내용은 답변하지 마세요.

### 참고 자료:
{context}

### 질문:
{question}

### 답변:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- 2. FastAPI 앱 설정 ---
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    user_query = request.query
    print(f"수신된 질문: {user_query}")

    # 1. RAG: 벡터 DB에서 관련 문서 검색 (Retrieval)
    retrieved_docs = retriever.invoke(user_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 2. 프롬프트 구성
    formatted_prompt = prompt.format(context=context, question=user_query)

    # 3. LLM 답변 생성 (Generation)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 부분을 제외하고 순수 답변만 추출
    answer = response_text.split("### 답변:")[1].strip()
    
    print(f"생성된 답변: {answer}")
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)