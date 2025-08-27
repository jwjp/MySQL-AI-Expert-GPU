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
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 파인튜닝된 LoRA 어댑터 결합
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("파인튜닝 모델 로드 완료!")

# RAG를 위한 임베딩 모델 및 벡터 DB 로드
# 한국어/다국어 성능 향상을 위해 multilingual 임베딩 사용
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
vectordb = Chroma(persist_directory="db", embedding_function=embedding_model)
# 다양성과 관련성 균형을 위한 MMR 검색기로 교체
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5})
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
    # 과도한 컨텍스트를 방지: 문서별 최대 1200자, 총 3500자 이내로 제한
    clipped = []
    limit_total = 3500
    for doc in retrieved_docs:
        txt = (doc.page_content or "").strip()
        if not txt:
            continue
        txt = txt[:1200]
        remaining = limit_total - sum(len(c) for c in clipped)
        if remaining <= 0:
            break
        clipped.append(txt[:remaining])
    context = "\n\n".join(clipped)

    # 참고 자료가 없으면 안전하게 응답 중단
    if not context.strip():
        msg = "참고 자료에서 관련 정보를 찾지 못했습니다. 질문을 더 구체적으로 하거나 데이터를 추가로 업로드해 주세요."
        print(f"생성된 답변: {msg}")
        return {"answer": msg}

    # 2. 프롬프트 구성
    formatted_prompt = prompt.format(context=context, question=user_query)

    # 3. LLM 답변 생성 (Generation) - 반복 방지 및 프롬프트 에코 제거
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.18,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # 프롬프트 길이만큼 제외하고 신규 토큰만 디코드 -> 에코 제거
    gen_ids = outputs[0][input_len:]
    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # 후처리: 템플릿 토큰이 다시 생성된 경우 앞부분만 사용
    for stopper in ["###", "SYSTEM:", "### 시스템:"]:
        if stopper in answer_text:
            answer_text = answer_text.split(stopper)[0].strip()
            break

    # 간단한 연속 문장 반복 제거
    def collapse_repeats(text: str) -> str:
        parts = [p.strip() for p in text.split("\n")]
        out = []
        last = None
        for p in parts:
            if p and p != last:
                out.append(p)
                last = p
        return "\n".join(out).strip()

    answer = collapse_repeats(answer_text)

    print(f"생성된 답변: {answer}")
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)