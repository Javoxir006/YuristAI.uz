from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

app = FastAPI(
    title="YuristAI.uz RAG API (Demo)",
    version="0.4.2",
    description="YuristAI.uz qonunchilik bazasi uchun RAG API test spetsifikatsiyasi demo versiyasi"
)

# Dummy saqlash ombori (Idempotentlik va async GET uchun)
BATCH_STORAGE: Dict[str, Dict[str, Any]] = {}
WEB_DIR = Path(__file__).parent / "web"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def get_llm_model() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return "demo-llm-v1"


def generate_openai_answer(question: str, language: Optional[str]) -> tuple[str, int, int]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY sozlanmagan")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    language_name = {
        "uz-Latn": "o'zbek tilida (lotin yozuvida)",
        "uz-Cyrl": "o'zbek tilida (kirill yozuvida)",
        "ru": "rus tilida",
        "en": "ingliz tilida",
    }.get(language or "", "savol berilgan tilda")
    request_body = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Siz O'zbekiston qonunchiligi bo'yicha savollarga javob beruvchi "
                    "yordamchisiz. Javobni qisqa, aniq va "
                    f"{language_name} yozing. Agar ishonchli ma'lumotga ega bo'lmasangiz, "
                    "buni ochiq ayting. Qonun moddasi, hujjat yoki manba raqamini "
                    "tekshirilmagan bo'lsa to'qib chiqarmang. Siz rasmiy yuridik "
                    "maslahatchi emassiz; zarur bo'lsa lex.uz dagi amaldagi rasmiy "
                    "hujjatni tekshirishni tavsiya qiling."
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
    }).encode("utf-8")
    request = Request(
        OPENAI_API_URL,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=45) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"OpenAI API so'rovi xato bilan yakunlandi (HTTP {exc.code})") from exc
    except URLError as exc:
        raise RuntimeError("OpenAI API bilan bog'lanib bo'lmadi") from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("OpenAI API javobini o'qib bo'lmadi") from exc

    try:
        answer = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI API javobida javob matni topilmadi") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("OpenAI API bo'sh javob qaytardi")
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return (
        answer.strip(),
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


@app.get("/", include_in_schema=False)
def get_homepage():
    return FileResponse(WEB_DIR / "index.html")

# --- REQUEST SCHEMAS ---
class Options(BaseModel):
    top_k: int = 5
    return_contexts: bool = True
    return_debug: bool = True
    temperature: float = 0.0
    seed: Optional[int] = 42
    as_of_date: Optional[str] = "2026-11-01"
    answer_language: Optional[str] = "auto"

class QuestionItem(BaseModel):
    id: str
    question: str = Field(..., min_length=1, max_length=2000)
    language: Optional[str] = None

class BatchRequest(BaseModel):
    run_id: str
    batch_id: str
    batch_index: int
    total_batches: int
    options: Optional[Options] = Options()
    items: List[QuestionItem]

# --- ENDPOINTS ---

@app.get("/v1/health", status_code=200)
def get_health():
    """Servis ishlayotganini tekshirish[cite: 2]"""
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/v1/meta", status_code=200)
def get_meta():
    """Model, embedding, indeks versiyasi va snapshot sanasi[cite: 2]"""
    return {
        "pipeline_version": "0.4.2",
        "llm_model": get_llm_model(),
        "embedding_model": "demo-multilingual-embedding",
        "reranker": "demo-reranker-v1",
        "index_version": "lex-idx-2026-10-28",
        "lex_snapshot_date": "2026-10-28"
    }

@app.post("/v1/rag/batch", status_code=200)
def process_batch(
    payload: BatchRequest, 
    authorization: Optional[str] = Header(None)
):
    """
    1-10 tagacha savolni qabul qilish va javob berish[cite: 1, 2].
    """
    # 1. Autentifikatsiya tekshiruvi[cite: 2]
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Noto'g'ri yoki yo'q API kalit (Bearer token talab qilinadi)"
        )

    # 2. Batch hajmi validatsiyasi (1-10 ta)[cite: 2]
    if len(payload.items) < 1 or len(payload.items) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Batch hajmi 1 dan 10 tagacha savoldan iborat bo'lishi kerak"
        )

    # 3. Idempotentlik tekshiruvi[cite: 2]
    if payload.batch_id in BATCH_STORAGE:
        return BATCH_STORAGE[payload.batch_id]

    received_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    results = []
    ok_count = 0
    no_answer_count = 0
    error_count = 0

    # 4. Har bir savolni qayta ishlash (Demo mantiq)
    for item in payload.items:
        # Xatolar izolyatsiyasi: bitta savoldagi xato butun batch'ni yiqitmaydi[cite: 7]
        try:
            # Demo javoblar simulyatsiyasi
            if os.getenv("OPENAI_API_KEY"):
                generation_started = time.perf_counter()
                answer, input_tokens, output_tokens = generate_openai_answer(
                    item.question,
                    item.language or (payload.options.answer_language if payload.options else None),
                )
                generation_ms = int((time.perf_counter() - generation_started) * 1000)
                ok_count += 1
                results.append({
                    "id": item.id,
                    "status": "ok",
                    "answer": answer,
                    "answer_language": item.language or "uz-Latn",
                    "answer_source": "openai",
                    "no_answer": False,
                    "confidence": None,
                    "citations": [],
                    "contexts": [],
                    "debug": {"rewritten_query": item.question, "retrieved_total": 0},
                    "timing": {"retrieval_ms": 0, "rerank_ms": 0, "generation_ms": generation_ms, "total_ms": generation_ms},
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                    "error": None
                })
            elif "javobsiz" in item.question.lower() or "yo'q" in item.question.lower():
                # Qonunda javob bo'lmagan holat[cite: 6]
                no_answer_count += 1
                results.append({
                    "id": item.id,
                    "status": "ok",
                    "answer": "Aprel amaldagi qonunchilikda ushbu savolga javob ko'rsatilmagan.",
                    "answer_language": item.language or "uz-Latn",
                    "answer_source": "demo",
                    "no_answer": True,
                    "confidence": 0.95,
                    "citations": [],
                    "contexts": [],
                    "debug": {"rewritten_query": item.question, "retrieved_total": 0},
                    "timing": {"retrieval_ms": 100, "rerank_ms": 50, "generation_ms": 500, "total_ms": 650},
                    "usage": {"input_tokens": 150, "output_tokens": 20},
                    "error": None
                })
            else:
                ok_count += 1
                results.append({
                    "id": item.id,
                    "status": "ok",
                    "answer": f"\"{item.question}\" bo'yicha amaldagi normaga muvofiq tegishli chora belgilangan [1].",
                    "answer_language": item.language or "uz-Latn",
                    "answer_source": "demo",
                    "no_answer": False,
                    "confidence": 0.88,
                    "citations": [
                        {
                            "ref": 1,
                            "doc_id": "lex:1234567",
                            "doc_title": "O'zbekiston Respublikasining Mehnat kodeksi",
                            "doc_type": "kodeks",
                            "url": "https://lex.uz/docs/1234567",
                            "article": "115-modda",
                            "part": "1-qism",
                            "edition_date": "2023-04-30",
                            "doc_status": "amalda",
                            "chunk_id": "lex:1234567#art-115#p1",
                            "quote": "Xodimning haftalik ish vaqti muddati 40 soatdan oshmasligi kerak."
                        }
                    ],
                    "contexts": [
                        {
                            "rank": 1,
                            "chunk_id": "lex:1234567#art-115#p1",
                            "doc_id": "lex:1234567",
                            "article": "115-modda",
                            "text": "Xodim uchun ish vaqtining normal muddati haftasiga qirq soatdan ortiq bo'lishi mumkin emas.",
                            "retrieval_score": 0.82,
                            "rerank_score": 0.94,
                            "used_in_answer": True
                        }
                    ],
                    "debug": {"rewritten_query": item.question, "retrieved_total": 10},
                    "timing": {"retrieval_ms": 120, "rerank_ms": 60, "generation_ms": 800, "total_ms": 980},
                    "usage": {"input_tokens": 450, "output_tokens": 60},
                    "error": None
                })
        except Exception as e:
            error_count += 1
            results.append({
                "id": item.id,
                "status": "error",
                "answer": None,
                "citations": [],
                "contexts": [],
                "error": {
                    "code": "LLM_ERROR",
                    "message": str(e),
                    "retryable": True
                }
            })

    completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    response_data = {
        "run_id": payload.run_id,
        "batch_id": payload.batch_id,
        "status": "completed",
        "system": {
            "pipeline_version": "0.4.2",
            "llm_model": get_llm_model(),
            "embedding_model": "demo-multilingual-embedding",
            "reranker": "demo-reranker-v1",
            "index_version": "lex-idx-2026-10-28",
            "lex_snapshot_date": "2026-10-28"
        },
        "timing": {
            "received_at": received_at,
            "completed_at": completed_at,
            "total_ms": 1200
        },
        "summary": {
            "total": len(payload.items),
            "ok": ok_count,
            "no_answer": no_answer_count,
            "error": error_count
        },
        "results": results
    }

    # Natijani xotiraga saqlash
    BATCH_STORAGE[payload.batch_id] = response_data
    return response_data

@app.get("/v1/rag/batch/{batch_id}", status_code=200)
def get_batch_result(batch_id: str):
    """Batch natijasini qayta olish[cite: 2]"""
    if batch_id in BATCH_STORAGE:
        return BATCH_STORAGE[batch_id]
    raise HTTPException(status_code=404, detail="Batch topilmadi")