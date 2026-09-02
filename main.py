
import os
import json
import math
import re
from pathlib import Path

import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

app = FastAPI()

# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from fastapi import FastAPI

# print("Current working directory:", os.getcwd())
# print(".env exists:", os.path.exists(".env"))

# env_path = Path(__file__).resolve().parent / ".env"

# print("Expected .env path:", env_path)
# print(".env exists at expected path:", env_path.exists())

# load_dotenv(env_path)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nvidia/nemotron-3-embed-1b"
)

MODELS = [
    os.getenv(
        "MODEL_1",
        "meta/llama-3.2-11b-vision-instruct"
    ),
    os.getenv(
        "MODEL_2",
        "openai/gpt-oss-20b"
    )
]

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "150"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
TOP_K = int(os.getenv("TOP_K", "5"))

DOCUMENT = ""
CHUNKS = []
EMBEDDINGS = None


def api_headers():
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY environment variable is not configured"
        )

    return {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def chunk_document(text):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return vectors / norms


def get_embeddings(texts, input_type):
    if not texts:
        return []

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float"
    }

    response = requests.post(
        f"{NVIDIA_BASE_URL}/embeddings",
        headers=api_headers(),
        json=payload,
        timeout=120
    )

    if not response.ok:
        raise RuntimeError(
            f"NVIDIA embedding API error: "
            f"{response.status_code} {response.text}"
        )

    data = response.json()["data"]
    data.sort(key=lambda item: item["index"])

    return [
        item["embedding"]
        for item in data
    ]


def index_document(text):
    global DOCUMENT
    global CHUNKS
    global EMBEDDINGS

    if not text or not text.strip():
        raise ValueError("Document is empty")

    DOCUMENT = text
    CHUNKS = chunk_document(text)

    if not CHUNKS:
        raise ValueError("Unable to create document chunks")

    vectors = get_embeddings(
        CHUNKS,
        "passage"
    )

    EMBEDDINGS = normalize_vectors(vectors)

    return len(CHUNKS)


def retrieve(query, top_k=TOP_K):
    if not CHUNKS or EMBEDDINGS is None:
        raise RuntimeError(
            "No indexed document available"
        )

    query_vector = get_embeddings(
        [query],
        "query"
    )[0]

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    norm = np.linalg.norm(query_vector)

    if norm:
        query_vector = query_vector / norm

    scores = EMBEDDINGS @ query_vector

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indices:
        results.append({
            "chunk_id": int(index),
            "text": CHUNKS[index],
            "score": round(
                float(scores[index]),
                6
            )
        })

    return results


def generate_answer(
    model,
    question,
    contexts
):
    context_text = "\n\n".join(
        f"[Context {i + 1}]\n{context}"
        for i, context in enumerate(contexts)
    )

    system_prompt = (
        "You are a retrieval augmented question answering "
        "assistant. Answer only from the supplied context. "
        "Do not introduce facts that are not supported by "
        "the context. If the answer cannot be determined "
        "from the context, say so."
    )

    user_prompt = (
        f"Retrieved context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer the question concisely. "
        "Mention the relevant context numbers when useful."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 500,
        "stream": False
    }

    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers=api_headers(),
        json=payload,
        timeout=120
    )

    if not response.ok:
        raise RuntimeError(
            f"NVIDIA generation API error: "
            f"{response.status_code} {response.text}"
        )

    data = response.json()

    return data["choices"][0]["message"]["content"]


def word_set(text):
    return set(tokenize(text))


def faithfulness(answer, contexts):
    answer_words = word_set(answer)

    if not answer_words:
        return 0.0

    context_words = set()

    for context in contexts:
        context_words.update(
            tokenize(context)
        )

    supported = answer_words & context_words

    return round(
        len(supported) / len(answer_words),
        4
    )


def answer_relevance(question, answer):
    question_words = word_set(question)
    answer_words = word_set(answer)

    if not question_words:
        return 0.0

    overlap = question_words & answer_words

    return round(
        len(overlap) / len(question_words),
        4
    )


def context_precision(question, contexts):
    if not contexts:
        return 0.0

    question_words = word_set(question)

    if not question_words:
        return 0.0

    relevant = 0

    for context in contexts:
        context_words = word_set(context)

        if question_words & context_words:
            relevant += 1

    return round(
        relevant / len(contexts),
        4
    )


def context_recall(gold_answer, contexts):
    if not gold_answer or not contexts:
        return 0.0

    gold_words = word_set(gold_answer)

    if not gold_words:
        return 0.0

    context_words = set()

    for context in contexts:
        context_words.update(
            tokenize(context)
        )

    covered = gold_words & context_words

    return round(
        len(covered) / len(gold_words),
        4
    )


def calculate_scores(
    question,
    answer,
    contexts,
    gold_answer=None
):
    scores = {
        "faithfulness": faithfulness(
            answer,
            contexts
        ),
        "answer_relevance": answer_relevance(
            question,
            answer
        ),
        "context_precision": context_precision(
            question,
            contexts
        ),
        "context_recall": context_recall(
            gold_answer,
            contexts
        )
    }

    scores["overall_score"] = round(
        sum(scores.values()) / len(scores),
        4
    )

    return scores


def run_rag(model, question, gold_answer=None):
    retrieved = retrieve(question)

    contexts = [
        item["text"]
        for item in retrieved
    ]

    answer = generate_answer(
        model,
        question,
        contexts
    )

    scores = calculate_scores(
        question,
        answer,
        contexts,
        gold_answer
    )

    return {
        "question": question,
        "answer": answer,
        "contexts": retrieved,
        "scores": scores
    }


def load_local_document():
    
    txt_files = list(Path(".").glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError("No .txt file found")

    path = txt_files[0]

    return path.read_text(
        encoding="utf-8"
    )



def load_local_questions():
    candidates = [
        "questions.json"
    ]

    for filename in candidates:
        path = Path(filename)

        if path.exists():
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

            if isinstance(data, dict):
                return (
                    data.get("questions")
                    or data.get("data")
                    or []
                )

            return data

    return []


def normalize_question(item):
    if isinstance(item, str):
        return {
            "question": item,
            "gold_answer": None
        }

    return {
        "question": (
            item.get("question")
            or item.get("query")
            or ""
        ),
        "gold_answer": (
            item.get("gold_answer")
            or item.get("answer")
            or item.get("reference_answer")
        )
    }


def evaluate_questions(questions):
    if not questions:
        raise ValueError(
            "No evaluation questions supplied"
        )

    model_results = {}

    for model in MODELS:
        results = []

        for item in questions:
            normalized = normalize_question(item)

            question = normalized["question"]
            gold_answer = normalized["answer"]

            if not question:
                continue

            result = run_rag(
                model,
                question,
                gold_answer
            )

            results.append(result)

        metric_names = [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "overall_score"
        ]

        aggregated = {}

        for metric in metric_names:
            values = [
                result["scores"][metric]
                for result in results
            ]

            aggregated[metric] = round(
                sum(values) / len(values),
                4
            ) if values else 0.0

        model_results[model] = {
            "scores": aggregated,
            "questions_evaluated": len(results),
            "results": results
        }

    leaderboard = []

    for model, result in model_results.items():
        leaderboard.append({
            "model": model,
            **result["scores"]
        })

    leaderboard.sort(
        key=lambda item: item["overall_score"],
        reverse=True
    )

    for rank, item in enumerate(
        leaderboard,
        start=1
    ):
        item["rank"] = rank

    return {
        "model_scores": model_results,
        "leaderboard": leaderboard
    }



@app.post("/")
def post(query: dict):
    """
    Main assessment endpoint.

    Supported operations:

    {"action": "load", "document": "..."}
    {"action": "index", "document": "..."}
    {"action": "query", "question": "..."}
    {"action": "evaluate", "questions": [...]}

    If action is omitted, the endpoint attempts to infer
    the requested operation from the supplied fields.
    """

    action = query.get("action")

    if action is None:
        if "questions" in query:
            action = "evaluate"
        elif "question" in query:
            action = "query"
        elif "document" in query:
            action = "load"
        else:
            action = "status"

    if action in {
        "load",
        "upload",
        "index"
    }:
        document = query.get("document")

        if document is None:
            document = load_local_document()

        if document is None:
            raise HTTPException(
                status_code=400,
                detail="No document supplied"
            )

        try:
            chunk_count = index_document(
                document
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc)
            )

        return {
            "status": "success",
            "document_loaded": True,
            "indexed": True,
            "chunks": chunk_count
        }

    if action in {
        "query",
        "rag"
    }:
        question = query.get("question")

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question is required"
            )

        if not CHUNKS:
            document = load_local_document()

            if document:
                index_document(document)

        if not CHUNKS:
            raise HTTPException(
                status_code=400,
                detail="No indexed document available"
            )

        results = {}

        for model in MODELS:
            try:
                results[model] = run_rag(
                    model,
                    question
                )
            except Exception as exc:
                results[model] = {
                    "error": str(exc)
                }

        return {
            "question": question,
            "results": results
        }

    if action in {
        "evaluate",
        "evaluation"
    }:
        questions = query.get("questions")

        if questions is None:
            questions = load_local_questions()

        if not CHUNKS:
            document = load_local_document()

            if document:
                index_document(document)

        if not CHUNKS:
            raise HTTPException(
                status_code=400,
                detail="No indexed document available"
            )

        try:
            return evaluate_questions(
                questions
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc)
            )

    if action == "status":
        return {
            "status": "ok",
            "document_loaded": bool(DOCUMENT),
            "chunks": len(CHUNKS),
            "models": MODELS,
            "embedding_model": EMBEDDING_MODEL,
            "api_key_configured": bool(
                NVIDIA_API_KEY
            )
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unknown action: {action}"
    )


@app.get("/key")
def get_key(key: str):
    """
    Returns configuration status without exposing
    the actual secret.
    """

    value = NVIDIA_API_KEY

    return {
        "key": key,
        "configured": value is not None
    }


@app.get("/info/{key}")
def get_info(key: str):
    if key == "models":
        return {
            "models": MODELS
        }

    if key == "embedding_model":
        return {
            "embedding_model": EMBEDDING_MODEL
        }

    if key == "chunks":
        return {
            "count": len(CHUNKS)
        }

    if key == "document":
        return {
            "loaded": bool(DOCUMENT),
            "chunks": len(CHUNKS)
        }

    if key == "config":
        return {
            "models": MODELS,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K
        }

    raise HTTPException(
        status_code=404,
        detail=f"Unknown info key: {key}"
    )

@app.get("/health", status_code=200)
def health():
    return {
        "status": "healthy"
    }

