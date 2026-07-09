import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from app.services.llm import create_chat_model  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

DATASET_PATH = ROOT_DIR / "evaluation" / "golden" / "golden_dataset.jsonl"
REPORT_PATH = ROOT_DIR / "evaluation" / "reports" / "ragas_report_v1.json"
CHAT_DB_PATH = BACKEND_DIR / "chat_history.db"
API_BASE_URL = "http://127.0.0.1:8000/api"
REQUEST_DELAY_SECONDS = 8
MAX_CHAT_ATTEMPTS = 4
REFUSAL_TERMS = (
    "kaynak yok",
    "doküman yok",
    "dokuman yok",
    "bulunmuyor",
    "bulunmamaktadır",
    "bilgi yok",
    "bilgi bulunmamaktadır",
    "bilgi bulunmuyor",
    "uyduramam",
    "kaynakta olmayan",
)


def load_golden_rows(dataset_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def post_json(path: str, payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    request = Request(
        f"{API_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ask_backend(question: str) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_CHAT_ATTEMPTS + 1):
        try:
            conversation = post_json("/conversations", {}, timeout=20)
            response = post_json(
                "/chat",
                {
                    "conversation_id": conversation["conversation_id"],
                    "message": question,
                },
                timeout=120,
            )
            answer = str(response.get("answer", ""))
            if "rate limit" not in answer.casefold():
                return response
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc

        time.sleep(REQUEST_DELAY_SECONDS * attempt)

    if last_error is not None:
        raise RuntimeError(f"Backend chat isteği başarısız oldu: {last_error}") from last_error

    return response


def context_texts_for_sources(sources: list[dict[str, Any]]) -> list[str]:
    chunk_ids = [
        source["chunk_id"]
        for source in sources
        if source.get("document_name") == "test-dokumani-bakim.txt" and source.get("chunk_id")
    ]
    if not chunk_ids:
        return []

    collection = get_vector_store().collection
    result = collection.get(ids=chunk_ids, include=["documents"])
    documents = result.get("documents") or []
    return [document for document in documents if document]


def build_eval_rows(golden_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []

    for index, row in enumerate(golden_rows, start=1):
        print(f"[{index}/{len(golden_rows)}] {row['id']} soruluyor...")
        response = ask_backend(row["question"])
        answer = str(response.get("answer", ""))
        sources = response.get("sources", [])
        contexts = context_texts_for_sources(sources)

        eval_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": row["ground_truth"],
                "category": row.get("category"),
                "must_include": row.get("must_include", []),
                "expected_source": row.get("expected_source"),
                "expected_behavior": row.get("expected_behavior"),
                "sources": sources,
            }
        )

        if index < len(golden_rows):
            time.sleep(REQUEST_DELAY_SECONDS)

    return eval_rows


def select_rows(
    rows: list[dict[str, Any]],
    ids: str | None,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    selected = rows

    if ids:
        wanted_ids = [item.strip() for item in ids.split(",") if item.strip()]
        rows_by_id = {row["id"]: row for row in rows}
        missing_ids = [row_id for row_id in wanted_ids if row_id not in rows_by_id]
        if missing_ids:
            raise RuntimeError(f"Dataset içinde bulunmayan id'ler: {', '.join(missing_ids)}")
        selected = [rows_by_id[row_id] for row_id in wanted_ids]

    if max_rows is not None:
        selected = selected[:max_rows]

    return selected


def build_eval_rows_from_chat_history(golden_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    connection = sqlite3.connect(CHAT_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        eval_rows: list[dict[str, Any]] = []
        for row in golden_rows:
            record = connection.execute(
                """
                SELECT assistant.content, assistant.sources_json
                FROM messages AS user
                JOIN messages AS assistant
                  ON assistant.conversation_id = user.conversation_id
                 AND assistant.role = 'assistant'
                 AND assistant.id > user.id
                WHERE user.role = 'user'
                  AND user.content = ?
                ORDER BY user.id DESC, assistant.id ASC
                LIMIT 1
                """,
                (row["question"],),
            ).fetchone()
            if record is None:
                raise RuntimeError(f"Sohbet geçmişinde cevap bulunamadı: {row['id']}")

            sources = json.loads(record["sources_json"] or "[]")
            eval_rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": record["content"],
                    "contexts": context_texts_for_sources(sources),
                    "ground_truth": row["ground_truth"],
                    "category": row.get("category"),
                    "must_include": row.get("must_include", []),
                    "expected_source": row.get("expected_source"),
                    "expected_behavior": row.get("expected_behavior"),
                    "sources": sources,
                }
            )

        return eval_rows
    finally:
        connection.close()


def create_eval_model(provider: str = "groq", model_name: str | None = None):
    if provider == "groq" and not model_name:
        return create_chat_model()

    settings = get_settings()
    if provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY tanımlı değil")

        return ChatGroq(
            model=model_name or settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
            max_tokens=8192,
        )

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY tanımlı değil")

        return ChatOpenAI(
            model=model_name or settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=4096,
        )

    raise ValueError(f"Desteklenmeyen LLM provider: {provider}")


def result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        records = json.loads(frame.to_json(orient="records", force_ascii=False))
    else:
        records = []

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        try:
            value = result[key]
        except Exception:
            continue
        if isinstance(value, list):
            numeric_values = [
                float(item)
                for item in value
                if isinstance(item, (int, float)) and item == item
            ]
            scores[key] = (
                sum(numeric_values) / len(numeric_values)
                if numeric_values
                else None
            )
        else:
            scores[key] = float(value) if value is not None else None

    return {
        "scores": scores,
        "rows": records,
    }


def _contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in terms)


def _must_include_score(row: dict[str, Any]) -> float | None:
    must_include = row.get("must_include") or []
    if not must_include:
        return None

    answer = str(row.get("answer", ""))
    matched = sum(1 for term in must_include if term.casefold() in answer.casefold())
    return matched / len(must_include)


def _behavior_passed(row: dict[str, Any]) -> bool:
    answer = str(row.get("answer", ""))
    expected_behavior = row.get("expected_behavior")

    if expected_behavior == "refuse_unknown":
        return _contains_any(answer, REFUSAL_TERMS)

    must_include = row.get("must_include") or []
    if must_include:
        return _must_include_score(row) >= 0.5

    return bool(answer.strip())


def behavior_scores(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not eval_rows:
        return {
            "overall_pass_rate": None,
            "by_behavior": {},
            "rows": [],
        }

    row_scores: list[dict[str, Any]] = []
    grouped: dict[str, list[bool]] = {}

    for row in eval_rows:
        expected_behavior = row.get("expected_behavior") or "belirsiz"
        passed = _behavior_passed(row)
        grouped.setdefault(expected_behavior, []).append(passed)
        row_scores.append(
            {
                "id": row.get("id"),
                "expected_behavior": expected_behavior,
                "passed": passed,
                "must_include_score": _must_include_score(row),
            }
        )

    by_behavior = {
        behavior: sum(results) / len(results)
        for behavior, results in grouped.items()
        if results
    }

    return {
        "overall_pass_rate": sum(item["passed"] for item in row_scores) / len(row_scores),
        "by_behavior": by_behavior,
        "rows": row_scores,
        "note": (
            "Bu skor Ragas answer_relevancy yerine geçmez; özellikle kaynakta_yok "
            "sorularında botun uydurmadan doğru şekilde reddedip reddetmediğini daha "
            "esnek kontrol eder."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--use-existing-answers",
        action="store_true",
        help="Dataset içindeki answer/context alanları doluysa backend'e sormadan değerlendirir.",
    )
    parser.add_argument(
        "--use-chat-history",
        action="store_true",
        help="Cevapları backend/chat_history.db içindeki en son eşleşen sohbetlerden alır.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["groq", "openrouter"],
        default=get_settings().ragas_llm_provider,
        help="Ragas yargılayıcı LLM sağlayıcısı.",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="Ragas yargılayıcı modelini değiştirir. Örn: openai/gpt-oss-20b veya openai/gpt-4o-mini",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Sadece virgülle ayrılmış soru id'lerini değerlendirir. Örn: q001,q008,q012",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Dataset başından en fazla bu kadar satır değerlendirir.",
    )
    args = parser.parse_args()

    golden_rows = select_rows(load_golden_rows(args.dataset), args.ids, args.max_rows)
    if args.use_chat_history:
        eval_rows = build_eval_rows_from_chat_history(golden_rows)
    elif args.use_existing_answers and all(row.get("answer") and row.get("contexts") for row in golden_rows):
        eval_rows = golden_rows
    else:
        eval_rows = build_eval_rows(golden_rows)

    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in eval_rows
        ]
    )

    llm = LangchainLLMWrapper(create_eval_model(args.llm_provider, args.eval_model))
    embeddings = LangchainEmbeddingsWrapper(get_vector_store().embedding_model)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=240, max_retries=8, max_wait=90, max_workers=1),
        batch_size=1,
        raise_exceptions=False,
    )

    report = result_to_dict(result)
    report["dataset_path"] = str(args.dataset)
    report["llm_provider"] = args.llm_provider
    report["eval_model"] = args.eval_model
    report["evaluated_rows"] = eval_rows
    report["behavior_scores"] = behavior_scores(eval_rows)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Ragas report written: {args.report}")
    print(json.dumps(report["scores"], ensure_ascii=False, indent=2))
    print(json.dumps(report["behavior_scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
