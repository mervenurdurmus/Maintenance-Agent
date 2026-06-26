KEYWORD_WEIGHT = 0.08


def rerank_contexts(question: str, contexts: list[dict], top_n: int) -> list[dict]:
    terms = {term.lower() for term in question.split() if len(term) > 2}

    ranked = []
    for context in contexts:
        text = context["text"].lower()
        keyword_hits = sum(1 for term in terms if term in text)
        adjusted_score = float(context["score"]) + (keyword_hits * KEYWORD_WEIGHT)
        ranked.append({**context, "score": round(adjusted_score, 4)})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_n]
