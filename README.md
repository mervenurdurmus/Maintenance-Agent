# Manufacturing Maintenance Agent

Fabrika bakım ekipleri için AI bakım asistanı. Sistem bakım dokümanlarını indeksler, alarm ve bakım sorularını kaynaklı cevaplar, deterministik tool çağırır ve Ragas ile offline değerlendirilebilir.

## Mimari

Detaylı mimari dokümanı: [`docs/architecture.md`](docs/architecture.md)

- Backend: Python, FastAPI, Groq, ChromaDB, Pydantic
- Frontend: React, Vite, Axios
- Evaluation: Pytest, Ragas, golden dataset, evaluation reports

## Backend çalıştırma

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

`backend/.env` içinde `GROQ_API_KEY` girilirse model cevabı üretir. Anahtar yoksa sistem indeksleme ve routing katmanlarını çalıştırır, model cevabı için uyarı döner.

## Frontend çalıştırma

```bash
cd frontend
pnpm install
pnpm run dev
```

Uygulama varsayılan olarak `http://localhost:5173` adresinden açılır.

## Test

```bash
cd backend
pytest
```

## Ragas Evaluation

Ragas runtime içinde değil, offline değerlendirme aşamasında çalışır.

```bash
python evaluation/run_ragas.py
```

Önce `evaluation/golden/golden_dataset.jsonl` içindeki `answer` ve `contexts` alanlarını uygulama çıktılarıyla doldur.
