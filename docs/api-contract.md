# API Contract

## POST /api/chat

Request:

```json
{
  "message": "E42 alarm kodu ne anlama geliyor?",
  "conversation_id": "demo-session-1"
}
```

Response:

```json
{
  "answer": "E42 alarmı motor sıcaklığının limit üstüne çıktığını gösterir.",
  "sources": [
    {
      "document_name": "alarm_codes.pdf",
      "chunk_id": "doc_123_c12",
      "score": 0.84
    }
  ],
  "tool_calls": [],
  "route": "alarm_question"
}
```

## POST /api/documents/upload

Request: `multipart/form-data`, field name: `file`

Response:

```json
{
  "document_id": "doc_123",
  "document_name": "maintenance_manual.pdf",
  "chunks_count": 42,
  "status": "indexed"
}
```
