# API Contract

## POST /api/chat

Request:

```json
{
  "message": "E42 alarm kodu ne anlama geliyor?",
  "conversation_id": "chat_3bf7d6..."
}
```

`conversation_id` zorunludur. Yeni bir sohbet için önce
`POST /api/conversations` çağrılır; aynı sohbet içindeki tüm mesajlar dönen
kimliği kullanır.

## POST /api/conversations

Yeni ve benzersiz bir sohbet oturumu oluşturur.

## GET /api/conversations

Sohbetleri son güncellenme zamanına göre listeler.

## GET /api/conversations/{conversation_id}/messages

Seçilen sohbetin kullanıcı ve asistan mesajlarını kronolojik sırayla döndürür.

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
  "tool_calls": [
    {
      "name": "semantic_search",
      "input": {"query": "E42 alarm kodu ne anlama geliyor?"},
      "output": {"matches": []}
    }
  ]
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
