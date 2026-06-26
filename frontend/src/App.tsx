import { FormEvent, useEffect, useRef, useState } from "react";
import { FileUp, Send } from "lucide-react";
import { sendMessage, uploadDocument } from "./api/client";
import type { ChatResponse } from "./types/api";

type ChatItem = {
  role: "user" | "assistant";
  content: string;
  metadata?: ChatResponse;
};

export function App() {
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState<ChatItem[]>([]);
  const [selectedResponse, setSelectedResponse] = useState<ChatResponse | null>(null);
  const [uploadStatus, setUploadStatus] = useState("Henüz doküman yüklenmedi.");
  const [isBusy, setIsBusy] = useState(false);
  const chatWindowRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const chatWindow = chatWindowRef.current;
    if (!chatWindow) return;

    chatWindow.scrollTop = chatWindow.scrollHeight;
  }, [chat, isBusy]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed) return;

    setMessage("");
    setChat((items) => [...items, { role: "user", content: trimmed }]);
    setIsBusy(true);

    try {
      const response = await sendMessage(trimmed);
      setSelectedResponse(response);
      setChat((items) => [...items, { role: "assistant", content: response.answer, metadata: response }]);
    } catch {
      setChat((items) => [
        ...items,
        { role: "assistant", content: "Backend ile bağlantı kurulamadı. Sunucunun çalıştığını kontrol et." },
      ]);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleFileChange(file?: File) {
    if (!file) return;

    setUploadStatus("Doküman indeksleniyor...");
    try {
      const response = await uploadDocument(file);
      setUploadStatus(`${response.document_name} indekslendi: ${response.chunks_count} chunk.`);
    } catch {
      setUploadStatus("Doküman yüklenemedi. PDF, TXT, MD veya CSV dosyası deneyebilirsin.");
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Manufacturing Maintenance Agent</p>
            <h1>Bakım asistanı</h1>
          </div>
          <label className="upload-button">
            <FileUp size={18} />
            Doküman yükle
            <input type="file" accept=".pdf,.txt,.md,.csv" onChange={(event) => handleFileChange(event.target.files?.[0])} />
          </label>
        </header>

        <p className="upload-status">{uploadStatus}</p>

        <section className="chat-panel" aria-label="Sohbet">
          <section className="chat-window" ref={chatWindowRef}>
            {chat.length === 0 ? (
              <div className="empty-state">
                <strong>Alarm, bakım, güvenlik veya tarih sorusu sor.</strong>
                <span>Örnek: E42 alarm kodu ne anlama geliyor?</span>
              </div>
            ) : (
              chat.map((item, index) => (
                <article className={`message ${item.role}`} key={`${item.role}-${index}`}>
                  {item.content}
                </article>
              ))
            )}
          </section>

          <form className="composer" onSubmit={handleSubmit}>
            <input
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Sorunu yaz..."
              disabled={isBusy}
            />
            <button type="submit" disabled={isBusy} aria-label="Mesaj gönder">
              <Send className={isBusy ? "send-icon spinning" : "send-icon"} size={18} />
            </button>
          </form>
        </section>
      </section>

      <aside className="inspector">
        <section>
          <h2>Route</h2>
          <p>{selectedResponse?.route ?? "Henüz soru yok."}</p>
        </section>

        <section>
          <h2>Kaynaklar</h2>
          {selectedResponse?.sources.length ? (
            selectedResponse.sources.map((source) => (
              <div className="source-row" key={source.chunk_id}>
                <strong>{source.document_name}</strong>
                <span>{source.chunk_id} - {source.score.toFixed(2)}</span>
              </div>
            ))
          ) : (
            <p>Kaynak bulunmadı.</p>
          )}
        </section>

        <section>
          <h2>Tool çağrıları</h2>
          {selectedResponse?.tool_calls.length ? (
            selectedResponse.tool_calls.map((tool) => (
              <pre key={tool.name}>{JSON.stringify(tool, null, 2)}</pre>
            ))
          ) : (
            <p>Tool çağrısı yok.</p>
          )}
        </section>
      </aside>
    </main>
  );
}
