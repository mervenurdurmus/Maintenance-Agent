import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { FileUp, Trash2 } from "lucide-react";

import { deleteDocument, listDocuments, uploadDocument } from "../api/client";
import { resolveAssetUrl } from "../api/client";
import type { DocumentInfo } from "../types/api";

export function FileUpload() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [status, setStatus] = useState("Dokümanlar yükleniyor...");
  const [isUploading, setIsUploading] = useState(false);
  const statusResetTimerRef = useRef<number | null>(null);

  function clearStatusResetTimer() {
    if (statusResetTimerRef.current) {
      window.clearTimeout(statusResetTimerRef.current);
      statusResetTimerRef.current = null;
    }
  }

  async function loadDocuments() {
    try {
      const items = await listDocuments();
      setDocuments(items);
      setStatus(items.length === 0 ? "Henüz doküman yüklenmedi." : `${items.length} doküman yüklü.`);
    } catch {
      setStatus("Doküman listesi alınamadı.");
    }
  }

  useEffect(() => {
    void loadDocuments();

    return () => {
      clearStatusResetTimer();
    };
  }, []);

  async function handleFileChange(file?: File) {
    if (!file) return;

    clearStatusResetTimer();
    setIsUploading(true);
    setStatus("Doküman indeksleniyor...");

    try {
      const response = await uploadDocument(file);
      await loadDocuments();
      setStatus(`${response.document_name} indekslendi: ${response.chunks_count} chunk.`);
    } catch (error) {
      setStatus(getUploadErrorMessage(error));
      statusResetTimerRef.current = window.setTimeout(() => {
        void loadDocuments();
      }, 12000);
    } finally {
      setIsUploading(false);
    }
  }

  function getUploadErrorMessage(error: unknown) {
    if (axios.isAxiosError(error)) {
      const detail = error.response?.data?.detail;
      if (typeof detail === "string" && detail.trim()) {
        return `Doküman yüklenemedi: ${detail}`;
      }
    }

    return "Doküman yüklenemedi. PDF, TXT, MD veya CSV dosyası deneyebilirsin.";
  }
  async function handleDeleteDocument(document: DocumentInfo) {
    clearStatusResetTimer();
    setStatus(`${document.document_name} siliniyor...`);

    try {
      await deleteDocument(document.document_id);
      await loadDocuments();
      setStatus(`${document.document_name} silindi.`);
    } catch {
      setStatus(`${document.document_name} silinemedi.`);
    }
  }

  return (
    <section className="file-upload" aria-label="Dokümanlar">
      <label className="upload-button">
        <FileUp size={18} />
        {isUploading ? "Yükleniyor..." : "Doküman yükle"}
        <input
          type="file"
          accept=".pdf,.txt,.md,.csv"
          disabled={isUploading}
          onChange={(event) => {
            void handleFileChange(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </label>

      <p className="upload-status" role="status">{status}</p>

      {documents.length > 0 && (
        <ul className="document-list">
          {documents.map((document) => (
            <li key={document.document_id}>
              {document.url ? (
                <a
                  href={resolveAssetUrl(document.url)}
                  target="_blank"
                  rel="noreferrer"
                  title={`${document.document_name} dosyasını aç`}
                >
                  {document.document_name}
                </a>
              ) : (
                <span>{document.document_name}</span>
              )}
              <button
                type="button"
                className="document-delete-button"
                aria-label={`${document.document_name} dosyasını sil`}
                title="Dokümanı sil"
                onClick={() => void handleDeleteDocument(document)}
              >
                <Trash2 size={13} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
