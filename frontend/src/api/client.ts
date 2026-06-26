import axios from "axios";
import type { ChatResponse, UploadResponse } from "../types/api";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000/api",
});

export async function sendMessage(message: string, conversationId = "demo-session-1") {
  const response = await api.post<ChatResponse>("/chat", {
    message,
    conversation_id: conversationId,
  });
  return response.data;
}

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post<UploadResponse>("/documents/upload", formData);
  return response.data;
}
