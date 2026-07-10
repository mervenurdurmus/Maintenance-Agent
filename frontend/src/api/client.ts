import axios from "axios";
import type {
  ChatResponse,
  AppSettings,
  ConversationInfo,
  DocumentInfo,
  HistoryMessage,
  HistoryToolCall,
  EvaluationStatus,
  UploadResponse,
} from "../types/api";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "/api",
});
const apiBaseUrl = import.meta.env.VITE_API_URL ?? "/api";
const assetBaseUrl = apiBaseUrl.replace(/\/api\/?$/, "");

export function resolveAssetUrl(url: string) {
  if (/^(https?:|blob:|data:)/i.test(url)) return url;
  return `${assetBaseUrl}${url}`;
}

export async function sendMessage(
  message: string,
  conversationId: string,
  image?: File | null,
) {
  if (image) {
    const formData = new FormData();
    formData.append("message", message);
    formData.append("conversation_id", conversationId);
    formData.append("image", image);
    const response = await api.post<ChatResponse>("/chat/image", formData);
    return response.data;
  }

  const response = await api.post<ChatResponse>("/chat", {
    message,
    conversation_id: conversationId,
  });
  return response.data;
}

export async function createConversation() {
  const response = await api.post<ConversationInfo>("/conversations");
  return response.data;
}

export async function listConversations() {
  const response = await api.get<ConversationInfo[]>("/conversations");
  return response.data;
}

export async function getConversationMessages(conversationId: string) {
  const response = await api.get<HistoryMessage[]>(
    `/conversations/${conversationId}/messages`,
  );
  return response.data;
}

export async function getConversationToolCalls(conversationId: string) {
  const response = await api.get<HistoryToolCall[]>(
    `/conversations/${conversationId}/tool-calls`,
  );
  return response.data;
}

export async function deleteConversation(conversationId: string) {
  const response = await api.delete(`/conversations/${conversationId}`);
  return response.data;
}

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post<UploadResponse>("/documents/upload", formData);
  return response.data;
}
export async function listDocuments() {
  const response = await api.get<DocumentInfo[]>("/documents");
  return response.data;
}

export async function deleteDocument(documentId: string) {
  const response = await api.delete(`/documents/${documentId}`);
  return response.data;
}

export async function getEvaluationStatus() {
  const response = await api.get<EvaluationStatus>("/evaluation/status");
  return response.data;
}

export async function getAppSettings() {
  const response = await api.get<AppSettings>("/settings");
  return response.data;
}

export async function updateChatLlmSettings(
  provider: "groq" | "openrouter",
  model: string,
) {
  const response = await api.patch<AppSettings>("/settings/chat-llm", {
    provider,
    model,
  });
  return response.data;
}
