export type Source = {
  document_name: string;
  chunk_id: string;
  score: number;
  page_number?: number | null;
};

export type ImageAttachment = {
  filename: string;
  content_type: string;
  url: string;
};

export type ToolCall = {
  name: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  turn_id?: string | null;
};

export type HistoryToolCall = ToolCall & {
  id: number;
  created_at: string;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  tool_calls: ToolCall[];
  turn_id?: string | null;
  attachments?: ImageAttachment[];
};

export type ConversationInfo = {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
  created_at: string;
  turn_id?: string | null;
  sources?: Source[];
  attachments?: ImageAttachment[];
};

export type UploadResponse = {
  document_id: string;
  document_name: string;
  chunks_count: number;
  status: "indexed";
};

export type DocumentInfo = {
  document_id: string;
  document_name: string;
  url?: string | null;
};

export type AppSettings = {
  chat_model: string;
  vision_model: string;
  embedding_model: string;
  top_k: number;
  rerank_top_n: number;
  chunk_size: number;
  chunk_overlap: number;
  ragas_llm: {
    default_provider: "groq" | "openrouter";
    providers: Array<EvaluationLlmProvider & { models: string[] }>;
  };
};

export type EvaluationQuestion = {
  id: string;
  question: string;
  category: string;
  expected_behavior: string;
  expected_source: string;
};

export type EvaluationReportInfo = {
  exists: boolean;
  path: string;
  size_bytes: number;
  updated_at?: number | null;
  scores?: Record<string, number | null> | null;
};

export type EvaluationLlmProvider = {
  id: "groq" | "openrouter";
  label: string;
  model: string;
  configured: boolean;
};

export type EvaluationStatus = {
  dataset_path: string;
  total_questions: number;
  categories: Record<string, number>;
  expected_behaviors: Record<string, number>;
  expected_sources: Record<string, number>;
  ragas_llm: {
    default_provider: "groq" | "openrouter";
    providers: EvaluationLlmProvider[];
  };
  questions: EvaluationQuestion[];
  report: EvaluationReportInfo;
};
