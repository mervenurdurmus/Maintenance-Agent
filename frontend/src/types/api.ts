export type Source = {
  document_name: string;
  chunk_id: string;
  score: number;
};

export type ToolCall = {
  name: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  tool_calls: ToolCall[];
  route: string;
};

export type UploadResponse = {
  document_id: string;
  document_name: string;
  chunks_count: number;
  status: "indexed";
};
