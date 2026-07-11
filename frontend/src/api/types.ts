// Mirrors backend/models/schemas.py — keep in sync with the FastAPI DTOs.

export type SearchMode = "hybrid" | "keyword" | "vector";

export interface ChunkHit {
  id: string;
  pdf_id: string;
  pdf_name: string;
  page_start: number;
  page_end: number;
  chunk_index: number;
  content: string;
  file_url?: string | null;
  score?: number | null;
  text_match?: number | null;
  vector_distance?: number | null;
}

export interface SearchResponse {
  query: string;
  mode: string;
  count: number;
  hits: ChunkHit[];
}

export interface IngestResponse {
  pdf_id: string;
  filename: string;
  sha256: string;
  num_pages: number;
  num_chunks: number;
  status: string;
}

export interface Citation {
  pdf_id: string;
  pdf_name: string;
  page_start: number;
  page_end: number;
  chunk_index: number;
  file_url?: string | null;
}

export interface ChatResponse {
  query: string;
  answer: string;
  citations: Citation[];
  chunks_used: ChunkHit[];
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface PdfStorageInfo {
  num_files: number;
  total_size_bytes: number;
  total_size_human: string;
}

export interface StorageResponse {
  typesense_data_dir: string;
  disk_used_bytes: number;
  disk_used_human: string;
  disk_total_bytes: number;
  disk_total_human: string;
  disk_used_pct: number;
  pdf_storage: PdfStorageInfo;
}

// Backend error envelope: core/exceptions.py
export interface ApiError {
  error: string;
  message: string;
  detail?: unknown;
}
