class RAGError(Exception):
    """Base class for all RAG system exceptions."""
    pass

class DatabaseError(RAGError):
    """Database read/write or connection errors."""
    pass

class DocumentNotFoundError(RAGError):
    """Raised when a requested document cannot be found."""
    pass

class InvalidFileTypeError(RAGError):
    """Raised when an unsupported file type is provided."""
    pass

class CorruptedPDFError(RAGError):
    """Raised when a PDF file is corrupted or unreadable."""
    pass

class DuplicateDocumentError(RAGError):
    """Raised when a document has already been ingested (using content hash)."""
    pass

class IngestionError(RAGError):
    """Generic error during ingestion pipeline stages."""
    pass

class EmbeddingError(RAGError):
    """Error generating query or chunk embeddings (API, Network, etc)."""
    pass

class VectorStoreError(RAGError):
    """Qdrant client, connection, lookup, or update errors."""
    pass

class RetrievalError(RAGError):
    """Errors during document or chunk retrieval stages."""
    pass

class GeminiResponseError(RAGError):
    """Gemini API call, generation, timeouts, or format errors."""
    pass

class SpeechRecognitionError(RAGError):
    """Audio input/recording or Whisper model failures."""
    pass

class TextToSpeechError(RAGError):
    """Piper synthesis or audio playback failures."""
    pass
