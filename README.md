# DAY10 - Multi-Agent RAG System

A production-ready, microservices-style Multi-Agent RAG (Retrieval-Augmented Generation) system using CrewAI, Groq, ChromaDB, and FastAPI.

## 🏗️ Architecture

```
DAY10/
├── agents/                      # CrewAI-based agent classes
│   ├── supervisor_agent.py      # Query planning and tool selection
│   ├── retriever_agent.py       # Document and web search
│   ├── generator_agent.py       # Answer synthesis
│   ├── feedback_agent.py        # Quality assurance
│   └── tools/                   # Agent tools
│       ├── chroma_tool.py       # ChromaDB search
│       ├── online_search_tool.py # Web search
│       └── summarize_tool.py    # Text summarization
├── orchestrator/                # Multi-agent coordination
│   ├── crew_manager.py          # Workflow orchestration
│   ├── memory_store.py          # Conversation memory
│   └── trace_logger.py          # Execution tracing
├── llm/                         # LLM abstraction layer
│   ├── base_llm.py              # LLM interface
│   ├── groq_client.py           # Groq implementation
│   └── prompt_templates/        # Agent prompts
├── embeddings/                  # Vector embeddings
│   ├── embedder.py              # Sentence transformers
│   ├── vector_store.py          # ChromaDB wrapper
│   └── chunk_tags.py            # Chunk tagging
├── retriever/                   # Information retrieval
│   ├── chroma_retriever.py      # Local search (BM25 + dense)
│   ├── web_retriever.py         # Web search
│   └── hybrid_retriever.py      # Combined retrieval
├── data_pipeline/               # Document processing
│   ├── file_loader.py           # Universal file parsing
│   ├── ocr_processor.py         # OCR for images/PDFs
│   ├── metadata_filter.py       # Document filtering
│   ├── chunker.py               # Text chunking
│   └── ingestion_pipeline.py    # Full ingestion flow
├── api/                         # FastAPI application
│   ├── routes/
│   │   ├── query.py             # Query endpoints
│   │   └── ingest.py            # Ingestion endpoints
│   ├── models/                  # Pydantic schemas
│   └── main.py                  # API entry point
├── config/                      # Configuration
│   ├── config.yaml              # Main config
│   ├── crew_config.yaml         # Agent config
│   └── env_example.txt          # Environment template
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Docker composition
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- Groq API key

### Installation

1. **Clone and navigate:**
   ```bash
   cd Agentic_RAG-Crewai-
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   # Copy and edit the environment template
   cp config/env_example.txt .env
   # Edit .env with your API keys
   ```

5. **Run the API:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## 📡 API Endpoints

### Query Processing

- `POST /api/v1/agent_query` - Process query through multi-agent system
- `POST /api/v1/search` - Direct search without full pipeline
- `GET /api/v1/trace/{trace_id}` - Get execution trace
- `GET /api/v1/history` - Get conversation history

### Document Ingestion

- `POST /api/v1/ingest` - Ingest documents from directory
- `POST /api/v1/ingest/file` - Ingest single file
- `POST /api/v1/ingest/upload` - Upload and ingest file
- `GET /api/v1/ingest/status` - Get ingestion status
- `GET /api/v1/ingest/files` - List ingested files

### Health & Status

- `GET /health` - Health check
- `GET /status` - Detailed status

## 📖 Usage Examples

### Query Example

```bash
curl -X POST "http://localhost:8000/api/v1/agent_query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of Python 3.12?",
    "include_sources": true,
    "include_trace": true
  }'
```

### Ingestion Example

```bash
# Ingest from directory
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "./data/raw",
    "force": false,
    "recursive": true
  }'

# Upload file
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@document.pdf"
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key (optional) | No |
| `SERPER_API_KEY` | Serper.dev API key (optional) | No |
| `TAVILY_API_KEY` | Tavily API key (optional) | No |
| `APP_ENV` | Environment (development/production) | No |
| `DEBUG` | Enable debug mode | No |
| `LOG_LEVEL` | Logging level | No |

### Config Files

- `config/config.yaml` - Main application configuration
- `config/crew_config.yaml` - Agent and workflow configuration

## 🔧 Features

### Multi-Agent Pipeline

1. **Supervisor Agent** - Analyzes queries, creates execution plans
2. **Retriever Agent** - Searches documents and web
3. **Generator Agent** - Synthesizes answers
4. **Feedback Agent** - Validates and improves responses

### Document Processing

- **Supported Formats:** PDF, DOCX, DOC, TXT, MD, HTML, CSV, XLSX, PPTX, Images
- **OCR:** Automatic OCR for scanned documents
- **Chunking:** Recursive, semantic, and fixed-size strategies
- **Deduplication:** Tracks processed files to avoid reprocessing

### Retrieval

- **Dense Search:** Semantic similarity with sentence-transformers
- **BM25:** Keyword-based retrieval
- **Fuzzy Matching:** Typo-tolerant search
- **Hybrid:** Combines local and web search

### Observability

- **Execution Traces:** Step-by-step tracking
- **Conversation Memory:** Context preservation
- **Health Checks:** Component monitoring

## 📊 Ingestion State Tracking

The system maintains a JSON file (`data/ingestion_state.json`) to track:

- Processed files and their hashes
- Ingestion timestamps
- Chunk counts
- Processing status

This prevents reprocessing of unchanged files.

## 🔒 Error Handling

- Comprehensive error handling at all levels
- Automatic retries with exponential backoff
- Fallback strategies when agents fail
- Detailed error messages and logging

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.

