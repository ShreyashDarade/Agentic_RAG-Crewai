# 🚀 Agentic RAG System with CrewAI

A **production-grade Retrieval-Augmented Generation (RAG)** system built with multi-agent orchestration using **CrewAI**, **OpenAI GPT-4**, and **Milvus Cloud**.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![Milvus](https://img.shields.io/badge/Milvus-Cloud-purple.svg)
![CrewAI](https://img.shields.io/badge/CrewAI-Latest-red.svg)

## ✨ Key Features

### 🤖 Multi-Agent Architecture (CrewAI)

| Agent          | Role                  | Capabilities                                                       |
| -------------- | --------------------- | ------------------------------------------------------------------ |
| **Supervisor** | Query Planning        | Deep intent analysis, multi-step decomposition, execution planning |
| **Retriever**  | Information Retrieval | Multi-modal search, cross-reference expansion, source attribution  |
| **Generator**  | Response Synthesis    | Context-aware generation, citation integration, structured output  |
| **Feedback**   | Quality Assurance     | Validation, scoring, improvement suggestions                       |

### 📄 Advanced Document Processing

- **Multi-format Support**: PDF, DOCX, XLSX, PPTX, HTML, Markdown, images
- **EasyOCR**: Multilingual OCR (English, Hindi, German, French, Spanish+)
- **spaCy NLP**: Entity extraction, keyword detection, text cleanup
- **Cross-Reference Linking**: Automatic linking between text ↔ tables ↔ images

### 🔍 State-of-the-Art Retrieval

| Feature                      | Description                                   |
| ---------------------------- | --------------------------------------------- |
| **HNSW Index**               | High-performance vector search (Milvus Cloud) |
| **RRF Fusion**               | Combines dense + BM25 for hybrid search       |
| **Cross-Encoder Re-ranking** | Improved relevance with ms-marco model        |
| **MMR Diversity**            | Prevents redundant results                    |
| **Multi-Query Retrieval**    | Query variations for better coverage          |

### 💾 Production Infrastructure

- **LLM**: OpenAI GPT-4o-mini (with function calling)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector Store**: Milvus Cloud (Zilliz) with HNSW indexing
- **Streaming**: Real-time response generation

## 📁 Project Structure

```
Agentic_RAG-Crewai/
├── api/                      # FastAPI application
│   ├── main.py              # App entry point
│   ├── models/              # Pydantic models
│   └── routes/              # API routes
├── agents/                   # CrewAI agents
│   ├── supervisor_agent.py  # Query analysis & planning
│   ├── retriever_agent.py   # Multi-modal retrieval
│   ├── generator_agent.py   # Response synthesis
│   ├── feedback_agent.py    # Quality validation
│   └── tools/               # Agent tools
│       ├── milvus_tool.py   # Milvus search tool
│       └── online_search_tool.py
├── config/
│   ├── config.yaml          # Main configuration
│   ├── crew_config.yaml     # Agent configurations
│   └── .env                 # Environment variables
├── data_pipeline/
│   ├── chunker.py           # Cross-reference chunking
│   ├── ocr_processor.py     # EasyOCR + spaCy
│   ├── file_loader.py       # Multi-format loader
│   └── ingestion_pipeline.py
├── embeddings/
│   ├── openai_embedder.py   # OpenAI embeddings
│   └── milvus_store.py      # Milvus Cloud store
├── llm/
│   ├── openai_client.py     # OpenAI GPT client
│   └── base_llm.py
├── retriever/
│   └── advanced_retriever.py # RRF, re-ranking, MMR
├── orchestrator/
│   ├── crew_manager.py      # Agent orchestration
│   ├── memory_store.py
│   └── trace_logger.py
├── run.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API Key
- Milvus Cloud Account (Zilliz Cloud)

### 1. Clone and Setup

```bash
git clone https://github.com/ShreyashDarade/Agentic_RAG-Crewai.git
cd Agentic_RAG-Crewai

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

```bash
# Copy environment template
cp config/.env.example config/.env
```

Edit `config/.env`:

```env
# Required: OpenAI
OPENAI_API_KEY=sk-your-openai-key-here

# Required: Milvus Cloud (Zilliz)
MILVUS_URI=https://your-cluster.api.gcp-us-west1.zillizcloud.com
MILVUS_TOKEN=your-milvus-api-token

# Optional: Web Search
SERPER_API_KEY=
TAVILY_API_KEY=
```

### 3. Get Milvus Cloud Credentials

1. Go to [cloud.zilliz.com](https://cloud.zilliz.com)
2. Create a free cluster
3. Get your **Public Endpoint** (URI)
4. Create an **API Key** (Token)
5. Add to your `.env` file

### 4. Run the Application

```bash
python run.py
```

### 5. Access the API

```
API: http://localhost:8000
Docs: http://localhost:8000/docs
Health: http://localhost:8000/health
```

## 📡 API Endpoints

### Query Processing

```bash
# Multi-agent query processing
POST /api/v1/agent_query
Content-Type: application/json

{
  "query": "What are the key findings in the Q3 report?",
  "use_web_search": false
}
```

### Document Ingestion

```bash
# Ingest from directory
POST /api/v1/ingest
{
  "directory": "./data/raw"
}

# Upload file
POST /api/v1/ingest/upload
Content-Type: multipart/form-data
file: <your-document>
```

### System Health

```bash
GET /health

# Response
{
  "status": "healthy",
  "version": "2.0.0",
  "components": {
    "llm": {"status": "healthy", "provider": "openai"},
    "vector_store": {
      "status": "healthy",
      "provider": "milvus_cloud",
      "index_type": "HNSW",
      "document_count": 1250
    }
  }
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Layer                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CrewAI Orchestrator                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Supervisor│──│ Retriever │──│ Generator │──│ Feedback  │ │
│  │   Agent   │  │   Agent   │  │   Agent   │  │   Agent   │ │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  OpenAI GPT-4   │  │  Milvus Cloud   │  │   Web Search    │
│  (LLM Engine)   │  │  (HNSW Index)   │  │   (Optional)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 🔄 Retrieval Pipeline

```
Query → Multi-Query Generation
            │
            ▼
    ┌───────┴───────┐
    ▼               ▼
Dense Search    BM25 Search
(HNSW)          (Keywords)
    │               │
    └───────┬───────┘
            ▼
      RRF Fusion (k=60)
            │
            ▼
   Cross-Encoder Re-ranking
            │
            ▼
    MMR Diversity (λ=0.5)
            │
            ▼
      Final Results
```

## 📊 HNSW Index Configuration

The system uses HNSW (Hierarchical Navigable Small World) indexing for optimal search performance:

| Parameter        | Value  | Description                                    |
| ---------------- | ------ | ---------------------------------------------- |
| `M`              | 32     | Graph connectivity (higher = better recall)    |
| `efConstruction` | 360    | Build-time quality (higher = better index)     |
| `efSearch`       | 128    | Search-time quality (higher = better accuracy) |
| `metric_type`    | COSINE | Similarity metric for normalized embeddings    |

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.2

embedding:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 1536

vector_db:
  provider: "milvus_cloud"
  index_type: "HNSW"
  hnsw:
    m: 32
    ef_construction: 360
    ef_search: 128

retrieval:
  fusion_method: "rrf"
  enable_rerank: true
  enable_diversity: true
  enable_bm25: true

chunking:
  strategy: "semantic"
  enable_cross_reference: true
  enable_hierarchy: true
```

## 📈 Performance Tips

1. **Increase HNSW M** for better recall (costs more memory)
2. **Increase efSearch** for better accuracy (costs query time)
3. **Use text-embedding-3-large** for higher quality embeddings
4. **Enable GPU** for EasyOCR if processing many images

## 🧪 Testing

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v --cov=.
```

## 📝 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent framework
- [OpenAI](https://openai.com/) - LLM and embeddings
- [Milvus](https://milvus.io/) / [Zilliz Cloud](https://cloud.zilliz.com) - Vector database
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine
- [spaCy](https://spacy.io/) - NLP processing

---

**Built with ❤️ for Production AI Systems**
