# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Running the Application
```bash
uv run main.py              # Run the main workflow
uv run main.py --debug      # Run in debug mode (fetches 5 papers regardless of date)
```

### Testing
```bash
# Manual testing via GitHub Actions workflow
# Go to Actions tab → "Test workflow" → "Run workflow"
# This runs the debug version with 5 papers
```

### Docker
```bash
docker-compose up --build   # Build and run with Docker
```

## Architecture

This is a Python application that recommends arXiv papers based on a user's Zotero library and sends daily email summaries.

### Core Components

1. **main.py** - Entry point and orchestration
   - Fetches Zotero corpus via API
   - Retrieves new arXiv papers
   - Coordinates recommendation and email sending

2. **recommender.py** - Paper recommendation engine
   - Uses sentence transformers for embedding generation (default: `google/embeddinggemma-300m`)
   - Implements time-decay weighting for recent Zotero papers
   - Calculates similarity scores between arXiv papers and Zotero library

3. **paper.py** - ArxivPaper class
   - Wrapper for arxiv.Result with additional features
   - Extracts paper content from LaTeX source
   - Integrates with Papers with Code for repository links
   - Generates TL;DR summaries via LLM

4. **llm.py** - LLM abstraction layer
   - Supports both local (llama-cpp-python) and API-based LLMs
   - Local: Qwen2.5-3B-Instruct (GGUF format)
   - API: OpenAI-compatible endpoints (configurable model)

5. **construct_email.py** - Email rendering and sending
   - HTML email templates with star ratings
   - SMTP integration for email delivery

### Data Flow

1. **Corpus Collection**: Fetch all papers from user's Zotero library with abstracts
2. **Filtering**: Apply gitignore-style patterns to exclude unwanted collections
3. **Paper Retrieval**: Get new arXiv papers from specified categories (ARXIV_QUERY)
4. **Embedding & Ranking**: Generate embeddings and score papers against Zotero corpus
5. **Content Enhancement**: Generate TL;DR summaries and fetch code repositories
6. **Email Generation**: Create HTML email with ranked paper list and send via SMTP

### Environment Configuration

Critical environment variables (see README.md for complete list):
- `ZOTERO_ID`, `ZOTERO_KEY` - Zotero API access
- `ARXIV_QUERY` - arXiv categories (e.g., "cs.AI+cs.CV+cs.LG+cs.CL")
- SMTP settings for email delivery
- `USE_LLM_API` - Toggle between local/API LLM
- `MAX_PAPER_NUM` - Limit papers processed (affects execution time)

### Deployment

- **GitHub Actions**: Primary deployment method with scheduled workflows
- **Docker**: Alternative deployment with cron scheduling
- **Local**: Direct execution with uv for development/testing

The recommendation algorithm uses time-weighted similarity scoring where newer papers in the Zotero library have higher influence on recommendations.