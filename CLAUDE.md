# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Build and Installation
```bash
# Install all dependencies including optional extras
poetry install --extras all

# Install specific extras (e.g., for Apple Silicon)
poetry install --extras all --extras apple

# Sync dependencies
poetry sync --extras all
```

### Code Quality
```bash
# Run linters and formatters
make lint
# Or directly:
poetry run ruff format --preview src/ tests/
poetry run black --preview --enable-unstable-feature=string_processing src/ tests/
poetry run ruff check --fix src/ tests/
```

### Testing
```bash
# Run all tests
make test
# Or:
poetry run pytest --verbose

# Run tests with coverage
poetry run pytest --cov=src/ --cov-report=xml

# Run specific tests
poetry run pytest tests/path/to/test.py -k "test_name"

# Run tests matching a pattern
poetry run pytest -k "agent or model"
```

## High-Level Architecture

### Core Components

**Model System** (`src/avalan/model/`)
- Multi-modal model support (text, vision, audio) with unified interfaces
- Backend abstraction layer supporting transformers, vLLM, mlx-lm
- Vendor integrations through AI URIs (`ai://` protocol)
- Each modality has its own module with specialized models

**Agent Framework** (`src/avalan/agent/`)
- Template-based agent configuration using TOML files
- Event-driven architecture with streaming support
- Tool integration system with native and custom tools
- Reasoning strategies (ReACT, Chain-of-Thought, etc.)

**Memory Management** (`src/avalan/memory/`)
- Unified memory API with multiple backends (PostgreSQL/pgvector, Elasticsearch, AWS S3)
- Recent memory for conversation context
- Permanent memory stores for long-term knowledge
- Document partitioning and embedding support

**CLI Interface** (`src/avalan/cli/`)
- Main entry points: `avalan` and `avl` commands
- Subcommands: agent, model, memory, flow, deploy
- Rich terminal output with streaming support

**Server Components** (`src/avalan/server/`)
- OpenAI API-compatible endpoints
- MCP (Model Context Protocol) support
- FastAPI-based REST API

### Key Design Patterns

1. **AI URI System**: Unified resource identifier for models and engines
   - Format: `ai://[key@]provider/model[/options]`
   - Enables seamless switching between local and cloud models

2. **Engine Abstraction**: Models are loaded through engine-specific implementations
   - TransformerEngine for HuggingFace models
   - VLLMEngine for optimized inference
   - MLXEngine for Apple Silicon optimization

3. **Event-Driven Streaming**: All model interactions support async streaming
   - Token-by-token generation for text models
   - Progress events for long-running operations
   - Tool call events for agent interactions

4. **Configuration-First Agents**: Agents defined in TOML with templates
   - Role, task, instructions, and rules sections
   - Template variables for dynamic configuration
   - Tool and memory configuration

### Testing Strategy

- Unit tests in `tests/` mirror source structure
- Integration tests for CLI commands
- Mock-based testing for external services
- Coverage reporting with pytest-cov

### Development Workflow

1. Make changes to source files in `src/avalan/`
2. Run `make lint` to ensure code quality
3. Run `make test` to verify functionality
4. For new features, add corresponding tests in `tests/`
5. Update documentation if adding new CLI commands or major features