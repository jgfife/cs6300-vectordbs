# Agent Instructions for Vector Database Assignment

## Build/Test Commands
- `make install` - Install environment and dependencies (creates .virtual_environment)
- `make chroma` - Run the main ChromaDB demo
- `. .virtual_environment/bin/activate; python3 src/chroma.py` - Run chroma demo directly
- No test framework configured - run demos to verify functionality

## Code Style Guidelines
- Use Python 3.12+ with type hints (`from __future__ import annotations`)
- Follow PEP 8 formatting with 4-space indentation
- Use `snake_case` for variables/functions, `PascalCase` for classes
- Include docstrings for modules and main functions
- Import order: standard library, third-party, local imports
- Use pathlib.Path for file operations, not string paths
- Use descriptive variable names (e.g., `dbDir`, `docs_insert`)
- Add blank lines to separate logical sections
- Use f-strings for string formatting
- Include shebang `#!/usr/bin/env python3` for executable scripts

## Dependencies
- Core: chromadb>=0.5.0, sentence-transformers>=3.0.0, numpy>=1.26
- Optional: faiss-cpu>=1.7.4, pymilvus>=2.4.4

## Project Structure
- `src/` - Main source code directory
- `db/` - Database storage (auto-created)
- Virtual environment in `.virtual_environment/`