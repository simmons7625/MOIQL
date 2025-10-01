# MOIQL Project Rules

## Development Guidelines

### Environment Setup
- Use `uv` for Python package management
- **NEVER use `pip install`** - always use `uv add` for adding dependencies
- Pre-commit hooks with ruff are configured

### Code Style
- Use ruff for linting and formatting
- Follow Python best practices for RL research code
- **Code comments are for explaining code logic only** - never use comments to communicate with the user or leave conversational notes

### Testing
- Test implementations on both Deep Sea Treasure and Highway environments
- Verify multi-objective learning behavior

### Documentation
- Document Pareto frontier results
- Explain trade-offs between objectives
- Keep README.md updated with experiment results
