# Building AI Agents with LangGraph

This repository contains a multi-agent system built with LangGraph, LangChain, and OpenAI. Learn how to build sophisticated AI agents that can collaborate to solve complex tasks.

## Prerequisites

- Python 3.12 or higher
- uv (Python package manager)
- API keys for OpenAI, Tavily, and Alpha Vantage

## Installation Instructions

### 1. Install uv

#### Windows
```powershell
pip install uv
```

#### macOS/Linux
```bash
pip install uv
```

Alternatively, you can install uv using curl:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/building-ai-agents-with-langgraph.git
cd building-ai-agents-with-langgraph
```

### 3. Install Dependencies

```bash
uv sync
```

This will create a virtual environment and install all required dependencies specified in `pyproject.toml`.

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key_here

# Optional: LangSmith Tracing (for debugging and monitoring)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
# LANGCHAIN_API_KEY="your_langsmith_api_key_here"
# LANGCHAIN_PROJECT="your_project_name"
```

To obtain the required API keys:
- OpenAI API key: [OpenAI Platform](https://platform.openai.com/)
- Tavily API key: [Tavily AI](https://tavily.com/)
- Alpha Vantage API key: [Alpha Vantage](https://www.alphavantage.co/)

The LangSmith tracing configuration is optional but recommended for debugging and monitoring your LangChain applications.

### 5. Run the Application

```bash
uv run python main.py
```

This will start the multi-agent system in your terminal. You can interact with the agents by typing your queries.

## Project Structure

```
building-ai-agents-with-langgraph/
├── agents/             # Agent implementations
├── graph/              # Graph workflow definitions
├── tools/              # Custom tools
├── .env                # Environment variables (create this file)
├── agent_config.json   # Agent configuration
├── main.py             # Main application entry point
├── pyproject.toml      # Project configuration
└── README.md           # This file
```

## Dependencies

The project uses the following main dependencies:
- LangGraph (^0.2.35)
- LangChain (^0.3.3)
- LangChain OpenAI (^0.2.2)
- OpenAI (^1.51.2)
- Jupyter Lab (^4.2.5)
- Matplotlib (^3.9.2)

For a complete list of dependencies, see `pyproject.toml`.

## Troubleshooting

### Common Issues

1. **uv not found**
   - Make sure you have Python 3.12+ installed
   - Try: `pip install uv` or `python -m pip install uv`

2. **Python version mismatch**
   ```bash
   uv python pin 3.12
   ```

3. **Missing API keys**
   - Ensure all required API keys are set in your `.env` file

### uv Commands Reference

- Update dependencies: `uv lock`
- Add a new dependency: `uv add package_name`
- Remove a dependency: `uv remove package_name`
- Show installed packages: `uv pip list`
- Run a command in the virtual environment: `uv run command`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
