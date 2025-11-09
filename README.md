# AI-Based Task Manager Agent

An enterprise-grade, open-source AI agent designed to efficiently categorize, prioritize, and store natural language tasks using a local Large Language Model (LLM) and a vector database.

## ✨ Features

- **Natural Language Processing**: Accepts free-form text input for tasks (via the ADD command)
- **Structured Output & Reliability**: Uses Pydantic to force the LLM to output reliable, clean JSON with predefined categories and priorities
- **Local LLM Integration**: Powered by Ollama for zero-cost, local inference, ensuring data privacy and control
- **Semantic Search**: Utilizes FAISS (via Ollama embeddings) for powerful task retrieval based on meaning
- **Orchestration**: Implements LangGraph for robust, state-managed multi-step workflows

##  Architecture and Components

| Component | Technology | Role |
|-----------|------------|------|
| Orchestration | LangGraph | Manages the sequential workflow |
| LLM Inference | Ollama (llama3:8b) | Performs categorization and summarization |
| Embeddings | Ollama (nomic-embed-text) | Generates vector representations |
| Vector DB | FAISS | Stores task embeddings for semantic search |
| Data Structure | Pydantic | Enforces reliable JSON output |

## 🚀 Setup and Installation

### Prerequisites
- Python 3.10+
- Ollama running on `http://127.0.0.1:11434`

### Quick Start

```bash
# Clone repository
git clone [YOUR_REPO_URL]
cd task_manager_agent

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama3:8b
ollama pull nomic-embed-text

# Run the agent
python main.py
```

## 💻 Usage and Commands

| Command | Description | Example |
|---------|------------|---------|
| ADD | Add new task | `ADD Write the user documentation` |
| FIND | Search tasks | `FIND tasks related to Q4 budget` |
| HELP | Show commands | `HELP` |
| EXIT | Quit app | `EXIT` |

### Example Usage

```bash
Task Manager > ADD We need to implement the new login feature immediately.

✅ Task Successfully Processed and Stored!

{
    "task_id": "...",
    "status": "Pending", 
    "original_input": "We need to implement the new login feature immediately.",
    "summary": "Implement the new login feature.",
    "category": "Development",
    "priority": "High"
}
```

## 📁 Project Structure

```
/task_manager_agent
├── main.py                 # CLI and command dispatcher
├── requirements.txt        # Dependencies
├── agents/                 # LLM and Orchestration Logic
├── core/                   # Utilities and Models
└── data/                  # FAISS index storage
```

## Video Demo and Explanation

```
https://github.com/Harshath143/AI_Task_Manager_Agent/blob/main/AI-Based%20Task%20Manager%20Agent%20Presentation%20%F0%9F%9A%80.mp4
```
