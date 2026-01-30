#  LangGraph Self-Correcting Agent

A simple and intuitive demonstration of **LangGraph** with **Groq** - featuring a self-correcting AI agent that generates, evaluates, and refines answers using a feedback loop.

##  Overview

This project showcases LangGraph's core concepts through a practical example:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │   Is Answer     │
│  START          │────▶│ Generate Answer │────▶│   Good?         │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                              ▲                          │
                              │                          │
                              │ NO (retry with feedback) │ YES
                              │                          │
                              └──────────────────────────▼
                                                   ┌─────────────────┐
                                                   │      END        │
                                                   └─────────────────┘
```

### Key Concepts Demonstrated

- **StateGraph**: Define state that flows through the workflow
- **Nodes**: Processing functions that transform state
- **Edges**: Connections between nodes
- **Conditional Edges**: Dynamic routing based on state
- **Feedback Loops**: Self-correction through iterative refinement

##  Quick Start

### 1. Prerequisites

- Python 3.10+
- A free Groq API key

### 2. Installation

```bash
# Clone or navigate to the project
cd agenticAI

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install langgraph langchain-groq langchain-core
```

### 3. Set up API Key

Get a free API key from [Groq Console](https://console.groq.com/keys), then:

```bash
export GROQ_API_KEY='your-api-key-here'
```

### 4. Run the Demo

```bash
python agent.py
```

##  Project Structure

```
agenticAI/
├── agent.py          # Main LangGraph demo with self-correcting agent
├── langgraph_demo.py # Alternative simpler demo
├── README.md         # This file
└── env/              # Virtual environment
```

##  How It Works

### State Definition

```python
class GraphState(TypedDict):
    question: str    # The user's question
    answer: str      # Generated answer
    is_good: bool    # Whether answer passed evaluation
    attempts: int    # Number of attempts made
    feedback: str    # Evaluator feedback for improvement
```

### Workflow Steps

1. **Generate Answer**: Uses Groq's LLM to generate an answer. If feedback exists from a previous attempt, it's included in the prompt for improvement.

2. **Evaluate Answer**: A QA auditor prompt checks if the answer is correct, complete, and well-formatted. Returns `VERDICT: YES` or `VERDICT: NO` with specific feedback.

3. **Decide Next Step**: 
   - If answer is good → END
   - If max attempts (10) reached → END
   - Otherwise → Retry with feedback

##  Example Output

```
==================================================
LangGraph Feedback Loop Agent Demo
==================================================

Enter your question: What is Python?

Generating answer (attempt 1)...
Evaluating answer...
Is good: True
------------------

==================================================
Final Answer:
==================================================
Python is a high-level, interpreted programming language known for 
its simple syntax and readability. It was created by Guido van Rossum 
and released in 1991. Python supports multiple programming paradigms 
including procedural, object-oriented, and functional programming.

Stats: 1 attempts.
```

##  Configuration

### Change the LLM Model

Edit `agent.py` to use a different Groq model:

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
    temperature=0.3  # Lower = more deterministic
)
```

### Adjust Max Attempts

```python
if state["attempts"] >= 10:  # Change this number
    return "end"
```

##  Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Groq API Docs](https://console.groq.com/docs)
- [LangChain Core](https://python.langchain.com/docs/)

##  License

MIT License - Feel free to use and modify!
