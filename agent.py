import os
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage



class GraphState(TypedDict):
    """State that flows through the graph."""
    question: str
    answer: str
    is_good: bool
    attempts: int
    feedback: str  



llm = None

def get_llm():
    
    global llm
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
    return llm



def generate_answer(state: GraphState):
    """Generate or refine an answer based on feedback."""
    question = state["question"]
    attempts = state["attempts"] + 1
    feedback = state.get("feedback", "")
    
    print(f"\nGenerating answer (attempt {attempts})...")

    # We modify the prompt to include the feedback if it exists
    if feedback:
        
        prompt = f"""
        You previously tried to answer this question but failed.
        
        QUESTION: {question}
        PREVIOUS ATTEMPT: {state['answer']}
        CRITIQUE FROM EVALUATOR: {feedback}

        Please provide a new answer that addresses the critique above and satisfies all constraints perfectly.
        """
    else:
        prompt = f"""
        You are a helpful AI assistant.
        Answer the following question clearly and accurately.

        Question: {question}
        """

    response = get_llm().invoke([HumanMessage(content=prompt)])

    return {
        "answer": response.content.strip(),
        "attempts": attempts,
    }


def evaluate_answer(state: GraphState) -> dict:
    """Evaluate and provide specific reasons for failure."""
    question = state["question"]
    answer = state["answer"]
    
    print("Evaluating answer...")

    
    prompt = f"""
    ### ROLE
    You are a high-level Quality Assurance Auditor.

    ### TASK
    Review the Generated Answer against the User Question. 
    If it is perfect, output 'VERDICT: YES'.
    If it is imperfect, output 'VERDICT: NO' followed by a specific 'FEEDBACK' explaining exactly what is wrong.

    - **User Question:** {question}
    - **Generated Answer:** {answer}

    ### INSTRUCTIONS
    - Be pedantic. Check word counts, facts, and formatting.
    - If the word count is wrong, state the actual count you found.
    - If a fact is wrong, state the correct fact.
    """

    response = get_llm().invoke([HumanMessage(content=prompt)])
    content = response.content.upper()
    
    is_good = "VERDICT: YES" in content
    
    
    feedback = ""
    if not is_good:
        # Simple extraction logic: everything after 'FEEDBACK'
        if "FEEDBACK" in content:
            feedback = response.content.split("FEEDBACK")[-1].strip(": \n")
        else:
            feedback = response.content # Fallback to full response

    print(f"Answer is {state['answer']}")
    print(f"Is good: {is_good}")
    print("Feedback:")
    print(feedback)
    print("-"*20)

    return {**state , "is_good": is_good, "feedback": feedback}



def decide_next_step(state: GraphState) -> str:
    """Decide whether to retry or finish."""
    if state["is_good"]:
        return "end"

    if state["attempts"] >= 10: # Lowered for testing
        print("   Max attempts reached, stopping...")
        return "end"

    print("Retrying...")
    return "generate_answer"


def build_graph():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(GraphState)

    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_answer", evaluate_answer)

    graph.add_edge(START, "generate_answer")
    graph.add_edge("generate_answer", "evaluate_answer")

    graph.add_conditional_edges(
        "evaluate_answer",
        decide_next_step,
        {
            "generate_answer": "generate_answer",
            "end": END,
        },
    )

    return graph.compile()



if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set.")
        sys.exit(1)

   
    print("LangGraph Feedback Loop Agent Demo")
    

    app = build_graph()
    question = input("\nEnter your question: ")

    initial_state: GraphState = {
        "question": question,
        "answer": "",
        "is_good": False,
        "attempts": 0,
        "feedback": ""
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("Final Answer:")
    print("=" * 50)
    print(result["answer"])
    print(f"\nStats: {result['attempts']} attempts.")