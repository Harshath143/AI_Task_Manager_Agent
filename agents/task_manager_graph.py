# agents/task_manager_graph.py

from typing import TypedDict
import time
import uuid
import json

from langgraph.graph import StateGraph, END
# Note: LangChain Document is already imported indirectly via core/vector_store

# Assuming these imports from the defined structure
from core.models import Task
from core.vector_store import get_vector_store, add_task_to_store
from agents.task_categorizer import categorize_and_prioritize 

# --- 1. Define Graph State (Memory) ---
class AgentState(TypedDict):
    """
    Represents the state of our graph.
    This is passed between nodes and serves as the agent's memory.
    """
    input_task_str: str          # The original natural language input
    categorized_task: dict | None # The structured output from the LLM
    task_id: str                 # Unique ID for the task

# --- 2. Define Graph Nodes (Steps) ---

def categorize_node(state: AgentState) -> AgentState:
    """Node 1: Calls the LLM to categorize and prioritize the task."""
    print(f"\n--- Node: CATEGORIZING TASK ---")
    input_task = state["input_task_str"]
    
    structured_task_model = categorize_and_prioritize(input_task)
    
    return {
        "categorized_task": structured_task_model.model_dump(),
        "input_task_str": input_task,
        "task_id": str(uuid.uuid4())
    }

def store_node(state: AgentState) -> AgentState:
    """Node 2: Stores the structured task in the FAISS vector store."""
    print(f"--- Node: STORING TASK ---")
    
    cat_task = state["categorized_task"]
    
    final_task = Task(
        task_id=state["task_id"],
        original_input=state["input_task_str"],
        category=cat_task['category'],
        priority=cat_task['priority'],
        summary=cat_task['summary'],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )
    
    vectorstore = get_vector_store() # Retrieve the initialized store
    add_task_to_store(vectorstore, final_task)
    
    print(f"Task Stored: ID={final_task.task_id}, Priority={final_task.priority}")
    
    return state 

# --- 3. Build the LangGraph ---

def build_task_manager_graph():
    """Builds and compiles the Task Manager Agent's workflow graph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("categorize", categorize_node)
    workflow.add_node("store", store_node)

    workflow.set_entry_point("categorize")

    workflow.add_edge("categorize", "store")
    workflow.add_edge("store", END) # End the graph execution after storage

    app = workflow.compile()
    print("LangGraph Task Manager Compiled successfully.")
    return app