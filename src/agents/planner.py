"""
Planner Agent - Task decomposition and planning node.
Model: llama3.1:8b (fast reasoning)
Role: Break down complex tasks into structured steps, decide agent routing
"""

from langchain_ollama import OllamaLLM
from src.config import get_agent_config


def create_planner_node():
    """Create and configure the Planner agent."""
    agent_config = get_agent_config("planner")
    model_config = agent_config.model
    
    llm = OllamaLLM(
        model=model_config.name,
        base_url="http://localhost:11434",
        temperature=model_config.temperature,
    )
    
    return llm


def planner_node(state: dict) -> dict:
    """
    Planning node - decomposes user task into structured plan.
    
    Args:
        state: Graph state containing 'user_task'
        
    Returns:
        Updated state with 'plan' field containing structured steps
    """
    llm = create_planner_node()
    
    task = state.get("user_task", "")
    if not task:
        return state
    
    planning_prompt = f"""You are an expert task planner and architect.
    
Your job is to decompose the following task into clear, structured steps.
For each step, decide which agent should handle it (planner, coder, or fast_utility).
Output a JSON-structured plan.

Task: {task}

Provide output as a JSON object with this structure:
{{
  "task_title": "Brief title",
  "complexity": "simple|moderate|complex",
  "steps": [
    {{"step": 1, "action": "description", "assigned_agent": "planner|coder|fast_utility", "notes": "any context"}},
    ...
  ],
  "estimated_tokens": "rough estimate",
  "notes": "overall strategy or warnings"
}}

Think step-by-step, then output ONLY the JSON object."""

    response = llm.invoke(planning_prompt)
    
    state["plan"] = response
    state["status"] = "planning_complete"
    
    return state
