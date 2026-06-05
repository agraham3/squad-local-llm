"""
Fast Utility Agent - Quick edits and simple tasks node.
Model: qwen2.5-coder:7b (fastest execution)
Role: Quick boilerplate, simple transformations, utility functions
"""

from langchain_ollama import OllamaLLM
from src.config import get_agent_config


def create_fast_utility_node():
    """Create and configure the Fast Utility agent."""
    agent_config = get_agent_config("fast_utility")
    model_config = agent_config.model
    
    llm = OllamaLLM(
        model=model_config.name,
        base_url="http://localhost:11434",
        temperature=model_config.temperature,
    )
    
    return llm


def fast_utility_node(state: dict) -> dict:
    """
    Fast utility node - handles quick edits, boilerplate, simple transformations.
    
    Args:
        state: Graph state
        
    Returns:
        Updated state with utility task results
    """
    llm = create_fast_utility_node()
    
    task = state.get("user_task", "")
    plan = state.get("plan", "")
    
    if not task:
        return state
    
    utility_prompt = f"""You are a quick, efficient code utility specialist.

Task: {task}

Plan context:
{plan}

Your goal is to:
1. Quickly generate boilerplate code, templates, or simple transformations
2. Keep responses focused and concise
3. Provide complete, working code
4. Prioritize speed and correctness over deep reasoning

Output the solution directly and concisely."""

    response = llm.invoke(utility_prompt)
    
    state["utility_output"] = response
    state["status"] = "utility_complete"
    
    return state
