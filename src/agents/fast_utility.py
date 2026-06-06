"""
Fast Utility Agent - Quick edits and simple tasks node.
Model: qwen2.5-coder:7b (fastest execution)
Role: Quick boilerplate, simple transformations, utility functions
"""

from langchain_ollama import OllamaLLM
from src.config import get_agent_config


def _save_utility_log(input_state: dict, output: str) -> None:
    """Save fast utility agent execution log."""
    try:
        from src.graph import _save_agent_log
        _save_agent_log("fast_utility", input_state, output)
    except Exception:
        pass  # Silently fail if logging is not available


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
    
    print("\n[*] FAST_UTILITY - Generating quick boilerplate...")
    
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
    
    print("[OK] FAST_UTILITY - Boilerplate generated")
    
    # Save log
    _save_utility_log(state, response)
    
    state["utility_output"] = response
    state["status"] = "utility_complete"
    
    return state
