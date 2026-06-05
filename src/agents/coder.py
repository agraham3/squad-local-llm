"""
Coder Agent - Code implementation and refactoring node.
Model: qwen3-coder (highest capability for coding)
Role: Generate, refactor, debug code - has full tool access
"""

from langchain_ollama import OllamaLLM
from src.config import get_agent_config


def create_coder_node():
    """Create and configure the Coder agent."""
    agent_config = get_agent_config("coder")
    model_config = agent_config.model
    
    llm = OllamaLLM(
        model=model_config.name,
        base_url="http://localhost:11434",
        temperature=model_config.temperature,
    )
    
    return llm


def coder_node(state: dict) -> dict:
    """
    Coder node - implements code solutions.
    
    Args:
        state: Graph state containing 'user_task', 'plan', and optional 'code_context'
        
    Returns:
        Updated state with 'implementation' field containing generated code
    """
    llm = create_coder_node()
    
    task = state.get("user_task", "")
    plan = state.get("plan", "")
    context = state.get("code_context", "")
    
    if not task:
        return state
    
    # Build prompt with context
    coding_prompt = f"""You are an expert software engineer specializing in writing clean, production-ready code.

Task: {task}

Plan from architect:
{plan}

Current code context (if any):
{context if context else "(no existing code context)"}

Your responsibilities:
1. Generate or refactor code according to the plan
2. Ensure code is clean, well-commented, and follows best practices
3. Provide complete, working code that can be directly used

Output:
- First, briefly explain your implementation strategy
- Then provide the complete code
- Finally, explain any design decisions made

Focus on correctness and clarity over brevity."""

    response = llm.invoke(coding_prompt)
    
    state["implementation"] = response
    state["status"] = "implementation_complete"
    
    return state
