"""
Coder Agent - Code implementation and refactoring node.
Model: qwen3-coder (highest capability for coding)
Role: Generate, refactor, debug code - has full tool access
"""

from langchain_ollama import OllamaLLM
from src.config import get_agent_config
from src.squad_runtime import extract_file_writes, write_agent_files


def _save_coder_log(input_state: dict, output: str) -> None:
    """Save coder agent execution log."""
    try:
        from src.graph import _save_agent_log
        _save_agent_log("coder", input_state, output)
    except Exception:
        pass  # Silently fail if logging is not available


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
    
    print("\n[*] CODER - Generating production-ready code...")
    
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
4. When creating or updating project files, use this exact format for every file:

FILE: relative/path/from/project/root.py
```python
exact file contents here
```

Output:
- First, briefly explain your implementation strategy
- Then provide the complete code using FILE blocks for files that should be saved
- Finally, explain any design decisions made

Do not put file contents only in prose. If the task requires a file, include a FILE block.
Focus on correctness and clarity over brevity."""

    response = llm.invoke(coding_prompt)
    files_written = []
    try:
        file_writes = extract_file_writes(response)
        if file_writes:
            files_written = write_agent_files(file_writes)
            print(f"[*] CODER - Wrote {len(files_written)} file(s) to project")
    except Exception as exc:
        state["status"] = "file_write_failed"
        state["implementation"] = response
        state["file_write_error"] = str(exc)
        print(f"[!] CODER - File write failed: {exc}")
        return state
    
    print("[OK] CODER - Implementation complete")
    
    # Save log
    _save_coder_log(state, response)
    
    state["implementation"] = response
    state["files_written"] = files_written
    state["status"] = "implementation_complete"
    
    return state
