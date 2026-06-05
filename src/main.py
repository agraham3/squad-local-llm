"""
Main entry point for the multi-agent system.
Demonstrates task execution with different agent types and routing.
"""

import sys
import argparse
from src.graph import invoke_graph, stream_graph
from src.config import AGENTS, MODELS


def print_system_info():
    """Print system configuration."""
    print("\n" + "="*60)
    print("MULTI-AGENT SYSTEM - LOCAL OLLAMA")
    print("="*60)
    
    print("\n[MODELS] Available Models:")
    for model_name, model_config in MODELS.items():
        print(f"  • {model_name}")
        print(f"    Role: {model_config.role.value}")
        print(f"    Description: {model_config.description}")
        print(f"    Temp: {model_config.temperature}, Max tokens: {model_config.max_tokens}")
    
    print("\n[AGENTS] Available Agents:")
    for agent_name, agent_config in AGENTS.items():
        print(f"  • {agent_name}: {agent_config.role}")
        print(f"    Model: {agent_config.model.name}")
        print(f"    {agent_config.description}")
    
    print("\n" + "="*60 + "\n")


def example_planning_task():
    """Example: Let planner decompose a complex task."""
    task = """Design a Python FastAPI-based REST API for a todo application with:
    - CRUD operations for todos
    - User authentication with JWT
    - SQLAlchemy ORM integration
    - Environment variable configuration
    - Error handling and validation
    
    Break this down into clear implementation steps."""
    
    print("[EXAMPLE 1] Complex Planning Task")
    print(f"Task: {task[:100]}...\n")
    
    final_state = invoke_graph(task, verbose=True)
    
    print("\n[PLAN OUTPUT]")
    print(final_state.get("plan", "No plan generated")[:500])


def example_coding_task():
    """Example: Route directly to coder for implementation."""
    task = """Generate a Python function that:
    1. Reads a JSON file
    2. Filters items where count > 5
    3. Sorts by date descending
    4. Returns formatted results
    
    Include proper error handling and type hints."""
    
    print("[EXAMPLE 2] Direct Coding Task")
    print(f"Task: {task[:100]}...\n")
    
    final_state = invoke_graph(task, verbose=True)
    
    print("\n[IMPLEMENTATION OUTPUT]")
    print(final_state.get("implementation", "No implementation generated")[:500])


def example_quick_task():
    """Example: Route to fast utility for boilerplate."""
    task = """Generate a quick Python class template for a data model with:
    - name (str)
    - email (str)
    - created_at (datetime)
    - is_active (bool)
    
    Make it compatible with Pydantic."""
    
    print("[EXAMPLE 3] Quick Utility Task")
    print(f"Task: {task[:100]}...\n")
    
    final_state = invoke_graph(task, verbose=True)
    
    print("\n[RESULT OUTPUT]")
    print(final_state.get("utility_output", "No output generated")[:500])


def interactive_mode():
    """Interactive mode - get task from user and execute."""
    print("\n[INTERACTIVE MODE]")
    print("Enter your task (or 'quit' to exit):\n")
    
    while True:
        try:
            task = input("Task > ").strip()
            
            if task.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not task:
                print("Please enter a task.")
                continue
            
            # Ask if user wants streaming or normal execution
            print("\nExecute with streaming output? (y/n): ", end="")
            use_stream = input().lower() in ["y", "yes"]
            
            if use_stream:
                stream_graph(task)
            else:
                final_state = invoke_graph(task, verbose=True)
                
                # Print results
                if final_state.get("plan"):
                    print("\n[PLAN OUTPUT]")
                    print(final_state["plan"][:500])
                
                if final_state.get("implementation"):
                    print("\n[IMPLEMENTATION OUTPUT]")
                    print(final_state["implementation"][:500])
                
                if final_state.get("utility_output"):
                    print("\n[UTILITY OUTPUT]")
                    print(final_state["utility_output"][:500])
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            print("Please try again.\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Local multi-agent system with Ollama"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["interactive", "example1", "example2", "example3", "info"],
        default="info",
        help="Execution mode"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Custom task (when mode is not an example)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming output"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "info":
            print_system_info()
            print("\nUsage examples:")
            print("  python -m src.main interactive       # Interactive mode")
            print("  python -m src.main example1          # Planning task example")
            print("  python -m src.main example2          # Coding task example")
            print("  python -m src.main example3          # Quick task example")
            print("  python -m src.main --task 'your task'  # Custom task")
        
        elif args.mode == "interactive":
            print_system_info()
            interactive_mode()
        
        elif args.mode == "example1":
            print_system_info()
            example_planning_task()
        
        elif args.mode == "example2":
            print_system_info()
            example_coding_task()
        
        elif args.mode == "example3":
            print_system_info()
            example_quick_task()
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
