"""
LangGraph state graph and routing logic.
Defines the multi-agent workflow: Planning → Routing → Execution → Completion
"""

from typing import Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from src.agents.planner import planner_node
from src.agents.coder import coder_node
from src.agents.fast_utility import fast_utility_node
from src.config import get_routing_agent, ROUTING_RULES


class AgentState(TypedDict):
    """State passed between agent nodes in the graph."""
    user_task: str
    plan: str
    code_context: str
    implementation: str
    utility_output: str
    status: str
    messages: list
    next_agent: str


def routing_decision_node(state: AgentState) -> dict:
    """
    Router node - decides which agent to execute based on plan and task.
    
    This is a conditional node that reads the plan and routes to:
    - Coder for implementation tasks
    - Fast Utility for simple/quick tasks
    - Can loop back to Planner if task is still unclear
    """
    task = state.get("user_task", "")
    plan = state.get("plan", "")
    
    # Parse plan to determine routing
    plan_lower = plan.lower() if plan else ""
    
    # Check for explicit agent assignment in plan
    if "coder" in plan_lower or "implement" in plan_lower or "code" in plan_lower:
        state["next_agent"] = "coder"
    elif "fast_utility" in plan_lower or "quick" in plan_lower or "boilerplate" in plan_lower:
        state["next_agent"] = "fast_utility"
    else:
        # Default: use intelligent routing based on task keywords
        state["next_agent"] = get_routing_agent(task)
    
    return state


def should_route_to_coder(state: AgentState) -> bool:
    """Conditional edge: route to coder?"""
    return state.get("next_agent") == "coder"


def should_route_to_utility(state: AgentState) -> bool:
    """Conditional edge: route to fast utility?"""
    return state.get("next_agent") == "fast_utility"


def should_route_back_to_planner(state: AgentState) -> bool:
    """Conditional edge: route back to planner for clarification?"""
    return state.get("next_agent") == "planner"


def create_graph():
    """
    Create and compile the LangGraph state graph.
    
    Workflow:
    1. START
    2. planner_node (decompose task into steps)
    3. routing_decision_node (decide which agent to execute)
    4. Conditional edges:
       - to coder_node (for implementation)
       - to fast_utility_node (for quick tasks)
       - back to planner_node (if clarification needed)
    5. END
    
    Returns:
        Compiled graph executable
    """
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("router", routing_decision_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("fast_utility", fast_utility_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    # After planner, always go to router
    workflow.add_edge("planner", "router")
    
    # Router conditional edges
    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("next_agent", "coder"),
        {
            "coder": "coder",
            "fast_utility": "fast_utility",
            "planner": "planner",  # Loop back if needed
        }
    )
    
    # Terminal edges
    workflow.add_edge("coder", END)
    workflow.add_edge("fast_utility", END)
    
    # Compile graph
    graph = workflow.compile()
    
    return graph


# Lazy-load graph on first use
_graph_instance = None


def get_graph():
    """Get or create the compiled graph."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_graph()
    return _graph_instance


def invoke_graph(user_task: str, verbose: bool = True) -> dict:
    """
    Execute the graph with a user task.
    
    Args:
        user_task: User's request or task
        verbose: Whether to print intermediate steps
        
    Returns:
        Final state after all agent nodes complete
    """
    graph = get_graph()
    
    initial_state: AgentState = {
        "user_task": user_task,
        "plan": "",
        "code_context": "",
        "implementation": "",
        "utility_output": "",
        "status": "initialized",
        "messages": [],
        "next_agent": "",
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {user_task}")
        print(f"{'='*60}\n")
    
    # Execute graph
    final_state = graph.invoke(initial_state)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Execution complete!")
        print(f"Final status: {final_state.get('status', 'unknown')}")
        print(f"{'='*60}\n")
    
    return final_state


def stream_graph(user_task: str) -> None:
    """
    Stream graph execution with real-time output.
    
    Args:
        user_task: User's request or task
    """
    graph = get_graph()
    
    initial_state: AgentState = {
        "user_task": user_task,
        "plan": "",
        "code_context": "",
        "implementation": "",
        "utility_output": "",
        "status": "initialized",
        "messages": [],
        "next_agent": "",
    }
    
    print(f"\n{'='*60}")
    print(f"Task: {user_task}")
    print(f"{'='*60}\n")
    
    # Stream execution
    for step in graph.stream(initial_state):
        for key, value in step.items():
            print(f"[{key}]")
            if isinstance(value, dict):
                for k, v in value.items():
                    if k not in ["messages", "next_agent"]:
                        print(f"  {k}: {str(v)[:200]}...")
            print()
