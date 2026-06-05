"""
Configuration for models, agents, and routing logic.
All Ollama models are configured to use local OpenAI-compatible endpoint.
"""

import os
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class ModelRole(str, Enum):
    """Agent role classification for task routing."""
    PLANNING = "planning"
    CODING = "coding"
    UTILITY = "utility"


class OllamaModel(BaseModel):
    """Configuration for a single Ollama model."""
    name: str = Field(..., description="Model identifier (e.g., 'llama3.1:8b')")
    base_url: str = Field(..., description="Ollama API base URL")
    api_key: str = Field(..., description="API key (dummy value for local Ollama)")
    temperature: float = Field(default=0.7, description="LLM temperature (0=deterministic, 1=creative)")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    role: ModelRole = Field(..., description="Primary role for this model")
    description: str = Field(default="", description="Human-readable description")


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    name: str = Field(..., description="Agent identifier")
    model: OllamaModel = Field(..., description="LLM model instance")
    role: str = Field(..., description="Agent's role (planner/coder/utility)")
    description: str = Field(..., description="Agent description")


# Load environment variables
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")

# Define available models with role-specific tuning
MODELS: Dict[str, OllamaModel] = {
    "llama3.1:8b": OllamaModel(
        name="llama3.1:8b",
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=0.3,  # Low temp for structured planning
        max_tokens=2048,
        role=ModelRole.PLANNING,
        description="Fast reasoning for task planning and decomposition"
    ),
    "qwen3-coder": OllamaModel(
        name="qwen3-coder:latest",
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=0.1,  # Very low temp for precise code
        max_tokens=4096,
        role=ModelRole.CODING,
        description="Highest-capability coder for complex implementations"
    ),
    "qwen2.5-coder:7b": OllamaModel(
        name="qwen2.5-coder:7b",
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        temperature=0.2,  # Low temp, but slightly more than qwen3
        max_tokens=2048,
        role=ModelRole.UTILITY,
        description="Fast execution for quick edits and boilerplate"
    ),
}

# Define agents
AGENTS: Dict[str, AgentConfig] = {
    "planner": AgentConfig(
        name="planner",
        model=MODELS["llama3.1:8b"],
        role="planning",
        description="Decomposes tasks into structured steps, decides routing to other agents"
    ),
    "coder": AgentConfig(
        name="coder",
        model=MODELS["qwen3-coder"],
        role="coding",
        description="Implements code changes, refactors, and complex solutions"
    ),
    "fast_utility": AgentConfig(
        name="fast_utility",
        model=MODELS["qwen2.5-coder:7b"],
        role="utility",
        description="Quick edits, boilerplate generation, simple transformations"
    ),
}


# Routing rules: task characteristics → agent selection
ROUTING_RULES: Dict[str, Dict] = {
    "planning": {
        "agents": ["planner"],
        "keywords": ["design", "plan", "architecture", "strategy", "break down", "steps"],
        "description": "Route to Planner for high-level reasoning"
    },
    "coding": {
        "agents": ["coder"],
        "keywords": ["implement", "refactor", "debug", "code", "function", "class", "fix"],
        "description": "Route to Coder for complex code generation"
    },
    "quick": {
        "agents": ["fast_utility", "coder"],
        "keywords": ["quick", "simple", "fast", "template", "boilerplate", "generate"],
        "description": "Route to Fast Utility first, fallback to Coder if needed"
    },
}


def get_agent_config(agent_name: str) -> Optional[AgentConfig]:
    """Retrieve agent configuration by name."""
    return AGENTS.get(agent_name)


def get_model_config(model_name: str) -> Optional[OllamaModel]:
    """Retrieve model configuration by name."""
    return MODELS.get(model_name)


def get_routing_agent(task: str) -> str:
    """
    Intelligently route task to appropriate agent based on keywords.
    Returns agent name.
    """
    task_lower = task.lower()
    
    # Check for explicit routing keywords
    for route_type, config in ROUTING_RULES.items():
        for keyword in config["keywords"]:
            if keyword in task_lower:
                return config["agents"][0]
    
    # Default: planner routes and decomposes
    return "planner"


if __name__ == "__main__":
    # Validate configuration on import
    print("✓ Configuration loaded successfully")
    print(f"\nAvailable models:")
    for model_name, model_config in MODELS.items():
        print(f"  - {model_name}: {model_config.role.value} ({model_config.description})")
    
    print(f"\nAvailable agents:")
    for agent_name, agent_config in AGENTS.items():
        print(f"  - {agent_name}: {agent_config.role} → {agent_config.model.name}")
