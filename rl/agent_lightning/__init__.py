"""
Agent Lightning Framework Integration.
Implements emergent communication and reinforcement learning for multi-agent systems.

Based on the Agent Lightning paper: "Training-Agent Disaggregation for Emergent Communication"
"""

from .agentops_integration import (
    AgentOpsMonitor,
    get_monitor,
    initialize_monitoring
)

from .emergent_communication import (
    EmergentCommunicationManager,
    CommunicationProtocol,
    CommunicationIntent
)

from .agent_tools import (
    AgentTool,
    AgentToolRegistry,
    MultiAgentToolCoordinator,
    get_tool_coordinator,
    initialize_agent_tools
)

from .reinforcement_learning import (
    AgentLightningRL,
    QLearningAgent,
    AgentState,
    RLAction,
    ActionType,
    get_rl_coordinator,
    initialize_rl_training
)

from .reward_system import (
    RewardSystemManager,
    RewardSignal,
    RewardSignalType,
    get_reward_system
)

from .enhanced_coordinator import (
    EnhancedAgentCoordinator,
    create_enhanced_coordinator
)

__version__ = "1.0.0"
__author__ = "NexaCorp AI Team"

# Main initialization function
def initialize_agent_lightning(agents, agentops_api_key=None):
    """
    Initialize the complete Agent Lightning framework.
    
    Args:
        agents: Dictionary of agent_id -> BaseAgent instances
        agentops_api_key: Optional AgentOps API key for cloud monitoring
    
    Returns:
        EnhancedAgentCoordinator with all Agent Lightning capabilities
    """
    # Initialize monitoring
    monitor = initialize_monitoring(agentops_api_key)
    
    # Initialize tool coordination
    tool_coordinator = initialize_agent_tools(agents)
    
    # Initialize RL training
    rl_coordinator = initialize_rl_training(agents)
    
    # Create enhanced coordinator
    enhanced_coordinator = create_enhanced_coordinator(agents, agentops_api_key)
    
    return enhanced_coordinator

__all__ = [
    # Core components
    "AgentOpsMonitor",
    "EmergentCommunicationManager", 
    "MultiAgentToolCoordinator",
    "AgentLightningRL",
    "RewardSystemManager",
    "EnhancedAgentCoordinator",
    
    # Data structures
    "CommunicationProtocol",
    "CommunicationIntent",
    "AgentTool",
    "AgentState",
    "RLAction",
    "ActionType",
    "RewardSignal",
    "RewardSignalType",
    
    # Factory functions
    "get_monitor",
    "get_tool_coordinator", 
    "get_rl_coordinator",
    "get_reward_system",
    "initialize_monitoring",
    "initialize_agent_tools",
    "initialize_rl_training",
    "create_enhanced_coordinator",
    "initialize_agent_lightning"
]
