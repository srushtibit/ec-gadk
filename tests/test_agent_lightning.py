"""
Tests for Agent Lightning Framework integration.
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import system components
from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent
from agents.escalation.escalation_agent import EscalationAgent
from agents.base_agent import Message, MessageType

# Import Agent Lightning components
from rl.agent_lightning.agentops_integration import AgentOpsMonitor, get_monitor
from rl.agent_lightning.emergent_communication import EmergentCommunicationManager
from rl.agent_lightning.agent_tools import AgentToolRegistry, MultiAgentToolCoordinator
from rl.agent_lightning.reinforcement_learning import AgentLightningRL, AgentState, ActionType
from rl.agent_lightning.reward_system import RewardSystemManager
from rl.agent_lightning.enhanced_coordinator import EnhancedAgentCoordinator, create_enhanced_coordinator

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    return {
        "communication_agent": CommunicationAgent(),
        "retrieval_agent": RetrievalAgent(),
        "critic_agent": CriticAgent(),
        "escalation_agent": EscalationAgent()
    }

@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        type=MessageType.QUERY,
        content="I can't access my email account",
        sender="user",
        recipient="communication_agent"
    )

class TestAgentOpsIntegration:
    """Test AgentOps monitoring integration."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = AgentOpsMonitor()
        assert monitor is not None
        assert not monitor.session_active
        assert len(monitor.agent_interactions) == 0
    
    def test_interaction_recording(self, sample_message):
        """Test recording agent interactions."""
        monitor = AgentOpsMonitor()
        
        monitor.record_agent_interaction(
            agent_id="test_agent",
            action_type="test_action",
            input_message=sample_message,
            reward_signal=0.8
        )
        
        assert len(monitor.agent_interactions) == 1
        interaction = monitor.agent_interactions[0]
        assert interaction.agent_id == "test_agent"
        assert interaction.action_type == "test_action"
        assert interaction.reward_signal == 0.8
    
    def test_communication_event_recording(self):
        """Test recording communication events."""
        monitor = AgentOpsMonitor()
        
        monitor.record_communication_event(
            sender_agent="agent1",
            receiver_agent="agent2",
            message_content="test message",
            communication_type="direct",
            success=True,
            latency_ms=150.0
        )
        
        assert len(monitor.communication_events) == 1
        event = monitor.communication_events[0]
        assert event.sender_agent == "agent1"
        assert event.receiver_agent == "agent2"
        assert event.success is True
        assert event.latency_ms == 150.0

class TestEmergentCommunication:
    """Test emergent communication framework."""
    
    def test_communication_manager_initialization(self):
        """Test communication manager initialization."""
        manager = EmergentCommunicationManager()
        assert manager is not None
        assert len(manager.communication_patterns) == 0
    
    @pytest.mark.asyncio
    async def test_communication_analysis(self, sample_message):
        """Test communication intent analysis."""
        manager = EmergentCommunicationManager()
        
        analysis = await manager.analyze_communication_intent(
            sender_agent="communication_agent",
            message=sample_message,
            available_agents=["retrieval_agent", "critic_agent"]
        )
        
        assert "recommended_protocol" in analysis
        assert "routing_suggestion" in analysis
        assert "urgency_assessment" in analysis
        assert analysis["routing_suggestion"] in ["retrieval_agent", "critic_agent"]
    
    @pytest.mark.asyncio
    async def test_communication_facilitation(self, sample_message):
        """Test communication facilitation."""
        manager = EmergentCommunicationManager()
        
        routed_messages = await manager.facilitate_communication(
            sender_agent="communication_agent",
            message=sample_message,
            available_agents=["retrieval_agent"]
        )
        
        assert len(routed_messages) > 0
        assert routed_messages[0].recipient == "retrieval_agent"

class TestAgentTools:
    """Test agent-as-tools framework."""
    
    def test_tool_registry_initialization(self):
        """Test tool registry initialization."""
        registry = AgentToolRegistry()
        assert registry is not None
        assert len(registry.tools) == 0
    
    def test_agent_tool_registration(self, mock_agents):
        """Test registering agents as tools."""
        registry = AgentToolRegistry()
        
        agent = mock_agents["communication_agent"]
        tool = registry.register_agent_as_tool(
            agent=agent,
            tool_name="comm_tool",
            description="Communication agent tool"
        )
        
        assert tool is not None
        assert "comm_tool" in registry.tools
        assert registry.get_tool("comm_tool") == tool
    
    def test_tool_coordinator_initialization(self, mock_agents):
        """Test tool coordinator initialization."""
        coordinator = MultiAgentToolCoordinator()
        coordinator.register_all_agents_as_tools(mock_agents)
        
        available_tools = coordinator.registry.get_available_tools()
        assert len(available_tools) == len(mock_agents)

class TestReinforcementLearning:
    """Test reinforcement learning components."""
    
    def test_rl_coordinator_initialization(self):
        """Test RL coordinator initialization."""
        rl_coordinator = AgentLightningRL()
        assert rl_coordinator is not None
        assert len(rl_coordinator.rl_agents) == 0
    
    def test_agent_registration(self):
        """Test registering agents for RL training."""
        rl_coordinator = AgentLightningRL()
        
        rl_agent = rl_coordinator.register_agent("test_agent")
        assert rl_agent is not None
        assert "test_agent" in rl_coordinator.rl_agents
    
    @pytest.mark.asyncio
    async def test_rl_decision_making(self, sample_message):
        """Test RL-based decision making."""
        rl_coordinator = AgentLightningRL()
        rl_coordinator.register_agent("communication_agent")
        
        state = AgentState(
            agent_id="communication_agent",
            current_message=sample_message,
            confidence_level=0.7
        )
        
        action = await rl_coordinator.make_rl_decision("communication_agent", state)
        assert action is not None
        assert action.action_type in ActionType
        assert 0.0 <= action.confidence <= 1.0

class TestRewardSystem:
    """Test reward system components."""
    
    def test_reward_system_initialization(self):
        """Test reward system initialization."""
        reward_system = RewardSystemManager()
        assert reward_system is not None
        assert reward_system.feedback_collector is not None
        assert reward_system.quality_assessor is not None
    
    @pytest.mark.asyncio
    async def test_reward_calculation(self):
        """Test comprehensive reward calculation."""
        reward_system = RewardSystemManager()
        
        from rl.agent_lightning.reinforcement_learning import RLAction
        
        action = RLAction(action_type=ActionType.ROUTE_TO_RETRIEVAL)
        outcome = {
            "query": "test query",
            "response": "test response",
            "response_time_ms": 2000,
            "success": True
        }
        user_feedback = {
            "user_id": "test_user",
            "satisfaction_score": 0.8
        }
        
        reward = await reward_system.process_outcome_and_calculate_reward(
            agent_id="test_agent",
            action=action,
            outcome=outcome,
            user_feedback=user_feedback
        )
        
        assert -1.0 <= reward <= 1.0

class TestEnhancedCoordinator:
    """Test enhanced coordinator integration."""
    
    def test_enhanced_coordinator_creation(self, mock_agents):
        """Test creating enhanced coordinator."""
        coordinator = create_enhanced_coordinator(mock_agents)
        assert coordinator is not None
        assert coordinator.monitor is not None
        assert coordinator.communication_manager is not None
        assert coordinator.tool_coordinator is not None
        assert coordinator.rl_coordinator is not None
    
    @pytest.mark.asyncio
    async def test_enhanced_query_processing(self, mock_agents):
        """Test enhanced query processing."""
        coordinator = create_enhanced_coordinator(mock_agents)
        
        # Mock the agent responses to avoid actual ADK calls
        with patch.object(coordinator.agents["communication_agent"], "process_message") as mock_comm:
            mock_response = Message(
                type=MessageType.RESPONSE,
                content="Hello! How can I help you?",
                sender="communication_agent",
                recipient="user"
            )
            mock_comm.return_value = mock_response
            
            result = await coordinator.process_query_enhanced("Hello", "test_user")
            
            assert result is not None
            assert "response" in result
            assert "total_processing_time_ms" in result
            assert result.get("agent_lightning_enabled", False) is True
    
    def test_training_mode_toggle(self, mock_agents):
        """Test training mode functionality."""
        coordinator = create_enhanced_coordinator(mock_agents)
        
        # Test enabling training mode
        coordinator.enable_training_mode()
        assert coordinator.training_mode is True
        
        # Test disabling training mode
        coordinator.disable_training_mode()
        assert coordinator.training_mode is False
    
    def test_statistics_collection(self, mock_agents):
        """Test statistics collection."""
        coordinator = create_enhanced_coordinator(mock_agents)
        
        stats = coordinator.get_agent_lightning_statistics()
        assert "rl_statistics" in stats
        assert "communication_patterns" in stats
        assert "tool_coordination" in stats
        assert "agent_states" in stats
        assert "training_mode" in stats

# Integration test
@pytest.mark.asyncio
async def test_full_agent_lightning_integration(mock_agents):
    """Test full Agent Lightning integration."""
    # Create enhanced coordinator
    coordinator = create_enhanced_coordinator(mock_agents)
    
    # Enable training mode
    coordinator.enable_training_mode()
    
    # Process a query (with mocked responses)
    with patch.object(coordinator.agents["communication_agent"], "process_message") as mock_comm:
        mock_response = Message(
            type=MessageType.RESPONSE,
            content="Test response",
            sender="communication_agent",
            recipient="user"
        )
        mock_comm.return_value = mock_response
        
        result = await coordinator.process_query_enhanced("Test query", "test_user")
        
        # Verify Agent Lightning features are working
        assert result["agent_lightning_enabled"] is True
        assert "workflow_steps" in result
        assert result["training_mode"] is True
    
    # Check that statistics are being collected
    stats = coordinator.get_agent_lightning_statistics()
    assert stats["rl_statistics"]["total_episodes"] > 0
    
    # Disable training mode
    coordinator.disable_training_mode()
    assert coordinator.training_mode is False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
