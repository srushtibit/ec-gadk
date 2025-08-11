"""
AgentOps Integration for Agent Lightning Framework.
Provides observability and monitoring for multi-agent RL training.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    import agentops
    from agentops import trace, record_action
    AGENTOPS_AVAILABLE = True
except ImportError:
    AGENTOPS_AVAILABLE = False
    # Mock decorators for when AgentOps is not available
    def trace(name: str = None, tags: List[str] = None):
        def decorator(func):
            return func
        return decorator
    
    def record_action(action_type: str, **kwargs):
        pass

from agents.base_agent import Message, MessageType, AgentAction

logger = logging.getLogger(__name__)

@dataclass
class AgentInteraction:
    """Represents an interaction between agents for RL training."""
    agent_id: str
    action_type: str
    input_message: Optional[Message] = None
    output_message: Optional[Message] = None
    decision_context: Dict[str, Any] = field(default_factory=dict)
    reward_signal: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommunicationEvent:
    """Represents emergent communication between agents."""
    sender_agent: str
    receiver_agent: str
    message_content: str
    communication_type: str  # 'direct', 'broadcast', 'tool_call'
    success: bool = True
    latency_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class AgentOpsMonitor:
    """Monitors agent interactions and provides data for RL training."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("AGENTOPS_API_KEY")
        self.is_initialized = False
        self.session_active = False
        
        # Interaction tracking
        self.agent_interactions: List[AgentInteraction] = []
        self.communication_events: List[CommunicationEvent] = []
        self.reward_signals: Dict[str, List[float]] = {}
        
        # Performance metrics
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
        if AGENTOPS_AVAILABLE and self.api_key:
            self._initialize_agentops()
    
    def _initialize_agentops(self):
        """Initialize AgentOps monitoring."""
        try:
            agentops.init(
                api_key=self.api_key,
                auto_start_session=False,
                trace_name="nexacorp-agent-lightning"
            )
            self.is_initialized = True
            logger.info("AgentOps initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AgentOps: {e}")
            self.is_initialized = False
    
    @trace(name="agent-session", tags=["multi-agent", "rl-training"])
    def start_session(self, session_name: str = "agent-lightning-session"):
        """Start an AgentOps session for monitoring."""
        if not self.is_initialized:
            logger.warning("AgentOps not initialized, using local monitoring only")
            self.session_active = True
            return
        
        try:
            agentops.start_session(tags=["agent-lightning", "emergent-communication"])
            self.session_active = True
            logger.info(f"Started AgentOps session: {session_name}")
        except Exception as e:
            logger.error(f"Failed to start AgentOps session: {e}")
    
    def end_session(self, outcome: str = "Success"):
        """End the current AgentOps session."""
        if not self.session_active:
            return
        
        try:
            if self.is_initialized:
                agentops.end_session(outcome)
            self.session_active = False
            logger.info(f"Ended AgentOps session with outcome: {outcome}")
        except Exception as e:
            logger.error(f"Failed to end AgentOps session: {e}")
    
    @trace(name="agent-interaction", tags=["agent-decision"])
    def record_agent_interaction(self, 
                                agent_id: str,
                                action_type: str,
                                input_message: Optional[Message] = None,
                                output_message: Optional[Message] = None,
                                decision_context: Dict[str, Any] = None,
                                reward_signal: Optional[float] = None):
        """Record an agent interaction for RL training."""
        interaction = AgentInteraction(
            agent_id=agent_id,
            action_type=action_type,
            input_message=input_message,
            output_message=output_message,
            decision_context=decision_context or {},
            reward_signal=reward_signal
        )
        
        self.agent_interactions.append(interaction)
        
        # Record in AgentOps if available
        if self.is_initialized:
            record_action(
                action_type=f"agent_{action_type}",
                agent_id=agent_id,
                input_content=input_message.content if input_message else None,
                output_content=output_message.content if output_message else None,
                decision_context=decision_context,
                reward=reward_signal
            )
        
        logger.debug(f"Recorded interaction: {agent_id} -> {action_type}")
    
    @trace(name="agent-communication", tags=["emergent-communication"])
    def record_communication_event(self,
                                  sender_agent: str,
                                  receiver_agent: str,
                                  message_content: str,
                                  communication_type: str,
                                  success: bool = True,
                                  latency_ms: Optional[float] = None):
        """Record emergent communication between agents."""
        event = CommunicationEvent(
            sender_agent=sender_agent,
            receiver_agent=receiver_agent,
            message_content=message_content,
            communication_type=communication_type,
            success=success,
            latency_ms=latency_ms
        )
        
        self.communication_events.append(event)
        
        # Record in AgentOps if available
        if self.is_initialized:
            record_action(
                action_type="agent_communication",
                sender=sender_agent,
                receiver=receiver_agent,
                communication_type=communication_type,
                success=success,
                latency_ms=latency_ms
            )
        
        logger.debug(f"Recorded communication: {sender_agent} -> {receiver_agent}")
    
    def record_reward_signal(self, agent_id: str, reward: float, context: Dict[str, Any] = None):
        """Record a reward signal for RL training."""
        if agent_id not in self.reward_signals:
            self.reward_signals[agent_id] = []
        
        self.reward_signals[agent_id].append(reward)
        
        # Record in AgentOps if available
        if self.is_initialized:
            record_action(
                action_type="reward_signal",
                agent_id=agent_id,
                reward=reward,
                context=context or {}
            )
        
        logger.debug(f"Recorded reward for {agent_id}: {reward}")
    
    def get_agent_performance_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent."""
        interactions = [i for i in self.agent_interactions if i.agent_id == agent_id]
        rewards = self.reward_signals.get(agent_id, [])
        
        if not interactions:
            return {"total_interactions": 0, "average_reward": 0.0}
        
        return {
            "total_interactions": len(interactions),
            "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "recent_interactions": len([i for i in interactions[-10:]]),
            "action_types": list(set(i.action_type for i in interactions))
        }
    
    def get_communication_patterns(self) -> Dict[str, Any]:
        """Analyze emergent communication patterns."""
        if not self.communication_events:
            return {"total_events": 0}
        
        # Analyze communication patterns
        sender_counts = {}
        receiver_counts = {}
        type_counts = {}
        
        for event in self.communication_events:
            sender_counts[event.sender_agent] = sender_counts.get(event.sender_agent, 0) + 1
            receiver_counts[event.receiver_agent] = receiver_counts.get(event.receiver_agent, 0) + 1
            type_counts[event.communication_type] = type_counts.get(event.communication_type, 0) + 1
        
        return {
            "total_events": len(self.communication_events),
            "sender_distribution": sender_counts,
            "receiver_distribution": receiver_counts,
            "communication_types": type_counts,
            "success_rate": sum(1 for e in self.communication_events if e.success) / len(self.communication_events),
            "average_latency": sum(e.latency_ms for e in self.communication_events if e.latency_ms) / 
                             len([e for e in self.communication_events if e.latency_ms]) if any(e.latency_ms for e in self.communication_events) else 0.0
        }
    
    def export_training_data(self, filepath: str):
        """Export collected data for RL training."""
        training_data = {
            "interactions": [
                {
                    "agent_id": i.agent_id,
                    "action_type": i.action_type,
                    "input_content": i.input_message.content if i.input_message else None,
                    "output_content": i.output_message.content if i.output_message else None,
                    "decision_context": i.decision_context,
                    "reward_signal": i.reward_signal,
                    "timestamp": i.timestamp
                }
                for i in self.agent_interactions
            ],
            "communications": [
                {
                    "sender": e.sender_agent,
                    "receiver": e.receiver_agent,
                    "content": e.message_content,
                    "type": e.communication_type,
                    "success": e.success,
                    "latency_ms": e.latency_ms,
                    "timestamp": e.timestamp
                }
                for e in self.communication_events
            ],
            "rewards": self.reward_signals,
            "performance_metrics": {
                agent_id: self.get_agent_performance_metrics(agent_id)
                for agent_id in set(i.agent_id for i in self.agent_interactions)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Exported training data to {filepath}")

# Global monitor instance
_monitor_instance: Optional[AgentOpsMonitor] = None

def get_monitor() -> AgentOpsMonitor:
    """Get the global AgentOps monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = AgentOpsMonitor()
    return _monitor_instance

def initialize_monitoring(api_key: Optional[str] = None) -> AgentOpsMonitor:
    """Initialize global monitoring with optional API key."""
    global _monitor_instance
    _monitor_instance = AgentOpsMonitor(api_key)
    return _monitor_instance
