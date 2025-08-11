"""
Reinforcement Learning Layer for Agent Lightning Framework.
Implements RL-based decision making for agent behavior optimization.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import pickle
from collections import defaultdict, deque

from agents.base_agent import Message, MessageType, BaseAgent
from rl.agent_lightning.agentops_integration import get_monitor

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions agents can take."""
    ROUTE_TO_RETRIEVAL = "route_to_retrieval"
    RESPOND_DIRECTLY = "respond_directly"
    ESCALATE_IMMEDIATELY = "escalate_immediately"
    REQUEST_CLARIFICATION = "request_clarification"
    USE_AGENT_TOOL = "use_agent_tool"
    BROADCAST_QUERY = "broadcast_query"
    NEGOTIATE_RESPONSE = "negotiate_response"
    # New action types for enhanced workflow
    ANALYZE_QUERY = "analyze_query"
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    EVALUATE_RESPONSE = "evaluate_response"
    SYNTHESIZE_RESPONSE = "synthesize_response"
    INITIATE_COMMUNICATION = "initiate_communication"
    PROCESS_FEEDBACK = "process_feedback"

@dataclass
class AgentState:
    """Represents the current state of an agent for RL."""
    agent_id: str
    current_message: Optional[Message] = None
    conversation_history: List[Message] = field(default_factory=list)
    confidence_level: float = 0.5
    workload: int = 0  # Number of active tasks
    recent_performance: float = 0.5  # Recent success rate
    available_tools: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RLAction:
    """Represents an RL action taken by an agent."""
    action_type: ActionType
    target_agent: Optional[str] = None
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5

@dataclass
class RLExperience:
    """Represents an RL experience for training."""
    state: AgentState
    action: RLAction
    reward: float
    next_state: Optional[AgentState] = None
    done: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class RewardCalculator:
    """Calculates rewards for agent actions based on outcomes."""
    
    def __init__(self):
        self.reward_weights = {
            "response_quality": 0.3,
            "response_time": 0.2,
            "user_satisfaction": 0.3,
            "escalation_accuracy": 0.2
        }
    
    def calculate_reward(self, 
                        action: RLAction,
                        outcome: Dict[str, Any],
                        user_feedback: Optional[Dict[str, Any]] = None) -> float:
        """Calculate reward for an agent action."""
        total_reward = 0.0
        
        # Response quality reward
        quality_score = outcome.get("quality_score", 0.5)
        total_reward += self.reward_weights["response_quality"] * quality_score
        
        # Response time reward (faster is better, but not at cost of quality)
        response_time = outcome.get("response_time_ms", 5000)
        time_reward = max(0, 1.0 - (response_time / 10000))  # Normalize to 0-1
        total_reward += self.reward_weights["response_time"] * time_reward
        
        # User satisfaction reward
        if user_feedback:
            satisfaction = user_feedback.get("satisfaction_score", 0.5)
            total_reward += self.reward_weights["user_satisfaction"] * satisfaction
        else:
            # Default neutral satisfaction if no feedback
            total_reward += self.reward_weights["user_satisfaction"] * 0.5
        
        # Escalation accuracy reward
        if action.action_type == ActionType.ESCALATE_IMMEDIATELY:
            escalation_needed = outcome.get("escalation_was_needed", True)
            escalation_reward = 1.0 if escalation_needed else -0.5  # Penalty for false escalation
            total_reward += self.reward_weights["escalation_accuracy"] * escalation_reward
        elif outcome.get("should_have_escalated", False):
            # Penalty for not escalating when needed
            total_reward += self.reward_weights["escalation_accuracy"] * -0.3
        
        # Normalize reward to [-1, 1] range
        return max(-1.0, min(1.0, total_reward))

class QLearningAgent:
    """Q-Learning implementation for agent decision making."""
    
    def __init__(self, 
                 agent_id: str,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995):
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state_hash -> action_type -> q_value
        self.q_table: Dict[str, Dict[ActionType, float]] = defaultdict(lambda: defaultdict(float))

        # Experience replay buffer
        self.experience_buffer: deque = deque(maxlen=10000)

        # Training statistics
        self.episode_count = 0
        self.reward_history: List[float] = []
        self.success_history: List[bool] = []
        self.total_steps = 0
        
        # Performance tracking
        self.total_actions = 0
        self.successful_actions = 0
        self.average_reward = 0.0
        
        self.monitor = get_monitor()
    
    def _state_to_hash(self, state: AgentState) -> str:
        """Convert agent state to a hashable string."""
        # Create a simplified state representation
        message_type = state.current_message.type.value if state.current_message else "none"
        confidence_bucket = int(state.confidence_level * 10)  # 0-10 buckets
        workload_bucket = min(state.workload, 5)  # Cap at 5
        performance_bucket = int(state.recent_performance * 10)  # 0-10 buckets
        
        return f"{message_type}_{confidence_bucket}_{workload_bucket}_{performance_bucket}"
    
    def select_action(self, state: AgentState, available_actions: List[ActionType]) -> RLAction:
        """Select action using epsilon-greedy policy."""
        state_hash = self._state_to_hash(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_type = np.random.choice(available_actions)
            confidence = 0.3  # Lower confidence for exploration
        else:
            # Exploit: best known action
            q_values = {action: self.q_table[state_hash][action] for action in available_actions}
            action_type = max(q_values, key=q_values.get)
            confidence = 0.8  # Higher confidence for exploitation
        
        # Determine target agent and parameters based on action
        target_agent = None
        tool_name = None
        parameters = {}
        
        if action_type == ActionType.ROUTE_TO_RETRIEVAL:
            target_agent = "retrieval_agent"
        elif action_type == ActionType.ESCALATE_IMMEDIATELY:
            target_agent = "escalation_agent"
        elif action_type == ActionType.USE_AGENT_TOOL:
            # Select best tool based on current state
            if state.available_tools:
                tool_name = state.available_tools[0]  # Simple selection for now
        
        return RLAction(
            action_type=action_type,
            target_agent=target_agent,
            tool_name=tool_name,
            parameters=parameters,
            confidence=confidence
        )
    
    def update_q_value(self, experience: RLExperience):
        """Update Q-value based on experience."""
        state_hash = self._state_to_hash(experience.state)
        action_type = experience.action.action_type
        
        current_q = self.q_table[state_hash][action_type]
        
        # Calculate target Q-value
        if experience.next_state and not experience.done:
            next_state_hash = self._state_to_hash(experience.next_state)
            next_max_q = max(self.q_table[next_state_hash].values()) if self.q_table[next_state_hash] else 0.0
            target_q = experience.reward + self.discount_factor * next_max_q
        else:
            target_q = experience.reward
        
        # Update Q-value
        updated_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state_hash][action_type] = updated_q
        
        # Update performance metrics
        self.total_actions += 1
        if experience.reward > 0:
            self.successful_actions += 1
        
        self.average_reward = (self.average_reward * (self.total_actions - 1) + experience.reward) / self.total_actions
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        
        # Record in monitoring
        self.monitor.record_reward_signal(
            agent_id=self.agent_id,
            reward=experience.reward,
            context={
                "action_type": action_type.value,
                "state_hash": state_hash,
                "q_value": updated_q
            }
        )
        
        logger.debug(f"Updated Q-value for {self.agent_id}: {state_hash}[{action_type.value}] = {updated_q:.3f}")
    
    def add_experience(self, experience: RLExperience):
        """Add experience to replay buffer."""
        self.experience_buffer.append(experience)
    
    def replay_training(self, batch_size: int = 32):
        """Perform experience replay training."""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Update Q-values for batch
        for experience in batch:
            self.update_q_value(experience)
        
        logger.debug(f"Completed replay training batch for {self.agent_id}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RL performance metrics."""
        success_rate = self.successful_actions / max(1, self.total_actions)
        
        return {
            "agent_id": self.agent_id,
            "total_actions": self.total_actions,
            "success_rate": success_rate,
            "average_reward": self.average_reward,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "experience_buffer_size": len(self.experience_buffer)
        }
    
    def save_model(self, filepath: str):
        """Save Q-table and training state."""
        model_data = {
            "q_table": dict(self.q_table),
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "average_reward": self.average_reward
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved RL model for {self.agent_id} to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table and training state."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), model_data["q_table"])
            self.learning_rate = model_data.get("learning_rate", self.learning_rate)
            self.discount_factor = model_data.get("discount_factor", self.discount_factor)
            self.epsilon = model_data.get("epsilon", self.epsilon)
            self.total_actions = model_data.get("total_actions", 0)
            self.successful_actions = model_data.get("successful_actions", 0)
            self.average_reward = model_data.get("average_reward", 0.0)
            
            logger.info(f"Loaded RL model for {self.agent_id} from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load RL model for {self.agent_id}: {e}")

class AgentLightningRL:
    """Main RL coordinator implementing Agent Lightning concepts."""
    
    def __init__(self):
        self.rl_agents: Dict[str, QLearningAgent] = {}
        self.reward_calculator = RewardCalculator()
        self.monitor = get_monitor()
        
        # Training state
        self.training_active = False
        self.training_episodes = 0
        self.global_performance_history: List[Dict[str, Any]] = []
    
    def register_agent(self, agent_id: str, **rl_params) -> QLearningAgent:
        """Register an agent for RL training."""
        rl_agent = QLearningAgent(agent_id, **rl_params)
        self.rl_agents[agent_id] = rl_agent
        
        logger.info(f"Registered RL agent: {agent_id}")
        return rl_agent

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.rl_agents:
            return {
                "total_episodes": 0,
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "learning_rate": 0.001,
                "training_active": self.training_active
            }

        # Calculate aggregate stats
        total_episodes = sum(agent.episode_count for agent in self.rl_agents.values())
        total_rewards = []
        success_counts = []

        for agent in self.rl_agents.values():
            if agent.reward_history:
                total_rewards.extend(agent.reward_history)
            if hasattr(agent, 'success_history'):
                success_counts.extend(agent.success_history)

        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
        success_rate = sum(success_counts) / len(success_counts) if success_counts else 0.0

        return {
            "total_episodes": total_episodes,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "learning_rate": list(self.rl_agents.values())[0].learning_rate if self.rl_agents else 0.001,
            "training_active": self.training_active,
            "num_agents": len(self.rl_agents),
            "total_rewards": len(total_rewards)
        }
    
    def get_available_actions(self, agent_id: str, state: AgentState) -> List[ActionType]:
        """Get available actions for an agent in a given state."""
        base_actions = [ActionType.RESPOND_DIRECTLY, ActionType.REQUEST_CLARIFICATION]
        
        if agent_id == "communication_agent":
            return base_actions + [
                ActionType.ROUTE_TO_RETRIEVAL,
                ActionType.USE_AGENT_TOOL
            ]
        elif agent_id == "retrieval_agent":
            return base_actions + [
                ActionType.USE_AGENT_TOOL,
                ActionType.ESCALATE_IMMEDIATELY
            ]
        elif agent_id == "critic_agent":
            return base_actions + [
                ActionType.USE_AGENT_TOOL,
                ActionType.NEGOTIATE_RESPONSE
            ]
        elif agent_id == "escalation_agent":
            return base_actions + [
                ActionType.ESCALATE_IMMEDIATELY,
                ActionType.USE_AGENT_TOOL
            ]
        
        return base_actions
    
    async def make_rl_decision(self, 
                              agent_id: str,
                              current_state: AgentState) -> RLAction:
        """Make an RL-based decision for an agent."""
        if agent_id not in self.rl_agents:
            # Fallback to default action
            return RLAction(action_type=ActionType.RESPOND_DIRECTLY)
        
        rl_agent = self.rl_agents[agent_id]
        available_actions = self.get_available_actions(agent_id, current_state)
        
        # Select action using RL policy
        action = rl_agent.select_action(current_state, available_actions)
        
        # Record the decision
        self.monitor.record_agent_interaction(
            agent_id=agent_id,
            action_type="rl_decision",
            input_message=current_state.current_message,
            decision_context={
                "available_actions": [a.value for a in available_actions],
                "selected_action": action.action_type.value,
                "confidence": action.confidence,
                "epsilon": rl_agent.epsilon
            }
        )
        
        return action
    
    def provide_reward_feedback(self,
                               agent_id: str,
                               action: RLAction,
                               outcome: Dict[str, Any],
                               user_feedback: Optional[Dict[str, Any]] = None):
        """Provide reward feedback for an agent action."""
        if agent_id not in self.rl_agents:
            return

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(action, outcome, user_feedback)

        # Record reward
        self.monitor.record_reward_signal(
            agent_id=agent_id,
            reward=reward,
            context={
                "action_type": action.action_type.value,
                "outcome": outcome,
                "user_feedback": user_feedback
            }
        )

        logger.debug(f"Provided reward feedback for {agent_id}: {reward:.3f}")
        return reward
    
    def create_experience(self,
                         agent_id: str,
                         state: AgentState,
                         action: RLAction,
                         reward: float,
                         next_state: Optional[AgentState] = None,
                         done: bool = False) -> RLExperience:
        """Create an RL experience for training."""
        experience = RLExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        if agent_id in self.rl_agents:
            self.rl_agents[agent_id].add_experience(experience)
            self.rl_agents[agent_id].update_q_value(experience)
        
        return experience
    
    def start_training_episode(self):
        """Start a new training episode."""
        self.training_active = True
        self.training_episodes += 1
        
        # Start monitoring session
        self.monitor.start_session(f"training_episode_{self.training_episodes}")
        
        logger.info(f"Started training episode {self.training_episodes}")
    
    def end_training_episode(self, episode_outcome: str = "completed"):
        """End the current training episode."""
        if not self.training_active:
            return
        
        # Perform replay training for all agents
        for rl_agent in self.rl_agents.values():
            rl_agent.replay_training()
        
        # Record episode performance
        episode_performance = {
            "episode": self.training_episodes,
            "outcome": episode_outcome,
            "timestamp": datetime.now().isoformat(),
            "agent_performance": {
                agent_id: rl_agent.get_performance_metrics()
                for agent_id, rl_agent in self.rl_agents.items()
            }
        }
        
        self.global_performance_history.append(episode_performance)
        
        # End monitoring session
        self.monitor.end_session(episode_outcome)
        
        self.training_active = False
        logger.info(f"Ended training episode {self.training_episodes} with outcome: {episode_outcome}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            "total_episodes": self.training_episodes,
            "training_active": self.training_active,
            "agent_statistics": {
                agent_id: rl_agent.get_performance_metrics()
                for agent_id, rl_agent in self.rl_agents.items()
            },
            "recent_performance": self.global_performance_history[-10:] if self.global_performance_history else [],
            "communication_patterns": self.monitor.get_communication_patterns()
        }
    
    def save_all_models(self, directory: str):
        """Save all RL models."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for agent_id, rl_agent in self.rl_agents.items():
            filepath = os.path.join(directory, f"{agent_id}_rl_model.pkl")
            rl_agent.save_model(filepath)
        
        # Save global training state
        global_state = {
            "training_episodes": self.training_episodes,
            "performance_history": self.global_performance_history
        }
        
        with open(os.path.join(directory, "global_training_state.json"), 'w') as f:
            json.dump(global_state, f, indent=2)
        
        logger.info(f"Saved all RL models to {directory}")
    
    def load_all_models(self, directory: str):
        """Load all RL models."""
        import os
        
        for agent_id in self.rl_agents:
            filepath = os.path.join(directory, f"{agent_id}_rl_model.pkl")
            if os.path.exists(filepath):
                self.rl_agents[agent_id].load_model(filepath)
        
        # Load global training state
        global_state_path = os.path.join(directory, "global_training_state.json")
        if os.path.exists(global_state_path):
            try:
                with open(global_state_path, 'r') as f:
                    global_state = json.load(f)
                
                self.training_episodes = global_state.get("training_episodes", 0)
                self.global_performance_history = global_state.get("performance_history", [])
                
                logger.info(f"Loaded global training state from {directory}")
            except Exception as e:
                logger.error(f"Failed to load global training state: {e}")

# Global RL coordinator instance
_rl_coordinator: Optional[AgentLightningRL] = None

def get_rl_coordinator() -> AgentLightningRL:
    """Get the global RL coordinator instance."""
    global _rl_coordinator
    if _rl_coordinator is None:
        _rl_coordinator = AgentLightningRL()
    return _rl_coordinator

def initialize_rl_training(agents: Dict[str, BaseAgent]) -> AgentLightningRL:
    """Initialize RL training for all agents."""
    global _rl_coordinator
    _rl_coordinator = AgentLightningRL()
    
    # Register all agents for RL training
    for agent_id in agents:
        _rl_coordinator.register_agent(agent_id)
    
    logger.info(f"Initialized RL training for {len(agents)} agents")
    return _rl_coordinator
