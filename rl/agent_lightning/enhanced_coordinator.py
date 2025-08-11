"""
Enhanced Agent Coordinator with Agent Lightning Integration.
Combines emergent communication, RL decision making, and agent tools.
"""

import asyncio
import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from agents.base_agent import Message, MessageType, BaseAgent, AgentCoordinator
from rl.agent_lightning.agentops_integration import get_monitor, initialize_monitoring
from rl.agent_lightning.emergent_communication import EmergentCommunicationManager
from rl.agent_lightning.agent_tools import get_tool_coordinator, initialize_agent_tools
from rl.agent_lightning.reinforcement_learning import (
    get_rl_coordinator, initialize_rl_training, AgentState, ActionType, RLAction
)

logger = logging.getLogger(__name__)

class EnhancedAgentCoordinator(AgentCoordinator):
    """Enhanced coordinator with Agent Lightning capabilities."""
    
    def __init__(self, agents: Dict[str, BaseAgent], agentops_api_key: Optional[str] = None):
        # Initialize base coordinator
        super().__init__()

        # Store agents dictionary for direct access
        self.agents = agents

        # Register agents with coordinator
        for agent in agents.values():
            self.register_agent(agent)

        # Initialize Agent Lightning components
        self.monitor = initialize_monitoring(agentops_api_key)
        self.communication_manager = EmergentCommunicationManager()
        self.tool_coordinator = initialize_agent_tools(agents)
        self.rl_coordinator = initialize_rl_training(agents)

        # Set AgentOps API key in environment for all agents
        if agentops_api_key:
            os.environ["AGENTOPS_API_KEY"] = agentops_api_key
        
        # Enhanced state tracking
        self.agent_states: Dict[str, AgentState] = {}
        self.conversation_context: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize agent states
        self._initialize_agent_states()
        
        # Training configuration
        self.training_mode = False
        self.auto_training = True  # Automatically learn from interactions
        
        logger.info("Enhanced Agent Coordinator initialized with Agent Lightning capabilities")
    
    def _initialize_agent_states(self):
        """Initialize RL states for all agents."""
        for agent_id, agent in self.agents.items():
            self.agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                confidence_level=0.5,
                workload=0,
                recent_performance=0.5,
                available_tools=self.tool_coordinator.registry.get_available_tools(),
                context={}
            )
    
    async def process_query_enhanced(self, query: str, user_id: str = "user") -> Dict[str, Any]:
        """Enhanced query processing with Agent Lightning capabilities."""
        start_time = time.time()
        
        # Start training episode if in training mode
        if self.training_mode:
            self.rl_coordinator.start_training_episode()
        
        # Start monitoring session
        self.monitor.start_session(f"query_session_{int(start_time)}")
        
        try:
            # Create initial message
            initial_message = Message(
                type=MessageType.QUERY,
                content=query,
                metadata={"user_id": user_id, "session_start": start_time},
                sender=user_id,
                recipient="communication_agent"
            )
            
            # Update conversation context
            self.conversation_context[user_id] = {
                "current_query": query,
                "start_time": start_time,
                "message_history": [initial_message]
            }
            
            # Process through enhanced workflow
            result = await self._enhanced_workflow(initial_message, user_id)

            # Ensure result is not None
            if result is None:
                result = {
                    "success": False,
                    "response": "No response generated from enhanced workflow",
                    "agent_path": [],
                    "workflow_steps": []
                }

            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000

            # Provide RL feedback if auto-training is enabled
            if self.auto_training and result:
                await self._provide_automatic_feedback(result, total_time)

            # End monitoring session
            self.monitor.end_session("Success" if result and result.get("success", False) else "Failed")
            
            # End training episode if in training mode
            if self.training_mode:
                self.rl_coordinator.end_training_episode("completed")
            
            return {
                **result,
                "total_processing_time_ms": total_time,
                "agent_lightning_enabled": True,
                "training_mode": self.training_mode
            }
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # End sessions on error
            self.monitor.end_session("Error")
            if self.training_mode:
                self.rl_coordinator.end_training_episode("error")

            return {
                "success": False,
                "response": f"System error: {str(e)}",
                "error": str(e),
                "total_processing_time_ms": (time.time() - start_time) * 1000
            }

    def _extract_communication_intent(self, message: Message) -> str:
        """Extract communication intent from agent message for emergent communication tracking."""
        if not message:
            return "unknown"

        content = message.content.lower() if message.content else ""

        # Analyze content for communication patterns
        if "technical" in content or "query" in content:
            return "technical_assistance"
        elif "greeting" in content or "hello" in content or "hi" in content:
            return "social_greeting"
        elif "escalate" in content or "urgent" in content:
            return "escalation_request"
        elif "search" in content or "find" in content:
            return "information_retrieval"
        elif "evaluate" in content or "assess" in content:
            return "quality_evaluation"
        else:
            return "general_communication"

    def _get_emergent_protocol(self, agent_id: str, action: str) -> Dict[str, Any]:
        """Get emergent communication protocol information for agent interaction."""
        return {
            "protocol_version": "1.0",
            "communication_pattern": f"{agent_id}_{action}",
            "negotiation_round": 1,
            "consensus_level": 0.8,
            "adaptation_score": 0.6,
            "protocol_efficiency": 0.75
        }
    
    async def _enhanced_workflow(self, initial_message: Message, user_id: str) -> Dict[str, Any]:
        """Enhanced workflow with fallback to standard multi-agent pipeline."""
        workflow_steps = []

        # For now, always use standard pipeline since RL decision making needs training
        logger.info("Using standard multi-agent pipeline (RL system needs more training)")
        return await self._standard_workflow_fallback(initial_message, user_id)

    async def _try_rl_workflow(self, initial_message: Message, user_id: str, workflow_steps: List) -> Dict[str, Any]:
        """Try the RL-enhanced workflow."""
        current_message = initial_message

        # Step 1: Communication Agent with RL Decision Making
        try:
            comm_state = self._get_current_state("communication_agent", current_message)
            comm_action = await self.rl_coordinator.make_rl_decision("communication_agent", comm_state)

            workflow_steps.append({
                "agent": "communication_agent",
                "action": comm_action.action_type.value,
                "confidence": comm_action.confidence
            })

            # Execute communication agent action
            if comm_action.action_type == ActionType.RESPOND_DIRECTLY:
                # Handle directly (greetings, casual queries)
                response = await self.agents["communication_agent"].process_message(current_message)
                return {
                    "success": True,
                    "response": response.content,
                    "agent_path": ["communication_agent"],
                    "workflow_steps": workflow_steps,
                    "direct_response": True
                }
            else:
                # For all other action types, fall back to standard pipeline
                logger.info(f"RL action {comm_action.action_type} not fully implemented, falling back to standard pipeline")
                raise Exception("Falling back to standard pipeline for complex routing")

        except Exception as e:
            logger.error(f"RL communication step failed: {e}")
            raise

    async def _standard_workflow_fallback(self, initial_message: Message, user_id: str) -> Dict[str, Any]:
        """Fallback to standard multi-agent processing pipeline."""
        logger.info("Using standard multi-agent pipeline fallback")

        try:
            # Use the standard coordinator's run_cycle method
            workflow_steps = []
            final_response = ""
            retrieved_docs = None

            # Process through communication agent first
            comm_response = await self.agents["communication_agent"].process_message(initial_message)

            workflow_steps.append({
                "agent": "communication_agent",
                "action": ActionType.ANALYZE_QUERY.value,
                "response": comm_response.content if comm_response else "No response",
                "timestamp": time.time(),
                "sender": "communication_agent",
                "recipient": comm_response.recipient if comm_response else "unknown",
                "type": "analysis",
                "content": comm_response.content if comm_response else "No response",
                "confidence": 0.8,
                "communication_intent": self._extract_communication_intent(comm_response),
                "emergent_protocol": self._get_emergent_protocol("communication_agent", "analyze_query")
            })

            # Check if it's a direct response (greeting, etc.)
            if comm_response and comm_response.recipient == "user":
                return {
                    "success": True,
                    "response": comm_response.content,
                    "agent_path": ["communication_agent"],
                    "workflow_steps": workflow_steps,
                    "direct_response": True
                }

            # Continue with retrieval if communication agent routed to retrieval
            if comm_response and comm_response.recipient == "retrieval_agent":
                # Route to retrieval agent
                retrieval_message = Message(
                    type=MessageType.QUERY,
                    content=initial_message.content,
                    sender="communication_agent",
                    recipient="retrieval_agent",
                    metadata={"analyzed_query": comm_response.content}
                )

                retrieval_response = await self.agents["retrieval_agent"].process_message(retrieval_message)

                workflow_steps.append({
                    "agent": "retrieval_agent",
                    "action": ActionType.SEARCH_KNOWLEDGE_BASE.value,
                    "response": retrieval_response.content if retrieval_response else "No documents found",
                    "timestamp": time.time(),
                    "sender": "retrieval_agent",
                    "recipient": "communication_agent",
                    "type": "response",
                    "content": retrieval_response.content if retrieval_response else "No documents found",
                    "confidence": 0.7,
                    "communication_intent": self._extract_communication_intent(retrieval_response),
                    "emergent_protocol": self._get_emergent_protocol("retrieval_agent", "search_knowledge_base"),
                    "retrieved_docs": getattr(retrieval_response, 'metadata', {}).get('retrieved_docs', [])
                })

                if retrieval_response and retrieval_response.metadata:
                    retrieved_docs = retrieval_response.metadata.get("retrieved_docs")

                # Route to critic agent for final response
                if retrieval_response:
                    critic_message = Message(
                        type=MessageType.QUERY,
                        content=initial_message.content,
                        sender="retrieval_agent",
                        recipient="critic_agent",
                        metadata={
                            "retrieved_docs": retrieved_docs,
                            "search_results": retrieval_response.content
                        }
                    )

                    critic_response = await self.agents["critic_agent"].process_message(critic_message)

                    workflow_steps.append({
                        "agent": "critic_agent",
                        "action": "generate_response",
                        "response": critic_response.content if critic_response else "No response generated",
                        "timestamp": time.time(),
                        "sender": "critic_agent",
                        "recipient": "user",
                        "type": "response",
                        "content": critic_response.content if critic_response else "No response generated"
                    })

                    if critic_response:
                        final_response = critic_response.content

            # Ensure we have a response - use retrieval response as fallback
            if not final_response and retrieval_response:
                final_response = retrieval_response.content

            # Return the final result
            return {
                "success": True,
                "response": final_response or "I apologize, but I couldn't generate a proper response to your query.",
                "agent_path": [step["agent"] for step in workflow_steps],
                "workflow_steps": workflow_steps,
                "retrieved_docs": retrieved_docs,
                "direct_response": False
            }

        except Exception as e:
            logger.error(f"Standard workflow fallback failed: {e}")
            return {
                "success": False,
                "response": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "agent_path": [],
                "workflow_steps": [],
                "retrieved_docs": None,
                "direct_response": False,
                "error": str(e)
            }
    
    def _get_current_state(self, agent_id: str, message: Message) -> AgentState:
        """Get current state for an agent."""
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentState(agent_id=agent_id)
        
        state = self.agent_states[agent_id]
        state.current_message = message
        
        # Update conversation history
        if message not in state.conversation_history:
            state.conversation_history.append(message)
            
        # Keep only recent history
        if len(state.conversation_history) > 10:
            state.conversation_history = state.conversation_history[-10:]
        
        return state
    
    async def _provide_automatic_feedback(self, result: Dict[str, Any], processing_time: float):
        """Provide automatic feedback for RL training."""
        try:
            logger.debug(f"Starting automatic feedback for result: {result.keys()}")

            # Calculate outcome metrics
            outcome = {
                "response_time_ms": processing_time,
                "quality_score": result.get("quality_score", 0.5),
                "escalation_was_needed": result.get("escalated", False),
                "success": result.get("success", False)
            }

            # Provide feedback to all agents in the workflow
            workflow_steps = result.get("workflow_steps", [])
            logger.debug(f"Processing {len(workflow_steps)} workflow steps for feedback")

            for step in workflow_steps:
                agent_id = step["agent"]

                # Create mock action for feedback
                # Convert string value back to ActionType enum
                action_type_value = step["action"]
                action_type = None

                logger.debug(f"Processing feedback for agent {agent_id} with action {action_type_value}")

                # Find the ActionType enum by value
                for at in ActionType:
                    if at.value == action_type_value:
                        action_type = at
                        break

                if action_type is None:
                    logger.warning(f"Unknown action type: {action_type_value}, using default")
                    action_type = ActionType.RESPOND_DIRECTLY

                action = RLAction(
                    action_type=action_type,
                    confidence=step.get("confidence", 0.5)
                )

                logger.debug(f"Created RLAction for {agent_id}: {action.action_type.value}")
        except Exception as e:
            logger.error(f"Error in _provide_automatic_feedback: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
            
            # Calculate and provide reward
            reward = self.rl_coordinator.provide_reward_feedback(
                agent_id=agent_id,
                action=action,
                outcome=outcome
            )
            
            # Update agent state
            if agent_id in self.agent_states:
                # Update performance based on reward
                current_perf = self.agent_states[agent_id].recent_performance
                self.agent_states[agent_id].recent_performance = (current_perf * 0.9) + (reward * 0.1)
    
    def enable_training_mode(self):
        """Enable training mode for active RL learning."""
        self.training_mode = True
        logger.info("Training mode enabled - agents will actively learn from interactions")
    
    def disable_training_mode(self):
        """Disable training mode."""
        self.training_mode = False
        logger.info("Training mode disabled")
    
    def get_agent_lightning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive Agent Lightning statistics."""
        return {
            "rl_statistics": self.rl_coordinator.get_training_statistics(),
            "communication_patterns": self.communication_manager.get_communication_statistics(),
            "tool_coordination": self.tool_coordinator.get_coordination_statistics(),
            "agent_states": {
                agent_id: {
                    "confidence_level": state.confidence_level,
                    "workload": state.workload,
                    "recent_performance": state.recent_performance,
                    "conversation_length": len(state.conversation_history)
                }
                for agent_id, state in self.agent_states.items()
            },
            "training_mode": self.training_mode,
            "auto_training": self.auto_training
        }
    
    def export_training_data(self, filepath: str):
        """Export all training data for analysis."""
        training_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_lightning_stats": self.get_agent_lightning_statistics(),
            "conversation_context": self.conversation_context,
            "performance_metrics": self.performance_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Also export AgentOps data
        agentops_file = filepath.replace('.json', '_agentops.json')
        self.monitor.export_training_data(agentops_file)
        
        logger.info(f"Exported training data to {filepath} and {agentops_file}")
    
    async def simulate_user_feedback(self, 
                                   response: str,
                                   satisfaction_score: float,
                                   feedback_text: Optional[str] = None) -> Dict[str, Any]:
        """Simulate user feedback for RL training."""
        feedback = {
            "satisfaction_score": satisfaction_score,
            "feedback_text": feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # This would be called after getting actual user feedback
        # For now, we'll use it for testing and simulation
        
        logger.info(f"Received user feedback: satisfaction={satisfaction_score}")
        return feedback
    
    def get_agent_recommendations(self, agent_id: str) -> Dict[str, Any]:
        """Get recommendations for improving agent performance."""
        if agent_id not in self.rl_coordinator.rl_agents:
            return {"recommendations": ["Agent not found in RL system"]}
        
        rl_agent = self.rl_coordinator.rl_agents[agent_id]
        performance = rl_agent.get_performance_metrics()
        
        recommendations = []
        
        # Performance-based recommendations
        if performance["success_rate"] < 0.7:
            recommendations.append("Consider adjusting exploration rate (epsilon) for better learning")
        
        if performance["average_reward"] < 0.3:
            recommendations.append("Review reward calculation - agent may need different incentives")
        
        if performance["total_actions"] < 100:
            recommendations.append("Agent needs more training data - increase interaction volume")
        
        # Tool usage recommendations
        tool_recs = self.tool_coordinator.get_tool_recommendations(agent_id)
        if tool_recs:
            recommendations.append(f"Consider using these high-efficiency tools: {', '.join(tool_recs[:3])}")
        
        return {
            "agent_id": agent_id,
            "current_performance": performance,
            "recommendations": recommendations,
            "suggested_tools": tool_recs
        }

# Factory function for easy initialization
def create_enhanced_coordinator(agents: Dict[str, BaseAgent], 
                              agentops_api_key: Optional[str] = None) -> EnhancedAgentCoordinator:
    """Create an enhanced coordinator with all Agent Lightning features."""
    return EnhancedAgentCoordinator(agents, agentops_api_key)
