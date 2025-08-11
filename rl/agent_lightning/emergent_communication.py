"""
Emergent Communication Framework for Agent Lightning.
Implements intelligent communication protocols between agents using Google ADK.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

from agents.base_agent import Message, MessageType, BaseAgent
from rl.agent_lightning.agentops_integration import get_monitor

logger = logging.getLogger(__name__)

class CommunicationProtocol(Enum):
    """Types of communication protocols between agents."""
    DIRECT = "direct"  # Direct message passing
    BROADCAST = "broadcast"  # Broadcast to all agents
    TOOL_CALL = "tool_call"  # Agent used as a tool by another agent
    NEGOTIATION = "negotiation"  # Multi-turn negotiation between agents
    CONSENSUS = "consensus"  # Consensus building among multiple agents

@dataclass
class CommunicationIntent:
    """Represents the intent behind agent communication."""
    purpose: str  # 'query_routing', 'quality_check', 'escalation_request', etc.
    urgency: float  # 0.0 to 1.0
    expected_response_type: str
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0

class EmergentCommunicationManager:
    """Manages emergent communication patterns between agents."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.monitor = get_monitor()
        
        # Communication state
        self.active_conversations: Dict[str, List[Message]] = {}
        self.communication_patterns: Dict[str, int] = {}
        self.agent_relationships: Dict[str, Dict[str, float]] = {}  # Trust/efficiency scores
        
        # ADK components for communication intelligence
        if ADK_AVAILABLE:
            self.session_service = InMemorySessionService()
            self.communication_agent = self._create_communication_agent()
            self.runner = Runner(
                agent=self.communication_agent,
                app_name="emergent_communication",
                session_service=self.session_service
            )
        else:
            self.session_service = None
            self.communication_agent = None
            self.runner = None
    
    def _create_communication_agent(self) -> LlmAgent:
        """Create an ADK agent for communication intelligence."""
        return LlmAgent(
            name="communication_coordinator",
            model=self.model_name,
            instruction="""You are an intelligent communication coordinator for a multi-agent system.
            
Your role is to:
1. Analyze communication intents and determine optimal routing
2. Suggest communication protocols (direct, broadcast, tool_call, negotiation, consensus)
3. Evaluate communication efficiency and suggest improvements
4. Detect emergent communication patterns

When analyzing a communication request, respond with JSON:
{
    "recommended_protocol": "direct|broadcast|tool_call|negotiation|consensus",
    "routing_suggestion": "agent_id or 'all' for broadcast",
    "urgency_assessment": 0.0-1.0,
    "expected_outcome": "description of expected result",
    "optimization_suggestions": ["suggestion1", "suggestion2"]
}

Be concise and focus on improving multi-agent coordination efficiency.""",
            description="Coordinates intelligent communication between agents"
        )
    
    async def analyze_communication_intent(self, 
                                         sender_agent: str,
                                         message: Message,
                                         available_agents: List[str]) -> Dict[str, Any]:
        """Analyze communication intent and suggest optimal routing."""
        if not ADK_AVAILABLE or not self.communication_agent:
            # Fallback to rule-based routing
            return self._fallback_routing_analysis(sender_agent, message, available_agents)
        
        try:
            # Create analysis prompt
            prompt = f"""
Analyze this communication request:

Sender: {sender_agent}
Message Type: {message.type.value}
Content: {message.content}
Available Agents: {', '.join(available_agents)}
Current Metadata: {json.dumps(message.metadata, indent=2)}

Determine the optimal communication strategy for this multi-agent system.
"""
            
            # Run ADK analysis
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
            session_id = f"comm_analysis_{int(time.time())}"
            await self.session_service.create_session(
                app_name="emergent_communication",
                user_id="system",
                session_id=session_id
            )
            
            events = self.runner.run_async(
                user_id="system",
                session_id=session_id,
                new_message=content
            )
            
            analysis_result = None
            async for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    try:
                        analysis_result = json.loads(event.content.parts[0].text.strip())
                        break
                    except json.JSONDecodeError:
                        # Fallback if JSON parsing fails
                        analysis_result = {"recommended_protocol": "direct", "routing_suggestion": available_agents[0] if available_agents else "unknown"}
                        break
            
            return analysis_result or self._fallback_routing_analysis(sender_agent, message, available_agents)
            
        except Exception as e:
            logger.error(f"Communication analysis failed: {e}")
            return self._fallback_routing_analysis(sender_agent, message, available_agents)
    
    def _fallback_routing_analysis(self, sender_agent: str, message: Message, available_agents: List[str]) -> Dict[str, Any]:
        """Fallback routing analysis when ADK is not available."""
        # Simple rule-based routing
        if message.type == MessageType.QUERY:
            if sender_agent == "communication_agent":
                return {
                    "recommended_protocol": "direct",
                    "routing_suggestion": "retrieval_agent",
                    "urgency_assessment": 0.5,
                    "expected_outcome": "Knowledge base search and response",
                    "optimization_suggestions": ["Cache similar queries"]
                }
        elif message.type == MessageType.RESPONSE:
            if sender_agent == "retrieval_agent":
                return {
                    "recommended_protocol": "direct",
                    "routing_suggestion": "critic_agent",
                    "urgency_assessment": 0.6,
                    "expected_outcome": "Response quality evaluation",
                    "optimization_suggestions": ["Parallel evaluation"]
                }
        
        return {
            "recommended_protocol": "direct",
            "routing_suggestion": available_agents[0] if available_agents else "unknown",
            "urgency_assessment": 0.5,
            "expected_outcome": "Standard processing",
            "optimization_suggestions": []
        }
    
    async def facilitate_communication(self,
                                     sender_agent: str,
                                     message: Message,
                                     available_agents: List[str]) -> List[Message]:
        """Facilitate intelligent communication between agents."""
        start_time = time.time()
        
        # Analyze communication intent
        analysis = await self.analyze_communication_intent(sender_agent, message, available_agents)
        
        # Record communication event
        protocol = analysis.get("recommended_protocol", "direct")
        routing = analysis.get("routing_suggestion", "unknown")
        
        # Execute communication based on analysis
        routed_messages = []
        
        if protocol == "direct" and routing in available_agents:
            # Direct communication
            routed_message = Message(
                type=message.type,
                content=message.content,
                metadata={
                    **message.metadata,
                    "communication_analysis": analysis,
                    "routed_by": "emergent_communication_manager"
                },
                sender=message.sender,
                recipient=routing
            )
            routed_messages.append(routed_message)
            
        elif protocol == "broadcast":
            # Broadcast to multiple agents
            for agent_id in available_agents:
                if agent_id != sender_agent:
                    routed_message = Message(
                        type=message.type,
                        content=message.content,
                        metadata={
                            **message.metadata,
                            "communication_analysis": analysis,
                            "broadcast_by": "emergent_communication_manager"
                        },
                        sender=message.sender,
                        recipient=agent_id
                    )
                    routed_messages.append(routed_message)
        
        # Record communication event
        latency_ms = (time.time() - start_time) * 1000
        self.monitor.record_communication_event(
            sender_agent=sender_agent,
            receiver_agent=routing,
            message_content=message.content,
            communication_type=protocol,
            success=len(routed_messages) > 0,
            latency_ms=latency_ms
        )
        
        # Update communication patterns
        pattern_key = f"{sender_agent}->{routing}:{protocol}"
        self.communication_patterns[pattern_key] = self.communication_patterns.get(pattern_key, 0) + 1
        
        return routed_messages
    
    def update_agent_relationship(self, agent1: str, agent2: str, efficiency_score: float):
        """Update the relationship efficiency between two agents."""
        if agent1 not in self.agent_relationships:
            self.agent_relationships[agent1] = {}
        if agent2 not in self.agent_relationships:
            self.agent_relationships[agent2] = {}
        
        # Update bidirectional relationship
        self.agent_relationships[agent1][agent2] = efficiency_score
        self.agent_relationships[agent2][agent1] = efficiency_score
        
        logger.debug(f"Updated relationship efficiency: {agent1} <-> {agent2} = {efficiency_score}")
    
    def get_optimal_communication_path(self, sender: str, message_type: MessageType) -> List[str]:
        """Determine optimal communication path based on learned patterns."""
        # Analyze historical patterns to suggest optimal routing
        relevant_patterns = {
            k: v for k, v in self.communication_patterns.items()
            if k.startswith(f"{sender}->") and message_type.value in k
        }
        
        if not relevant_patterns:
            return []  # No learned patterns yet
        
        # Return the most frequently used successful pattern
        best_pattern = max(relevant_patterns.items(), key=lambda x: x[1])
        path_info = best_pattern[0].split('->')
        if len(path_info) > 1:
            return [path_info[1].split(':')[0]]  # Extract recipient
        
        return []
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get comprehensive communication statistics."""
        return {
            "total_patterns": len(self.communication_patterns),
            "most_common_patterns": sorted(
                self.communication_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "agent_relationships": self.agent_relationships,
            "active_conversations": len(self.active_conversations),
            "communication_events": self.monitor.get_communication_patterns()
        }
