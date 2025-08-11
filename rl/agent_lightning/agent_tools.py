"""
Agent Tools Framework for Agent Lightning.
Converts agents into tools that can be used by other agents using Google ADK.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from pydantic import BaseModel, Field
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None

from agents.base_agent import Message, MessageType, BaseAgent
from rl.agent_lightning.agentops_integration import get_monitor

logger = logging.getLogger(__name__)

class AgentToolInput(BaseModel):
    """Input schema for agent tools."""
    query: str = Field(description="The query or request to send to the agent")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for the agent")
    priority: Optional[str] = Field(default="normal", description="Priority level: low, normal, high, urgent")

class AgentToolOutput(BaseModel):
    """Output schema for agent tools."""
    response: str = Field(description="The agent's response")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    metadata: Dict[str, Any] = Field(description="Additional metadata from the agent")
    processing_time_ms: float = Field(description="Time taken to process the request")

class AgentTool:
    """Wrapper that converts an agent into a tool for other agents."""
    
    def __init__(self, agent: BaseAgent, tool_name: str, description: str):
        self.agent = agent
        self.tool_name = tool_name
        self.description = description
        self.monitor = get_monitor()
        
        # Performance tracking
        self.call_count = 0
        self.total_processing_time = 0.0
        self.success_rate = 0.0
        self.recent_calls: List[Dict[str, Any]] = []
    
    async def __call__(self, query: str, context: Optional[Dict[str, Any]] = None, priority: str = "normal") -> Dict[str, Any]:
        """Execute the agent tool."""
        start_time = time.time()
        self.call_count += 1
        
        try:
            # Create message for the agent
            message = Message(
                type=MessageType.QUERY,
                content=query,
                metadata={
                    "context": context or {},
                    "priority": priority,
                    "tool_call": True,
                    "caller_tool": self.tool_name
                },
                sender="tool_caller",
                recipient=self.agent.agent_id
            )
            
            # Process message through the agent
            response_message = await self.agent.process_message(message)
            
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            if response_message:
                # Extract confidence from metadata if available
                confidence = response_message.metadata.get("confidence", 0.8)
                
                result = {
                    "response": response_message.content,
                    "confidence": confidence,
                    "metadata": response_message.metadata,
                    "processing_time_ms": processing_time
                }
                
                # Record successful call
                self._record_call(True, processing_time, result)
                
                # Monitor the interaction
                self.monitor.record_agent_interaction(
                    agent_id=self.agent.agent_id,
                    action_type="tool_call",
                    input_message=message,
                    output_message=response_message,
                    decision_context={"tool_name": self.tool_name, "priority": priority}
                )
                
                return result
            else:
                # No response from agent
                result = {
                    "response": "Agent did not provide a response",
                    "confidence": 0.0,
                    "metadata": {"error": "no_response"},
                    "processing_time_ms": processing_time
                }
                
                self._record_call(False, processing_time, result)
                return result
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_result = {
                "response": f"Error executing agent tool: {str(e)}",
                "confidence": 0.0,
                "metadata": {"error": str(e)},
                "processing_time_ms": processing_time
            }
            
            self._record_call(False, processing_time, error_result)
            logger.error(f"Agent tool {self.tool_name} failed: {e}")
            return error_result
    
    def _record_call(self, success: bool, processing_time: float, result: Dict[str, Any]):
        """Record tool call statistics."""
        call_record = {
            "timestamp": time.time(),
            "success": success,
            "processing_time_ms": processing_time,
            "confidence": result.get("confidence", 0.0)
        }
        
        self.recent_calls.append(call_record)
        
        # Keep only last 100 calls
        if len(self.recent_calls) > 100:
            self.recent_calls = self.recent_calls[-100:]
        
        # Update success rate
        recent_successes = sum(1 for call in self.recent_calls[-20:] if call["success"])
        self.success_rate = recent_successes / min(20, len(self.recent_calls))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent tool."""
        avg_processing_time = self.total_processing_time / max(1, self.call_count)
        
        return {
            "tool_name": self.tool_name,
            "agent_id": self.agent.agent_id,
            "call_count": self.call_count,
            "success_rate": self.success_rate,
            "average_processing_time_ms": avg_processing_time,
            "recent_performance": self.recent_calls[-10:] if self.recent_calls else []
        }

class AgentToolRegistry:
    """Registry for managing agent tools."""
    
    def __init__(self):
        self.tools: Dict[str, AgentTool] = {}
        self.tool_usage_stats: Dict[str, int] = {}
        self.monitor = get_monitor()
    
    def register_agent_as_tool(self, agent: BaseAgent, tool_name: str, description: str) -> AgentTool:
        """Register an agent as a tool."""
        agent_tool = AgentTool(agent, tool_name, description)
        self.tools[tool_name] = agent_tool
        self.tool_usage_stats[tool_name] = 0
        
        logger.info(f"Registered agent {agent.agent_id} as tool: {tool_name}")
        return agent_tool
    
    def get_tool(self, tool_name: str) -> Optional[AgentTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_functions(self) -> List[Callable]:
        """Get list of tool functions for Google ADK integration."""
        tool_functions = []
        
        for tool_name, agent_tool in self.tools.items():
            # Create a function that can be used by Google ADK
            async def tool_function(query: str, context: Optional[Dict[str, Any]] = None, priority: str = "normal") -> str:
                """Tool function for Google ADK integration."""
                result = await agent_tool(query, context, priority)
                
                # Update usage stats
                self.tool_usage_stats[tool_name] += 1
                
                # Return formatted response
                return json.dumps({
                    "response": result["response"],
                    "confidence": result["confidence"],
                    "processing_time_ms": result["processing_time_ms"]
                })
            
            # Set function metadata for ADK
            tool_function.__name__ = f"use_{tool_name}"
            tool_function.__doc__ = f"{agent_tool.description}. Returns JSON with response, confidence, and processing time."
            
            tool_functions.append(tool_function)
        
        return tool_functions
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool registry statistics."""
        return {
            "total_tools": len(self.tools),
            "tool_usage_stats": self.tool_usage_stats,
            "tool_performance": {
                name: tool.get_performance_metrics()
                for name, tool in self.tools.items()
            }
        }

class MultiAgentToolCoordinator:
    """Coordinates tool usage between multiple agents."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.registry = AgentToolRegistry()
        self.monitor = get_monitor()
        
        # Tool usage optimization
        self.tool_recommendations: Dict[str, List[str]] = {}  # agent_id -> recommended tools
        self.tool_efficiency_matrix: Dict[str, Dict[str, float]] = {}  # tool -> agent -> efficiency
        
        # ADK agent for tool coordination
        if ADK_AVAILABLE:
            self.session_service = InMemorySessionService()
            self.coordinator_agent = self._create_coordinator_agent()
            self.runner = Runner(
                agent=self.coordinator_agent,
                app_name="tool_coordination",
                session_service=self.session_service
            )
        else:
            self.session_service = None
            self.coordinator_agent = None
            self.runner = None
    
    def _create_coordinator_agent(self) -> LlmAgent:
        """Create an ADK agent for tool coordination."""
        return LlmAgent(
            name="tool_coordinator",
            model=self.model_name,
            instruction="""You are a tool coordination agent for a multi-agent system.

Your role is to:
1. Analyze tool usage requests and recommend optimal tools
2. Suggest tool combinations for complex tasks
3. Optimize tool usage patterns based on performance data
4. Detect when new tools might be needed

When analyzing a tool request, respond with JSON:
{
    "recommended_tools": ["tool1", "tool2"],
    "execution_strategy": "sequential|parallel|conditional",
    "expected_efficiency": 0.0-1.0,
    "reasoning": "explanation of recommendation"
}

Focus on maximizing efficiency and task completion success.""",
            description="Coordinates optimal tool usage across agents"
        )
    
    async def recommend_tools(self, 
                            requesting_agent: str,
                            task_description: str,
                            available_tools: List[str]) -> Dict[str, Any]:
        """Recommend optimal tools for a given task."""
        if not ADK_AVAILABLE or not self.coordinator_agent:
            # Fallback to simple recommendation
            return {
                "recommended_tools": available_tools[:2] if available_tools else [],
                "execution_strategy": "sequential",
                "expected_efficiency": 0.7,
                "reasoning": "Fallback recommendation"
            }
        
        try:
            # Get tool performance data
            tool_stats = self.registry.get_registry_statistics()
            
            prompt = f"""
Analyze this tool usage request:

Requesting Agent: {requesting_agent}
Task: {task_description}
Available Tools: {', '.join(available_tools)}

Tool Performance Data:
{json.dumps(tool_stats, indent=2)}

Recommend the optimal tools and execution strategy.
"""
            
            content = types.Content(role='user', parts=[types.Part(text=prompt)])
            
            session_id = f"tool_rec_{int(time.time())}"
            await self.session_service.create_session(
                app_name="tool_coordination",
                user_id="system",
                session_id=session_id
            )
            
            events = self.runner.run_async(
                user_id="system",
                session_id=session_id,
                new_message=content
            )
            
            recommendation = None
            async for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    try:
                        recommendation = json.loads(event.content.parts[0].text.strip())
                        break
                    except json.JSONDecodeError:
                        recommendation = {
                            "recommended_tools": available_tools[:1] if available_tools else [],
                            "execution_strategy": "sequential",
                            "expected_efficiency": 0.6,
                            "reasoning": "JSON parsing failed, using fallback"
                        }
                        break
            
            return recommendation or {
                "recommended_tools": [],
                "execution_strategy": "sequential",
                "expected_efficiency": 0.5,
                "reasoning": "No recommendation generated"
            }
            
        except Exception as e:
            logger.error(f"Tool recommendation failed: {e}")
            return {
                "recommended_tools": available_tools[:1] if available_tools else [],
                "execution_strategy": "sequential",
                "expected_efficiency": 0.5,
                "reasoning": f"Error in recommendation: {str(e)}"
            }
    
    def register_all_agents_as_tools(self, agents: Dict[str, BaseAgent]):
        """Register all agents as tools for cross-agent usage."""
        tool_definitions = {
            "communication_agent": "Analyzes and classifies user queries, handles greetings and non-technical responses",
            "retrieval_agent": "Searches knowledge base and synthesizes responses using RAG",
            "critic_agent": "Evaluates response quality and provides feedback scores",
            "escalation_agent": "Assesses severity and handles escalation for critical issues"
        }
        
        for agent_id, agent in agents.items():
            if agent_id in tool_definitions:
                self.registry.register_agent_as_tool(
                    agent=agent,
                    tool_name=f"{agent_id}_tool",
                    description=tool_definitions[agent_id]
                )
        
        logger.info(f"Registered {len(agents)} agents as tools")
    
    def create_enhanced_agent_with_tools(self, 
                                       base_agent: BaseAgent,
                                       available_tools: List[str]) -> Optional[LlmAgent]:
        """Create an enhanced ADK agent that can use other agents as tools."""
        if not ADK_AVAILABLE:
            logger.warning("Google ADK not available, cannot create enhanced agent")
            return None
        
        # Get tool functions
        tool_functions = []
        for tool_name in available_tools:
            tool = self.registry.get_tool(tool_name)
            if tool:
                # Create a wrapper function for ADK
                def create_tool_function(agent_tool: AgentTool):
                    async def tool_func(query: str, context: str = None, priority: str = "normal") -> str:
                        """Tool function wrapper for ADK."""
                        context_dict = json.loads(context) if context else {}
                        result = await agent_tool(query, context_dict, priority)
                        return json.dumps(result)
                    
                    tool_func.__name__ = f"use_{agent_tool.tool_name}"
                    tool_func.__doc__ = f"{agent_tool.description}. Input: query (str), context (JSON str), priority (str). Returns: JSON response."
                    return tool_func
                
                tool_functions.append(create_tool_function(tool))
        
        # Create enhanced ADK agent
        enhanced_agent = LlmAgent(
            name=f"enhanced_{base_agent.agent_id}",
            model=self.model_name,
            instruction=f"""You are an enhanced version of {base_agent.agent_id} with access to other agents as tools.

Your capabilities include:
- {', '.join(base_agent.get_capabilities())}

Available agent tools:
{chr(10).join([f"- {tool_name}: {self.registry.get_tool(tool_name).description}" for tool_name in available_tools if self.registry.get_tool(tool_name)])}

Use these tools intelligently to:
1. Delegate specialized tasks to appropriate agents
2. Gather additional context or validation
3. Coordinate complex multi-step workflows
4. Optimize overall system performance

Always consider tool efficiency and avoid unnecessary tool calls. Provide clear, helpful responses to users.""",
            description=f"Enhanced {base_agent.agent_id} with multi-agent tool access",
            tools=tool_functions
        )
        
        return enhanced_agent
    
    def update_tool_efficiency(self, tool_name: str, agent_id: str, efficiency_score: float):
        """Update efficiency score for tool usage by specific agent."""
        if tool_name not in self.tool_efficiency_matrix:
            self.tool_efficiency_matrix[tool_name] = {}
        
        self.tool_efficiency_matrix[tool_name][agent_id] = efficiency_score
        
        # Update recommendations
        if agent_id not in self.tool_recommendations:
            self.tool_recommendations[agent_id] = []
        
        # Sort tools by efficiency for this agent
        agent_tools = [(tool, scores.get(agent_id, 0.0)) 
                      for tool, scores in self.tool_efficiency_matrix.items()]
        agent_tools.sort(key=lambda x: x[1], reverse=True)
        
        self.tool_recommendations[agent_id] = [tool for tool, _ in agent_tools[:5]]  # Top 5 tools
        
        logger.debug(f"Updated tool efficiency: {tool_name} for {agent_id} = {efficiency_score}")
    
    def get_tool_recommendations(self, agent_id: str) -> List[str]:
        """Get recommended tools for a specific agent."""
        return self.tool_recommendations.get(agent_id, [])
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool coordination statistics."""
        return {
            "registry_stats": self.registry.get_registry_statistics(),
            "tool_efficiency_matrix": self.tool_efficiency_matrix,
            "tool_recommendations": self.tool_recommendations,
            "total_tool_calls": sum(self.registry.tool_usage_stats.values())
        }

# Global tool coordinator instance
_tool_coordinator: Optional[MultiAgentToolCoordinator] = None

def get_tool_coordinator() -> MultiAgentToolCoordinator:
    """Get the global tool coordinator instance."""
    global _tool_coordinator
    if _tool_coordinator is None:
        _tool_coordinator = MultiAgentToolCoordinator()
    return _tool_coordinator

def initialize_agent_tools(agents: Dict[str, BaseAgent]) -> MultiAgentToolCoordinator:
    """Initialize the agent tools framework with available agents."""
    global _tool_coordinator
    _tool_coordinator = MultiAgentToolCoordinator()
    _tool_coordinator.register_all_agents_as_tools(agents)
    return _tool_coordinator
