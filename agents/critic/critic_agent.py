

import logging
from typing import Dict, Any, List, Optional
import asyncio
import json

from agents.base_agent import BaseAgent, Message, MessageType
# Google ADK imports - will be used when properly configured
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from agent-specific .env file
agent_dir = Path(__file__).parent
load_dotenv(agent_dir / '.env')

try:
    # Set up Google API key for ADK
    if not os.getenv('GOOGLE_API_KEY') and os.getenv('GEMINI_API_KEY'):
        os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY')

    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    ADK_AVAILABLE = True
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Google ADK loaded successfully for Critic Agent")
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Google ADK not available: {e}. Using mock implementation.")
    ADK_AVAILABLE = False

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """Evaluates the quality of a response using Google ADK."""

    def __init__(self, agent_id: str = "critic_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.llm_model = self.system_config.get("llm.models.critic", "gemini-2.0-flash")

        # Initialize Google ADK components if available
        if ADK_AVAILABLE:
            self.session_service = InMemorySessionService()
            self.session_id = "critic_session"
            self.user_id = "system_user"

            # Create ADK LlmAgent for response evaluation
            self.adk_agent = LlmAgent(
                name="response_critic",
                model=self.llm_model,
                instruction=self._get_critic_instruction(),
                description="Evaluates the quality and accuracy of support responses"
            )

            # Initialize runner
            self.runner = Runner(
                agent=self.adk_agent,
                app_name="critic_app",
                session_service=self.session_service
            )

            # Session will be initialized lazily when first needed
            self._session_initialized = False
        else:
            # Mock implementation for testing
            self.session_service = None
            self.adk_agent = None
            self.runner = None
            self._session_initialized = True  # No session needed for mock

    async def _ensure_session_initialized(self):
        """Ensure ADK session is initialized."""
        if not self._session_initialized:
            try:
                await self.session_service.create_session(
                    app_name="critic_app",
                    user_id=self.user_id,
                    session_id=self.session_id,
                    state={}
                )
                self._session_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize ADK session: {e}")
                raise

    def _get_critic_instruction(self) -> str:
        """Get instruction for the ADK critic agent."""
        return """You are a response quality evaluation agent for a customer support system. Your role is to:

1. Evaluate the quality, accuracy, and helpfulness of support responses
2. Assess whether the response adequately addresses the user's query
3. Check for completeness, clarity, and actionability
4. Provide a quality score and detailed feedback

Evaluation Criteria:
- **Accuracy**: Is the information correct and up-to-date?
- **Completeness**: Does it fully address the user's question?
- **Clarity**: Is the response easy to understand?
- **Actionability**: Does it provide clear next steps?
- **Relevance**: Is it directly related to the user's query?
- **Tone**: Is it professional and helpful?

Output Format:
Provide a JSON response with:
- "score": A number from 0.0 to 1.0 (1.0 being perfect)
- "feedback": Detailed explanation of the evaluation
- "strengths": What the response does well
- "improvements": Specific suggestions for improvement
- "overall_assessment": Brief summary of quality level

Input: Original user query + Support response to evaluate
Output: Structured quality assessment in JSON format"""

    def get_capabilities(self) -> List[str]:
        """Returns the capabilities of the critic agent."""
        return ["response_evaluation", "quality_assessment", "feedback_generation", "llm_evaluation"]

    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Processes a response from the RetrievalAgent and evaluates its quality.

        Args:
            message: The incoming message from the RetrievalAgent.

        Returns:
            A final response message for the user.
        """
        if message.type != MessageType.RESPONSE:
            return None

        try:
            # 1. Evaluate the response using Google ADK
            evaluation = await self._evaluate_with_adk(message)
            if not evaluation:
                # Fallback evaluation when ADK fails
                evaluation = self._create_fallback_evaluation(message)
                logger.warning("Using fallback evaluation due to ADK failure")

            # 2. Forward the responder's content to user with evaluation attached
            final_response = Message(
                type=MessageType.RESPONSE,
                content=message.content,  # Pass the synthesized content through
                metadata={
                    "original_query": message.metadata.get("original_query"),
                    "retrieved_docs": message.metadata.get("retrieved_docs", []),
                    "evaluation": evaluation,
                    "final_answer_by": self.agent_id,
                },
                sender=self.agent_id,
                recipient="user",  # Final destination is the user
                language=message.language
            )

            self._log_action("response_evaluation", {"evaluation_score": evaluation.get("overall_score", 0)})

            return final_response

        except Exception as e:
            logger.error(f"Error in CriticAgent: {e}")
            return self._create_error_response(f"An unexpected error occurred during evaluation: {e}", message)

    async def _evaluate_with_adk(self, message: Message) -> Optional[Dict[str, Any]]:
        """
        Uses Google ADK to evaluate the response based on the original query and retrieved context.

        Args:
            message: The message from the RetrievalAgent.

        Returns:
            A dictionary containing the evaluation scores and feedback.
        """
        query = message.metadata.get("original_query", "")
        retrieved_docs = message.metadata.get("retrieved_docs", [])
        response_text = message.content

        context = "\n\n".join([f"**Source:** {doc['chunk']['source_file']}\n**Content:** {doc['chunk']['content']}" for doc in retrieved_docs])

        prompt = f"""
Evaluate this support response:

User query: {query}

Context used:
{context}

Response to evaluate:
{response_text}

Please provide a detailed evaluation in JSON format.
"""

        try:
            # Ensure session is initialized
            await self._ensure_session_initialized()

            # Create content for ADK
            content = types.Content(
                role='user',
                parts=[types.Part(text=prompt)]
            )

            # Run the ADK agent
            events = self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            )

            # Extract the response
            raw_response = None
            async for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    raw_response = event.content.parts[0].text.strip()
                    break

            if not raw_response:
                logger.warning("No response received from ADK agent")
                return None

            # Parse JSON response
            cleaned = raw_response.strip()
            # If model replied with plain text or non-JSON, attempt to extract JSON substring
            if not cleaned.startswith('{'):
                start = cleaned.find('{')
                end = cleaned.rfind('}')
                if start != -1 and end != -1 and end > start:
                    cleaned = cleaned[start:end+1]
            # Remove code fences if present
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            evaluation = json.loads(cleaned)
            logger.info(f"ADK ({self.llm_model}) evaluated response with score: {evaluation.get('overall_score', evaluation.get('score'))}")
            return evaluation
        except Exception as e:
            logger.error(f"ADK evaluation failed: {e}.")
            return None

    def _create_fallback_evaluation(self, message: Message) -> Dict[str, Any]:
        """Create a basic evaluation when ADK is not available."""
        response_text = message.content or ""
        retrieved_docs = message.metadata.get("retrieved_docs", [])

        # Basic scoring based on response characteristics
        confidence = 0.7 if len(response_text) > 50 else 0.4
        protocol_efficiency = 0.8 if retrieved_docs else 0.5
        consensus_level = 0.6  # Default moderate consensus

        # Adjust based on content quality indicators
        if any(word in response_text.lower() for word in ["step", "solution", "contact", "resolve"]):
            confidence += 0.1
            protocol_efficiency += 0.1

        if len(retrieved_docs) > 1:
            consensus_level += 0.2

        # Cap at 1.0
        confidence = min(1.0, confidence)
        protocol_efficiency = min(1.0, protocol_efficiency)
        consensus_level = min(1.0, consensus_level)

        return {
            "confidence": confidence,
            "protocol_efficiency": protocol_efficiency,
            "consensus_level": consensus_level,
            "overall_score": (confidence + protocol_efficiency + consensus_level) / 3,
            "reasoning": "Fallback evaluation based on response characteristics",
            "evaluation_method": "fallback"
        }

    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Creates a standardized error message."""
        return Message(
            type=MessageType.ERROR,
            content=error_message,
            sender=self.agent_id,
            recipient="user",
            metadata={"original_message_id": original_message.id}
        )

