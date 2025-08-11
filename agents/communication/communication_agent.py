
import logging
from typing import Dict, Any, List, Optional
import asyncio
import numpy as np

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
    logger.info("Google ADK loaded successfully for Communication Agent")
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Google ADK not available: {e}. Using mock implementation.")
    ADK_AVAILABLE = False
from utils.language_utils import detect_language, translate_to_english

logger = logging.getLogger(__name__)

class CommunicationAgent(BaseAgent):
    """Handles initial user interaction, query analysis, and rephrasing using Google ADK."""

    def __init__(self, agent_id: str = "communication_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.llm_model = self.system_config.get("llm.models.communication", "gemini-2.0-flash")

        # Initialize Google ADK components if available
        if ADK_AVAILABLE:
            self.session_service = InMemorySessionService()
            self.session_id = "communication_session"
            self.user_id = "system_user"

            # Create ADK LlmAgent for query analysis
            self.adk_agent = LlmAgent(
                name="communication_analyzer",
                model=self.llm_model,
                instruction=self._get_communication_instruction(),
                description="Analyzes and enhances user queries for better processing"
            )

            # Initialize runner
            self.runner = Runner(
                agent=self.adk_agent,
                app_name="communication_app",
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

        # Strategy attributes (simplified without RL)
        self.prompt_strategies = [
            "direct_rewrite",
            "context_enhanced",
            "keyword_focused",
            "intent_based"
        ]
        self.current_strategy = "direct_rewrite"

    async def _ensure_session_initialized(self):
        """Ensure ADK session is initialized."""
        if not ADK_AVAILABLE:
            return  # No session needed for mock

        if not self._session_initialized:
            try:
                await self.session_service.create_session(
                    app_name="communication_app",
                    user_id=self.user_id,
                    session_id=self.session_id,
                    state={}
                )
                self._session_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize ADK session: {e}")
                raise

    def _get_communication_instruction(self) -> str:
        """Get instruction for the ADK communication agent."""
        return """You are an intelligent communication agent for a customer support system. Your role is to analyze user queries and respond appropriately based on the query type.

QUERY CLASSIFICATION:
1. **Greetings/Social**: "hi", "hello", "how are you", "good morning", etc.
2. **Non-technical**: General questions, casual conversation, thanks, goodbye
3. **Technical**: Specific problems, error messages, system issues, account problems

RESPONSE STRATEGY:

For GREETINGS/SOCIAL queries:
- Respond warmly and professionally
- Introduce yourself as an AI support assistant
- Ask how you can help with technical issues

For NON-TECHNICAL queries:
- Provide helpful, friendly responses
- Guide toward technical support if appropriate
- Keep responses concise and professional

For TECHNICAL queries:
- Analyze and enhance the query for knowledge base search
- Extract key technical terms, error messages, system names
- Preserve urgency indicators and specific details
- Output format: "TECHNICAL_QUERY: [enhanced query for knowledge base search]"

EXAMPLES:
Input: "hi there"
Output: "Hello! 👋 I'm your AI support assistant. I'm here to help you with technical issues, account problems, or any questions you might have. How can I assist you today?"

Input: "thanks for your help"
Output: "You're very welcome! 😊 If you have any other questions or need further assistance, please don't hesitate to ask. I'm here to help!"

Input: "how are you doing today"
Output: "I'm doing well, thank you for asking! 😊 As an AI technical support assistant, I'm here and ready to help you with any technical issues or questions you might have. Is there anything I can assist you with today?"

Input: "I can't share my screen during zoom calls"
Output: "TECHNICAL_QUERY: screen sharing not working zoom calls unable to share screen"

Always be helpful, professional, and guide users toward getting the technical support they need."""

    def get_capabilities(self) -> List[str]:
        """Returns the capabilities of the communication agent."""
        return [
            "query_analysis",
            "multilingual_understanding",
            "intent_clarification",
            "llm_interaction"
        ]

    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Processes the user's query by analyzing and rephrasing it using Ollama.

        Args:
            message: The incoming message from the user.

        Returns:
            A new message for the RetrievalAgent with the analyzed query.
        """
        # Handle response from RetrievalAgent: summarize and forward to user, and send to Critic for scoring
        if message.type == MessageType.RESPONSE and message.sender == "retrieval_agent":
            try:
                original_query = message.metadata.get("original_query", "")
                
                # Summarize the retrieved content to create a conversational response
                final_content = await self._summarize_and_respond(message.content, original_query)

                # Forward to critic for evaluation (async via coordinator)
                self.send_message(
                    recipient="critic_agent",
                    content=message.content, # Send original content to critic
                    message_type=MessageType.RESPONSE,
                    metadata={
                        "original_query": original_query,
                        "retrieved_docs": message.metadata.get("retrieved_docs", [])
                    }
                )

                # Forward final summarized answer to user
                return Message(
                    type=MessageType.RESPONSE,
                    content=final_content,
                    metadata={
                        "original_query": original_query,
                        "retrieved_docs": message.metadata.get("retrieved_docs", []),
                        "final_answer_by": self.agent_id
                    },
                    sender=self.agent_id,
                    recipient="user",
                    language=message.language
                )
            except Exception as e:
                logger.error(f"Error forwarding retrieval response: {e}")
                return None

        if message.type != MessageType.QUERY:
            return None

        # Use ADK to intelligently analyze and respond to the query
        response = await self._analyze_with_adk(message.content)
        if response and not response.startswith("TECHNICAL_QUERY:"):
            # This is a direct response (greeting, non-technical, etc.)
            return Message(
                type=MessageType.RESPONSE,
                content=response,
                sender=self.agent_id,
                recipient="user",
            )

        try:
            # 1. Detect language and translate if necessary
            lang_result = detect_language(message.content)
            query_text = message.content
            if lang_result.language != 'en':
                translation = translate_to_english(query_text, lang_result.language)
                if translation.confidence > 0.7:
                    query_text = translation.translated_text
                    logger.info(f"Translated query from {lang_result.language} to English.")

            # 2. Process technical query with ADK
            analyzed_query = query_text
            adk_response = await self._analyze_with_adk(query_text)
            if adk_response and adk_response.startswith("TECHNICAL_QUERY:"):
                # Extract the enhanced query from the ADK response
                analyzed_query = adk_response.replace("TECHNICAL_QUERY:", "").strip()
            elif adk_response:
                # This shouldn't happen for technical queries, but fallback to original
                analyzed_query = query_text

            # 3. Route to RetrievalAgent
            retrieval_message = Message(
                type=MessageType.QUERY,
                content=analyzed_query,
                metadata={
                    "original_query": message.content,
                    "original_language": lang_result.language,
                    "analysis_by": self.agent_id,
                },
                sender=self.agent_id,
                recipient="retrieval_agent",
                language='en'
            )

            self._log_action("query_analysis", {
                "original_query": message.content,
                "analyzed_query": analyzed_query,
                "model": self.llm_model
            })

            return retrieval_message

        except Exception as e:
            logger.error(f"Error in CommunicationAgent: {e}")
            return self._create_error_response(f"An unexpected error occurred: {e}", message)

    async def _analyze_with_adk(self, query: str) -> Optional[str]:
        """
        Uses Google ADK to analyze, rephrase, and enhance the user's query.

        Args:
            query: The user query to analyze.

        Returns:
            The enhanced query, or None if an error occurs.
        """
        if not ADK_AVAILABLE:
            # Mock implementation - simple classification
            logger.info(f"Using mock ADK implementation for query: '{query}'")
            query_lower = query.lower().strip()

            # Simple greeting detection
            if any(greeting in query_lower for greeting in ["hi", "hello", "hey", "good morning", "good afternoon"]):
                return "Hello! 👋 I'm your AI support assistant. I'm here to help you with technical issues, account problems, or any questions you might have. How can I assist you today?"

            # Simple thanks detection
            elif any(thanks in query_lower for thanks in ["thanks", "thank you"]):
                return "You're very welcome! 😊 If you have any other questions or need further assistance, please don't hesitate to ask."

            # Non-technical queries (casual conversation, requests for non-IT things)
            elif any(casual in query_lower for casual in ["coffee", "lunch", "break", "weather", "how are you", "what's up", "joke", "story"]):
                return "I'm an AI technical support assistant, so I can't help with that request. However, I'm here to assist you with any technical issues, account problems, or IT-related questions you might have. Is there anything technical I can help you with?"

            # Check for technical keywords
            elif any(tech in query_lower for tech in ["password", "login", "email", "system", "error", "problem", "issue", "network", "vpn", "zoom", "computer", "software", "application", "website", "account", "access", "install", "update", "configure", "troubleshoot", "fix", "broken", "not working", "can't", "cannot", "unable", "failed", "screen", "share", "sharing"]):
                return f"TECHNICAL_QUERY: {query}"

            # Default to non-technical for unclear queries - let ADK handle it naturally
            else:
                # Continue to ADK for natural response generation
                pass

        try:
            # Ensure session is initialized
            await self._ensure_session_initialized()

            # Create content for ADK
            content = types.Content(
                role='user',
                parts=[types.Part(text=query)]
            )

            # Run the ADK agent
            events = self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            )

            # Extract the response
            analyzed_query = None
            async for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    analyzed_query = event.content.parts[0].text.strip()
                    break

            if analyzed_query:
                logger.info(f"ADK ({self.llm_model}) analyzed query: '{query}' -> '{analyzed_query}'")
                return analyzed_query
            else:
                logger.warning("No response received from ADK agent")
                return None

        except Exception as e:
            logger.error(f"ADK invocation failed: {e}")
            return None

    async def _summarize_and_respond(self, retrieved_content: str, original_query: str) -> str:
        """
        Uses Google ADK to summarize retrieved content and generate a conversational response.
        """
        print(f"DEBUG: _summarize_and_respond called. ADK_AVAILABLE = {ADK_AVAILABLE}")
        if not ADK_AVAILABLE:
            # If ADK is not available, just return the content as is.
            print("DEBUG: ADK not available, returning original content.")
            return retrieved_content

        prompt = f"""You are a helpful AI support assistant. You have been given the following information from a knowledge base to help answer a user's query. Your task is to synthesize this information into a clear, concise, and friendly response to the user. Do not simply list the information. Explain it in a conversational way.

Knowledge Base Information:
---
{retrieved_content}
---

User's original query was: "{original_query}"

Now, formulate a response to the user.
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
            summarized_response = None
            async for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    summarized_response = event.content.parts[0].text.strip()
                    break
            
            if summarized_response:
                logger.info(f"ADK summarized response for query '{original_query}'")
                return summarized_response
            else:
                logger.warning("No summary response received from ADK agent, returning original content.")
                return retrieved_content

        except Exception as e:
            logger.error(f"ADK summarization failed: {e}")
            return retrieved_content

    def _get_strategy_prompt(self, query: str, strategy: str) -> str:
        """Get prompt based on the selected strategy."""
        base_query = f'User message: "{query}"'

        if strategy == "direct_rewrite":
            return f"""
Rewrite the user's message into a concise, precise KB search query.

{base_query}

Rules:
- One line, 3-12 words.
- Include product/app names and error cues.
- No filler; return ONLY the rewritten query.
"""

        elif strategy == "context_enhanced":
            return f"""
Analyze the user's message and create an enhanced search query with context.

{base_query}

Rules:
- Add relevant technical context and synonyms.
- Include related terms that might appear in documentation.
- Keep it focused but comprehensive.
- Return ONLY the enhanced query.
"""

        elif strategy == "keyword_focused":
            return f"""
Extract the most important keywords from the user's message for knowledge base search.

{base_query}

Rules:
- Focus on technical terms, product names, and action words.
- Remove filler words and casual language.
- Prioritize searchable terms.
- Return ONLY the keyword-focused query.
"""

        elif strategy == "intent_based":
            return f"""
Identify the user's intent and create a search query that captures their goal.

{base_query}

Rules:
- Focus on what the user wants to achieve.
- Include the problem type and desired outcome.
- Make it specific to support documentation.
- Return ONLY the intent-based query.
"""

        else:
            # Fallback to direct rewrite
            return self._get_strategy_prompt(query, "direct_rewrite")





    # Utility Methods
    def _assess_query_complexity(self, query: str) -> float:
        """Assess the complexity of a query (0-1 scale)."""
        factors = 0
        query_lower = query.lower()

        # Length factor
        if len(query.split()) > 10:
            factors += 0.3

        # Technical terms
        tech_terms = ["configuration", "authentication", "synchronization", "troubleshoot", "diagnostic"]
        if any(term in query_lower for term in tech_terms):
            factors += 0.4

        # Multiple issues mentioned
        if any(word in query_lower for word in ["and", "also", "additionally", "furthermore"]):
            factors += 0.3

        return min(1.0, factors)

    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Creates a standardized error message."""
        return Message(
            type=MessageType.ERROR,
            content=error_message,
            sender=self.agent_id,
            recipient=original_message.sender,
            metadata={"original_message_id": original_message.id}
        )
