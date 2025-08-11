

import logging
import re
from typing import Dict, Any, List, Optional
import asyncio

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
    logger.info("Google ADK loaded successfully for Retrieval Agent")
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Google ADK not available: {e}. Using mock implementation.")
    ADK_AVAILABLE = False
from kb.unified_knowledge_base import get_knowledge_base, SearchResult

logger = logging.getLogger(__name__)

class RetrievalAgent(BaseAgent):
    """Retrieves information and uses Google ADK to synthesize answers (RAG)."""

    def __init__(self, agent_id: str = "retrieval_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        self.knowledge_base = get_knowledge_base()
        self.llm_model = self.system_config.get("llm.models.retrieval", "gemini-2.0-flash")
        self.max_results = self.system_config.get("agents.retrieval.max_documents", 5)
        self.min_similarity_score = self.system_config.get("knowledge_base.similarity_threshold", 0.7)

        # Initialize Google ADK components if available
        if ADK_AVAILABLE:
            self.session_service = InMemorySessionService()
            self.session_id = "retrieval_session"
            self.user_id = "system_user"

            # Create ADK LlmAgent for RAG synthesis
            self.adk_agent = LlmAgent(
                name="retrieval_synthesizer",
                model=self.llm_model,
                instruction=self._get_rag_instruction(),
                description="Synthesizes answers from retrieved knowledge base documents"
            )

            # Initialize runner
            self.runner = Runner(
                agent=self.adk_agent,
                app_name="retrieval_app",
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
        if not ADK_AVAILABLE:
            return  # No session needed for mock

        if not self._session_initialized:
            try:
                await self.session_service.create_session(
                    app_name="retrieval_app",
                    user_id=self.user_id,
                    session_id=self.session_id,
                    state={}
                )
                self._session_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize ADK session: {e}")
                raise

    def _get_rag_instruction(self) -> str:
        """Get instruction for the ADK RAG agent."""
        return """You are an intelligent knowledge synthesis agent for a customer support system. Your role is to provide helpful, accurate responses based on the knowledge base context.

RESPONSE STRATEGY:

1. **If relevant context is found:**
   - Synthesize a clear, actionable answer from the provided documents
   - Provide step-by-step instructions when appropriate
   - Include specific details like system requirements, prerequisites, or warnings
   - Structure your response with bullet points or numbered steps for clarity
   - Maintain a helpful, professional tone

2. **If context is insufficient:**
   - Clearly state what information is missing
   - Suggest alternative approaches or resources
   - Recommend escalation if the issue seems complex

3. **If no relevant context:**
   - Acknowledge that the specific information isn't in the knowledge base
   - Provide general guidance if possible
   - Suggest contacting support for specialized help

FORMATTING GUIDELINES:
- Use bullet points (•) for lists
- Use numbered steps (1., 2., 3.) for procedures
- Keep responses concise but comprehensive
- Include relevant warnings or prerequisites
- End with an offer for further assistance

ESCALATION TRIGGERS:
If the query involves:
- Security breaches or data loss
- System-wide outages
- Legal or compliance issues
- Urgent deadlines with critical impact
Then include: "⚠️ This appears to be a high-priority issue that may require immediate escalation."

Always be helpful, accurate, and guide users toward successful resolution of their technical issues."""

    def get_capabilities(self) -> List[str]:
        """Returns the capabilities of the retrieval agent."""
        return [
            "knowledge_retrieval",
            "semantic_search",
            "response_synthesis",
            "rag_with_llm"
        ]

    async def process_message(self, message: Message) -> Optional[Message]:
        """
        Processes a query, retrieves relevant documents, and synthesizes a response.

        Args:
            message: The incoming message from the CommunicationAgent.

        Returns:
            A response message for the CriticAgent to evaluate.
        """
        if message.type != MessageType.QUERY:
            return None

        try:
            # 1. Search the knowledge base (two-pass: strict then lenient)
            search_results = self.knowledge_base.search(
                query=message.content,
                max_results=self.max_results,
                min_score=self.min_similarity_score
            )
            if not search_results:
                # Fallback: lower threshold to get top-k regardless of score
                search_results = self.knowledge_base.search(
                    query=message.content,
                    max_results=self.max_results,
                    min_score=0.0
                )

            # 2. Synthesize a response using Google ADK (RAG)
            synthesized_response = await self._synthesize_with_adk(message.content, search_results)
            if not synthesized_response:
                return self._create_error_response("Failed to synthesize a response.", message)

            # 3. Create a response message for the CommunicationAgent (who will forward to user)
            response_message = Message(
                type=MessageType.RESPONSE,
                content=synthesized_response,
                metadata={
                    "original_query": message.metadata.get("original_query", message.content),
                    "retrieved_docs": [res.to_dict() for res in search_results],
                    "synthesis_by": self.agent_id,
                    "model": self.llm_model
                },
                sender=self.agent_id,
                recipient="communication_agent",
                language=message.language
            )

            self._log_action("response_synthesis", {
                "query": message.content,
                "num_retrieved": len(search_results),
                "response_length": len(synthesized_response)
            })

            return response_message

        except Exception as e:
            logger.error(f"Error in RetrievalAgent: {e}")
            return self._create_error_response(f"An unexpected error occurred: {e}", message)

    async def _synthesize_with_adk(self, query: str, search_results: List[SearchResult]) -> Optional[str]:
        """
        Uses Google ADK to synthesize a helpful response from the retrieved documents.

        Args:
            query: The user's original query.
            search_results: A list of relevant documents from the knowledge base.

        Returns:
            A synthesized, human-readable response.
        """
        if not search_results:
            return "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your query."

        # Prepare the context from retrieved documents
        def _truncate(text: str, limit: int = 800) -> str:
            return text if len(text) <= limit else text[:limit] + "..."

        # Clean content by removing hardcoded answer lines and keeping resolution content
        import re
        def _clean_content(text: str) -> str:
            """Clean content by removing hardcoded answer lines and keeping resolution content."""
            lines = text.split('\n')
            cleaned_lines = []

            for line in lines:
                line = line.strip()
                # Skip hardcoded answer lines, subjects, metadata, and tags
                if (line and
                    not line.startswith('Answer:') and
                    not line.startswith('Subject:') and
                    not line.startswith('Body:') and
                    not line.startswith('Dear ') and
                    not line.startswith('Type:') and
                    not line.startswith('Queue:') and
                    not line.startswith('Priority:') and
                    not line.startswith('Language:') and
                    not line.startswith('Business Type:') and
                    not line.startswith('Tag ') and
                    not re.search(r'Tag \d+:', line) and
                    len(line) > 10):
                    cleaned_lines.append(line)

            return '\n'.join(cleaned_lines)

        context_parts = []
        for i, res in enumerate(search_results[:3]):  # Use fewer but more relevant docs
            cleaned_content = _clean_content(_truncate(res.chunk.content or ""))
            if cleaned_content:  # Only include if there's meaningful content
                context_parts.append(f"Document {i+1} (Relevance: {res.score:.3f}):\n{cleaned_content}")

        if not context_parts:
            # Try to extract any useful information from the raw content
            fallback_info = []
            for res in search_results[:2]:
                if res.chunk.content:
                    # Extract key information even from noisy content
                    content = res.chunk.content
                    if any(keyword in content.lower() for keyword in ["step", "solution", "resolve", "fix", "contact", "email", "phone", "process"]):
                        fallback_info.append(content[:300] + "..." if len(content) > 300 else content)

            if fallback_info:
                return f"Based on the available information:\n\n" + "\n\n".join(fallback_info)
            else:
                return "I found some documents but they don't contain specific resolution information for your query. Please contact support for personalized assistance."

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an expert technical support assistant. Based on the knowledge base documents below, provide a comprehensive, actionable solution to the user's problem.

User's Problem: {query}

Knowledge Base Information:
{context}

Instructions:
1. Start with a direct answer or solution statement
2. Provide specific step-by-step instructions from the knowledge base
3. Include relevant details like contact information, deadlines, or requirements
4. If troubleshooting is needed, present all necessary steps clearly
5. Use numbered steps for processes and bullet points for options
6. Be thorough and decisive - don't just say "here's how to resolve" and stop
7. If multiple solutions exist, explain when to use each one
8. End with next steps or who to contact if the solution doesn't work

Format your response as a complete, helpful answer that fully addresses the user's problem.

Provide your response:"""

        if not ADK_AVAILABLE:
            # Mock implementation - provide a more comprehensive response
            logger.info(f"Using mock ADK implementation for synthesis")
            if not search_results:
                return "I couldn't find relevant information in the knowledge base for your query. Please contact support for assistance."

            # Extract key information from the best result
            best_result = search_results[0]
            content = best_result.chunk.content or ""

            # Try to extract actionable information
            mock_response = f"Based on the knowledge base, here's how to resolve your issue:\n\n"

            # Look for step-by-step instructions or solutions
            if "step" in content.lower() or "solution" in content.lower():
                mock_response += f"**Solution from {best_result.chunk.source_file}:**\n"
                mock_response += content[:500] + ("..." if len(content) > 500 else "")
            else:
                mock_response += f"**Information from {best_result.chunk.source_file}:**\n"
                mock_response += content[:400] + ("..." if len(content) > 400 else "")

            # Add additional context if available
            if len(search_results) > 1:
                mock_response += f"\n\n**Additional Information:**\n"
                mock_response += search_results[1].chunk.content[:200] + ("..." if len(search_results[1].chunk.content) > 200 else "")

            mock_response += "\n\nIf you need further assistance, please contact support."
            return mock_response

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
            synthesized_answer = None
            async for event in events:
                if event.is_final_response() and event.content and event.content.parts:
                    synthesized_answer = event.content.parts[0].text.strip()
                    break

            if synthesized_answer:
                cleaned = self._postprocess_answer(synthesized_answer)
                if not cleaned:
                    cleaned = self._build_extractive_answer(search_results)
                logger.info(f"ADK ({self.llm_model}) synthesized response for query: '{query}'")
                return cleaned
            else:
                logger.warning("No response received from ADK agent")
                return self._build_extractive_answer(search_results)

        except Exception as e:
            logger.error(f"ADK invocation for synthesis failed: {e}")
            return self._build_extractive_answer(search_results)

    def _postprocess_answer(self, text: str) -> str:
        """Remove chain-of-thought; keep only concise bullet points (max 6 lines)."""
        if not text:
            return ""
        # Strip HTML tags and known markers
        t = text.replace("```json", "").replace("```", "").strip()
        t = re.sub(r"<[^>]+>", "", t)
        # If think tags present, strip content between them
        if "<think>" in t:
            end = t.rfind("</think>")
            if end != -1:
                t = t[end + len("</think>"):].strip()
            else:
                # Drop everything up to first bullet
                pass
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        # Keep from first bullet-like line
        start_idx = 0
        for i, ln in enumerate(lines):
            if ln.startswith("-") or ln.startswith("*") or (ln[:2].isdigit() and ln.strip().find('.') == 1) or ln[:3].strip().isdigit():
                start_idx = i
                break
        kept = lines[start_idx:]
        # Filter out ticket metadata lines and CSV metadata
        noise_patterns = [
            r"^ticket\b", r"^complaint id\b", r"^employee name\b", r"^domain\b", r"^priority\b",
            r"^queue\b", r"^business type\b", r"^tag\b", r"^language\b", r"^source:\b",
            r"^type:\b", r"customer support team", r"tag \d+:", r"^subject:\b"
        ]
        filtered = []
        for ln in kept:
            low = ln.lower()
            if any(re.search(pat, low) for pat in noise_patterns):
                continue
            # Remove trailing periods-only lines or very long metadata-like lines
            if len(ln) > 300:
                continue
            filtered.append(ln)
        kept = filtered
        # If no bullets at all, trigger extractive fallback by returning empty string
        if not any(ln.startswith(('-', '*')) or (ln[:2].isdigit() and '.' in ln[:4]) for ln in kept):
            return ""
        else:
            # Keep max 6 bullet lines
            kept = [ln for ln in kept if ln][:6]
        return "\n".join(kept).strip()

    def _build_extractive_answer(self, search_results: List[SearchResult]) -> str:
        """Create a helpful answer by extracting relevant resolution content from retrieved docs."""
        if not search_results:
            return "I couldn't find relevant information in the knowledge base for your query."

        # Look for the most relevant document
        best_result = search_results[0]
        content = (best_result.chunk.content or "").strip()

        # Try to extract meaningful resolution content
        resolution_content = self._extract_resolution_content(content)

        if resolution_content:
            return f"Based on the knowledge base, here's how to resolve your issue:\n\n{resolution_content}"
        else:
            # Fallback: provide a summary of what was found
            return f"I found information related to your query in the knowledge base. The document discusses: {content[:300]}... \n\nFor more specific guidance, please contact technical support."

    def _extract_resolution_content(self, content: str) -> str:
        """Extract meaningful resolution content from a document."""
        # Split content into lines for analysis
        lines = content.split('\n')

        # Look for resolution patterns (avoid hardcoded "Answer:" lines)
        resolution_patterns = [
            r'resolution:?\s*(.+)',
            r'solution:?\s*(.+)',
            r'fix:?\s*(.+)',
            r'steps?:?\s*(.+)',
            r'to resolve:?\s*(.+)',
            r'troubleshooting:?\s*(.+)'
        ]

        import re
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Answer:') or line.startswith('Subject:'):
                continue  # Skip hardcoded answer lines and subjects

            for pattern in resolution_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    resolution = match.group(1).strip()
                    if len(resolution) > 20:  # Ensure it's substantial content
                        return resolution

        # If no explicit resolution found, look for procedural content
        procedural_lines = []
        for line in lines:
            line = line.strip()
            if (line and
                not line.startswith('Answer:') and
                not line.startswith('Subject:') and
                not line.startswith('Body:') and
                len(line) > 30):
                procedural_lines.append(line)
                if len(procedural_lines) >= 3:
                    break

        if procedural_lines:
            return '\n'.join(procedural_lines)

        return ""

    def _create_error_response(self, error_message: str, original_message: Message) -> Message:
        """Creates a standardized error message."""
        return Message(
            type=MessageType.RESPONSE,
            content=error_message,
            sender=self.agent_id,
            recipient=original_message.sender,
            metadata={"original_message_id": original_message.id}
        )

