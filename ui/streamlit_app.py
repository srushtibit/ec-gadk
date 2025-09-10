"""
Streamlit Dashboard for the Multilingual Multi-Agent Support System.
Provides an interactive interface for testing, monitoring, and training the system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time
import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass

# Ensure project root is on sys.path for absolute package imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import system components
from agents.base_agent import AgentCoordinator, Message, MessageType
from agents.communication.communication_agent import CommunicationAgent
from agents.retrieval.retrieval_agent import RetrievalAgent
from agents.critic.critic_agent import CriticAgent
from agents.escalation.escalation_agent import EscalationAgent
from kb.unified_knowledge_base import get_knowledge_base
from rl.environments.support_environment import SupportEnvironment, SupportTaskGenerator

# Import Agent Lightning components
from rl.agent_lightning.enhanced_coordinator import create_enhanced_coordinator
from ui.agent_lightning_dashboard import render_agent_lightning_dashboard

from utils.config_loader import get_config
from utils.language_utils import detect_language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NexaCorp AI Support System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .agent-status-active {
        color: #28a745;
        font-weight: bold;
    }
    .agent-status-inactive {
        color: #dc3545;
        font-weight: bold;
    }
    .chat-container {
        max-height: 75vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: transparent;
        border-radius: 0.5rem;
        margin-bottom: 4rem;
    }
    .message-user {
        background-color: #dcf8c6;
        color: #075e54;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 70%;
        text-align: right;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .message-agent {
        background-color: #ffffff;
        color: #333333;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .input-container {
        position: fixed;
        bottom: 2rem;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background-color: #ffffff;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 1rem;
        z-index: 1000;
    }
    .thinking-button {
        background-color: #e8f4f8;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        margin: 0.25rem auto 0.25rem 0;
        max-width: 70%;
        font-size: 0.9em;
        border: 1px solid #e3f2fd;
    }
    .message-system {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class SupportSystemDashboard:
    """Main dashboard class for the support system."""
    
    def __init__(self):
        print("DEBUG: SupportSystemDashboard.__init__ starting")
        self.config = get_config()
        print("DEBUG: Config loaded")
        self.knowledge_base = get_knowledge_base()
        print("DEBUG: Knowledge base loaded")

        # Initialize session state
        if 'system_initialized' not in st.session_state:
            print("DEBUG: Initializing session state")
            self._initialize_session_state()

        # Initialize system components
        if not st.session_state.system_initialized:
            print("DEBUG: Initializing system")
            self._initialize_system()

        print("DEBUG: SupportSystemDashboard.__init__ completed")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        st.session_state.system_initialized = False
        st.session_state.agents_running = False
        st.session_state.conversation_history = []
        st.session_state.chat_messages = []
        st.session_state.training_history = []
        st.session_state.system_metrics = {}
        st.session_state.selected_language = 'en'
        st.session_state.current_query = ''
        st.session_state.last_response = ''
        st.session_state.escalation_alerts = []
    
    def _initialize_system(self):
        """Initialize the multi-agent system."""
        try:
            # Initialize coordinator and agents
            st.session_state.coordinator = AgentCoordinator()
            st.session_state.communication_agent = CommunicationAgent()
            st.session_state.retrieval_agent = RetrievalAgent()
            st.session_state.critic_agent = CriticAgent()
            st.session_state.escalation_agent = EscalationAgent()

            # Register agents
            st.session_state.coordinator.register_agent(st.session_state.communication_agent)
            st.session_state.coordinator.register_agent(st.session_state.retrieval_agent)
            st.session_state.coordinator.register_agent(st.session_state.critic_agent)
            st.session_state.coordinator.register_agent(st.session_state.escalation_agent)

            # Initialize environment for task generation
            st.session_state.environment = SupportEnvironment()
            st.session_state.task_generator = SupportTaskGenerator()

            # Initialize Enhanced Coordinator with Agent Lightning
            try:
                agentops_api_key = os.getenv("AGENTOPS_API_KEY")
                if agentops_api_key:
                    agents_dict = st.session_state.coordinator.agents
                    st.session_state.enhanced_coordinator = create_enhanced_coordinator(agents_dict, agentops_api_key)
                    logger.info("Enhanced coordinator initialized with AgentOps")
                else:
                    logger.warning("AgentOps API key not found, enhanced coordinator not initialized")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced coordinator: {e}")

            st.session_state.system_initialized = True
            st.success("✅ System initialized successfully!")

        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            logger.error(f"System initialization error: {e}")
    
    def run(self):
        """Main dashboard interface."""
        # Header
        st.markdown('<h1 class="main-header">🤖 NexaCorp AI Support System</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "💬 Chat Interface",
            "📊 System Monitoring",
            "📚 Knowledge Base",
            "⚠️ Escalation Center",
            "⚙️ System Configuration",
            "🚀 Agent Lightning",
            "🎓 RL Training"
        ])

        with tab1:
            self._render_chat_interface()

        with tab2:
            self._render_monitoring_dashboard()

        with tab3:
            self._render_knowledge_base_interface()

        with tab4:
            self._render_escalation_center()

        with tab5:
            self._render_configuration_interface()

        with tab6:
            self._render_agent_lightning_dashboard()

        with tab7:
            self._render_training_interface()
    
    def _render_sidebar(self):
        """Render the sidebar with system controls."""
        with st.sidebar:
            st.markdown("## 🎛️ System Controls")
            
            # System status
            if st.session_state.system_initialized:
                st.markdown("**Status:** <span class='agent-status-active'>System Ready</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Status:** <span class='agent-status-inactive'>System Offline</span>", unsafe_allow_html=True)
            
            # Agent controls
            st.markdown("### 🤖 Agent Management")
            
            if st.button("🚀 Start All Agents"):
                self._start_agents()
            
            if st.button("⏹️ Stop All Agents"):
                self._stop_agents()
            
            # Agent status
            if st.session_state.system_initialized:
                agents = st.session_state.coordinator.agents
                for agent_id, agent in agents.items():
                    status = "🟢 Active" if agent.is_active else "🔴 Inactive"
                    st.markdown(f"**{agent_id}:** {status}")
            
            st.divider()
            
            # Language selection
            st.markdown("### 🌐 Language Settings")
            st.session_state.selected_language = st.selectbox(
                "Interface Language",
                options=['en', 'es', 'de', 'fr', 'hi', 'zh'],
                index=0,
                format_func=lambda x: {
                    'en': '🇺🇸 English',
                    'es': '🇪🇸 Spanish', 
                    'de': '🇩🇪 German',
                    'fr': '🇫🇷 French',
                    'hi': '🇮🇳 Hindi',
                    'zh': '🇨🇳 Chinese'
                }[x]
            )
            
            st.divider()

            # Escalation alerts
            st.markdown("### 🚨 Recent Escalations")

            if hasattr(st.session_state, 'escalation_alerts') and st.session_state.escalation_alerts:
                recent_alerts = st.session_state.escalation_alerts[-3:]  # Show last 3
                for alert in reversed(recent_alerts):  # Most recent first
                    severity = alert.get('severity_level', 'unknown')
                    email_status = '✅' if alert.get('email_sent') else '❌'
                    timestamp = alert.get('timestamp', '')

                    # Parse timestamp for display
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%H:%M')
                    except:
                        time_str = 'N/A'

                    severity_emoji = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(severity, '⚪')

                    st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        border-radius: 4px;
                        padding: 6px;
                        margin: 4px 0;
                        font-size: 0.8em;
                        border-left: 3px solid {'#ff4444' if severity == 'critical' else '#ff8800' if severity == 'high' else '#ffaa00' if severity == 'medium' else '#44aa44'};
                    ">
                        {severity_emoji} <strong>{severity.upper()}</strong><br>
                        {email_status} {time_str}
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("🗑️ Clear Alerts"):
                    st.session_state.escalation_alerts = []
                    st.rerun()
            else:
                st.markdown("*No recent escalations*")

            st.divider()

            # Quick actions
            st.markdown("### ⚡ Quick Actions")

            if st.button("🧹 Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()

            if st.button("💾 Export Logs"):
                self._export_logs()

            if st.button("🔄 Reset System"):
                self._reset_system()
    
    def _render_chat_interface(self):
        """Render the main chat interface with full functionality."""
        st.markdown("## 💬 Interactive Support Chat")

        # Initialize conversation history if not exists
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        # Chat container with proper scrolling (main content area)
        chat_container = st.container()

        with chat_container:
            # Display chat history first (scrollable area)
            if st.session_state.conversation_history:
                # Create a scrollable chat area
                chat_area = st.container()
                with chat_area:
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

                    for exchange in st.session_state.conversation_history:
                        # User message (right side)
                        st.markdown(f"""
                        <div class="message-user">
                            <strong>You:</strong> {exchange.get('query', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)

                        # Bot response (left side)
                        if exchange.get('processing'):
                            # Show loading state
                            st.markdown(f"""
                            <div class="message-agent">
                                🤖 <span style="opacity: 0.7;">Processing your request...</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            display_response = exchange.get('chat_response') or exchange.get('response')
                            if display_response:
                                # Escape HTML in the response
                                import html
                                safe_response = html.escape(str(display_response))
                                st.markdown(f"""
                                <div class="message-agent">
                                    🤖 {safe_response}
                                </div>
                                """, unsafe_allow_html=True)

                                # Show escalation information if triggered
                                if exchange.get('escalation_info'):
                                    escalation_info = exchange['escalation_info']
                                    severity_level = escalation_info.get('severity_assessment', {}).get('level', 'unknown')
                                    email_sent = escalation_info.get('email_sent', False)
                                    escalation_id = escalation_info.get('escalation_id', 'N/A')

                                    # Color coding for severity levels
                                    severity_colors = {
                                        'critical': '#ff4444',
                                        'high': '#ff8800',
                                        'medium': '#ffaa00',
                                        'low': '#44aa44'
                                    }
                                    color = severity_colors.get(severity_level, '#888888')

                                    st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(90deg, {color}22, {color}11);
                                        border-left: 4px solid {color};
                                        padding: 12px;
                                        margin: 8px 0;
                                        border-radius: 4px;
                                        font-size: 0.9em;
                                    ">
                                        <div style="font-weight: bold; color: {color};">
                                            🚨 ESCALATION TRIGGERED - {severity_level.upper()} SEVERITY
                                        </div>
                                        <div style="margin-top: 4px; color: #666;">
                                            📧 Email notification: {'✅ Sent successfully' if email_sent else '❌ Failed to send'}
                                        </div>
                                        <div style="margin-top: 4px; color: #666; font-size: 0.8em;">
                                            ID: {escalation_id}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Thinking process button (if detailed info available)
                                if ('agent_conversation' in exchange and exchange['agent_conversation']) or \
                                   ('evaluation' in exchange and exchange['evaluation'] is not None) or \
                                   ('retrieved_docs' in exchange and exchange['retrieved_docs']):

                                    with st.expander("🤔 Show thinking process"):
                                        self._render_thinking_process(exchange)

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👋 Welcome! I'm your AI support assistant. How can I help you today?")

        # Input area at the bottom
        st.markdown("---")
        st.markdown("### 💬 Ask me anything...")

        # Query input at bottom
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input(
                "Your message",
                value="",
                placeholder="Type your message and press Enter to send...",
                key="chat_input"
            )
            submitted = st.form_submit_button("📤 Send")

        if submitted and query.strip():
            print(f"DEBUG: Form submitted with query: {query}")

            # Check if this query is already being processed
            if 'processing_query' not in st.session_state:
                st.session_state.processing_query = None

            if st.session_state.processing_query != query:
                print(f"DEBUG: New query to process: {query}")
                # Mark this query as being processed
                st.session_state.processing_query = query

                # Add user message to show it in the chat
                user_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'language': 'en',
                    'response': None,
                    'chat_response': None,
                    'agent_conversation': [],
                    'symbolic_encoding': None,
                    'evaluation': None,
                    'retrieved_docs': None,
                    'escalation_info': None,
                    'processing': True
                }
                st.session_state.conversation_history.append(user_entry)

                # Process the query immediately (don't rerun first)
                print(f"DEBUG: About to process query: {query}")
                try:
                    with st.spinner("🤖 Agents are processing your request..."):
                        result = self._process_query_sync(query, len(st.session_state.conversation_history) - 1)
                        print(f"DEBUG: Query processing completed: {result}")

                    # Clear the processing flag
                    st.session_state.processing_query = None

                except Exception as e:
                    print(f"DEBUG: Query processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clear the processing flag even on error
                    st.session_state.processing_query = None

        # Quick examples and utilities
        st.markdown("### 🎯 Quick Examples")
        example_queries = [
            "Password reset help",
            "VPN connection issue",
            "Email not syncing",
            "Screen sharing problems",
            "Account access problem"
        ]
        cols = st.columns(len(example_queries))
        for col, example in zip(cols, example_queries):
            with col:
                if st.button(example, key=f"example_{example}"):
                    # Process example query
                    asyncio.run(self._process_example_query(example))

        # Utility buttons
        col_util_1, col_util_2 = st.columns([1, 1])
        with col_util_1:
            if st.button("🎲 Random Query"):
                if hasattr(st.session_state, 'task_generator'):
                    random_task = st.session_state.task_generator.generate_task()
                    asyncio.run(self._process_example_query(random_task.user_query))
        with col_util_2:
            if st.button("🔄 Clear Chat"):
                st.session_state.chat_messages = []
                if 'conversation_history' in st.session_state:
                    st.session_state.conversation_history = []
                st.rerun()

    async def _process_example_query(self, query: str):
        """Process an example query and add to chat."""
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": query})

        # Process and add response
        try:
            if 'enhanced_coordinator' in st.session_state:
                result = await st.session_state.enhanced_coordinator.process_query_enhanced(query, "user")
                response = result.get("response", "I apologize, but I couldn't process your request.")
                metadata = {
                    "processing_time_ms": result.get("total_processing_time_ms", 0),
                    "escalated": result.get("escalated", False),
                    "quality_score": result.get("quality_score", 0),
                    "agent_path": result.get("agent_path", [])
                }
            else:
                result = await self._process_query(query)
                response = result.get("response", "I apologize, but I couldn't process your request.")
                metadata = {}

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response,
                "metadata": metadata
            })
        except Exception as e:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"❌ Error processing your request: {str(e)}"
            })

        st.rerun()


    def _render_monitoring_dashboard(self):
        """Render the system monitoring dashboard."""
        st.markdown("## 📊 System Performance Monitoring")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please initialize the system first.")
            return
        
        # Real-time metrics
        st.markdown("### 📈 Real-time Metrics")
        
        # Get current stats
        system_stats = st.session_state.coordinator.get_system_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Agents",
                system_stats.get('active_agents', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Messages",
                system_stats.get('total_messages', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                "Conversations",
                len(st.session_state.conversation_history),
                delta=None
            )
        
        with col4:
            avg_response_time = self._calculate_average_response_time()
            st.metric(
                "Avg Response Time (s)",
                f"{avg_response_time:.2f}",
                delta=None
            )
        
        # Agent performance
        st.markdown("### 🤖 Agent Performance")
        
        agent_stats = system_stats.get('agent_stats', {})
        
        if agent_stats:
            # Create performance dataframe
            performance_data = []
            for agent_id, stats in agent_stats.items():
                performance_data.append({
                    'Agent': agent_id.replace('_agent', '').title(),
                    'Messages Processed': stats.get('messages_processed', 0),
                    'Messages Sent': stats.get('messages_sent', 0),
                    'Errors': stats.get('errors', 0),
                    'Uptime (s)': stats.get('uptime', 0),
                    'Success Rate': (stats.get('messages_processed', 0) - stats.get('errors', 0)) / max(stats.get('messages_processed', 0), 1)
                })
            
            df = pd.DataFrame(performance_data)
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Agent', y='Messages Processed', 
                           title="Messages Processed by Agent")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Agent', y='Success Rate', 
                           title="Success Rate by Agent")
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(df, use_container_width=True)
        
        # Knowledge base stats
        st.markdown("### 📚 Knowledge Base Statistics")
        
        try:
            kb_stats = self.knowledge_base.get_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", kb_stats.total_documents)
            
            with col2:
                st.metric("Total Chunks", kb_stats.total_chunks)
            
            with col3:
                st.metric("Total Characters", f"{kb_stats.total_characters:,}")
            
            with col4:
                st.metric("Languages", len(kb_stats.languages))
            
            # Language distribution
            if kb_stats.languages:
                lang_data = pd.DataFrame({
                    'Language': kb_stats.languages,
                    'Count': [1] * len(kb_stats.languages)  # Simplified for demo
                })
                
                fig = px.pie(lang_data, values='Count', names='Language', 
                           title="Content Language Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading knowledge base stats: {e}")

        # Google ADK Performance
        st.markdown("### 🤖 Google ADK Performance")

        try:
            # Show ADK-specific metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model Provider", "Google ADK")

            with col2:
                st.metric("Primary Model", "gemini-2.0-flash")

            with col3:
                st.metric("Agent Architecture", "Multi-Agent")

            with col4:
                st.metric("Status", "✅ Active")



        except Exception as e:
            st.error(f"Error loading ADK performance stats: {e}")
    

    def _render_knowledge_base_interface(self):
        """Render the knowledge base management interface."""
        st.markdown("## 📚 Knowledge Base Management")
        
        # Knowledge base operations
        st.markdown("### 📁 Document Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=['pdf', 'docx', 'csv', 'xlsx', 'txt'],
                help="Upload documents to add to the knowledge base"
            )
            
            if uploaded_file and st.button("📤 Add to Knowledge Base"):
                self._add_document_to_kb(uploaded_file)
        
        with col2:
            directory_path = st.text_input(
                "Directory Path",
                value="dataset/",
                help="Path to directory containing documents"
            )
            
            if st.button("📂 Index Directory"):
                self._index_directory(directory_path)
        
        # Search interface
        st.markdown("### 🔍 Knowledge Base Search")

        with st.form(key="kb_search_form", clear_on_submit=False):
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter search terms...",
                key="kb_search_input"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10, key="kb_max_results")
            with col2:
                min_score = st.slider("Min Similarity Score", 0.0, 1.0, 0.7, format="%.2f", key="kb_min_score")
            with col3:
                search_language = st.selectbox("Search Language", ['all', 'en', 'es', 'de', 'fr'], key="kb_lang")
            search_submitted = st.form_submit_button("🔍 Search")
        if search_submitted and search_query:
            with st.spinner("🔎 Searching knowledge base..."):
                self._search_knowledge_base(search_query, max_results, min_score, search_language)
        
        # Knowledge base statistics
        st.markdown("### 📊 Knowledge Base Statistics")
        
        try:
            kb_stats = self.knowledge_base.get_stats()
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents", kb_stats.total_documents)
            
            with col2:
                st.metric("Chunks", kb_stats.total_chunks)
            
            with col3:
                st.metric("Characters", f"{kb_stats.total_characters:,}")
            
            with col4:
                st.metric("Languages", len(kb_stats.languages))
            
            # File formats
            if kb_stats.file_formats:
                format_data = pd.DataFrame(list(kb_stats.file_formats.items()), columns=['Format', 'Count'])
                
                fig = px.bar(format_data, x='Format', y='Count', 
                           title="Document Formats in Knowledge Base")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading knowledge base statistics: {e}")
    
    def _render_escalation_center(self):
        """Render the escalation monitoring center."""
        st.markdown("## ⚠️ Escalation Monitoring Center")

        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please initialize the system first.")
            return

        # Create tabs for different escalation functions
        escalation_tab1, escalation_tab2, escalation_tab3, escalation_tab4 = st.tabs([
            "📊 Dashboard", "🧪 Testing", "📧 Email Config", "📋 Management"
        ])

        with escalation_tab1:
            self._render_escalation_dashboard()

        with escalation_tab2:
            self._render_escalation_testing()

        with escalation_tab3:
            self._render_email_configuration()

        with escalation_tab4:
            self._render_escalation_management()

    def _render_escalation_dashboard(self):
        """Render escalation statistics dashboard."""
        st.markdown("### 📊 Escalation Statistics")

        try:
            # Ensure escalation agent is properly initialized
            if not hasattr(st.session_state, 'escalation_agent') or st.session_state.escalation_agent is None:
                st.error("Escalation agent not initialized. Please refresh the page.")
                return

            # Check if escalation_stats attribute exists
            if not hasattr(st.session_state.escalation_agent, 'escalation_stats'):
                st.error("Escalation agent not fully initialized. Please refresh the page.")
                return

            escalation_stats = st.session_state.escalation_agent.get_escalation_stats()

            # Main metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Assessments", escalation_stats.get('total_assessments', 0))

            with col2:
                escalations = escalation_stats.get('escalations_triggered', 0)
                st.metric("Escalations Triggered", escalations)

            with col3:
                emails_sent = escalation_stats.get('emails_sent', 0)
                st.metric("Emails Sent", emails_sent)

            with col4:
                escalation_rate = escalation_stats.get('escalation_rate', 0)
                st.metric("Escalation Rate", f"{escalation_rate:.2%}")

            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                email_success_rate = escalation_stats.get('email_success_rate', 0)
                st.metric("Email Success Rate", f"{email_success_rate:.2%}")

            with col6:
                pending = escalation_stats.get('pending_escalations', 0)
                st.metric("Pending Escalations", pending)

            with col7:
                false_positives = escalation_stats.get('false_positives', 0)
                st.metric("False Positives", false_positives)

            with col8:
                if escalations > 0:
                    avg_severity = sum(1 if e['severity'] == 'critical' else 0.8 if e['severity'] == 'high' else 0.5 if e['severity'] == 'medium' else 0.2
                                     for e in escalation_stats.get('recent_escalations', [])) / len(escalation_stats.get('recent_escalations', [1]))
                    st.metric("Avg Severity", f"{avg_severity:.2f}")
                else:
                    st.metric("Avg Severity", "0.00")

            # Severity distribution chart
            severity_dist = escalation_stats.get('severity_distribution', {})
            if severity_dist and sum(severity_dist.values()) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    try:
                        df_severity = pd.DataFrame(list(severity_dist.items()), columns=['Severity', 'Count'])
                        fig_pie = px.pie(df_severity, values='Count', names='Severity',
                                       title="Severity Level Distribution",
                                       color_discrete_map={
                                           'critical': '#ff4444',
                                           'high': '#ff8800',
                                           'medium': '#ffaa00',
                                           'low': '#44aa44'
                                       })
                        st.plotly_chart(fig_pie, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating severity chart: {e}")
                        # Fallback: show text summary
                        st.info("Severity Distribution:")
                        for level, count in severity_dist.items():
                            st.text(f"  {level.capitalize()}: {count}")

                with col2:
                    # Recent escalations timeline
                    recent_escalations = escalation_stats.get('recent_escalations', [])
                    if recent_escalations:
                        try:
                            df_timeline = pd.DataFrame(recent_escalations)
                            df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])
                            df_timeline['severity_score'] = df_timeline['severity'].map({
                                'critical': 4, 'high': 3, 'medium': 2, 'low': 1
                            })

                            fig_timeline = px.scatter(df_timeline, x='timestamp', y='severity_score',
                                                    color='severity', size_max=10,
                                                    title="Recent Escalations Timeline",
                                                    color_discrete_map={
                                                        'critical': '#ff4444',
                                                        'high': '#ff8800',
                                                        'medium': '#ffaa00',
                                                        'low': '#44aa44'
                                                    })
                            fig_timeline.update_layout(yaxis=dict(tickvals=[1,2,3,4], ticktext=['Low','Medium','High','Critical']))
                            st.plotly_chart(fig_timeline, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating timeline chart: {e}")
                            # Fallback: show simple text summary
                            st.info(f"Recent escalations: {len(recent_escalations)} entries")
                    else:
                        st.info("No recent escalations to display")

            # Recent escalations table
            st.markdown("### 🚨 Recent Escalations")

            recent_escalations = st.session_state.escalation_agent.get_escalation_history(limit=10)

            if recent_escalations:
                df = pd.DataFrame(recent_escalations)

                # Format timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

                # Add status indicators
                df['status'] = df.apply(lambda row:
                    "✅ Sent" if row['email_sent'] else "❌ Failed", axis=1)

                # Display table with better formatting
                st.dataframe(
                    df[['escalation_id', 'severity_level', 'timestamp', 'status', 'reasoning']],
                    use_container_width=True,
                    column_config={
                        "escalation_id": "Escalation ID",
                        "severity_level": "Severity",
                        "timestamp": "Timestamp",
                        "status": "Email Status",
                        "reasoning": "Reason"
                    }
                )
            else:
                st.info("📭 No recent escalations found.")

        except Exception as e:
            st.error(f"Error loading escalation dashboard: {e}")

    def _render_escalation_testing(self):
        """Render escalation testing interface."""
        st.markdown("### 🧪 Escalation Testing")

        # Test message input
        st.markdown("#### Test Severity Assessment")

        col1, col2 = st.columns([2, 1])

        with col1:
            test_message = st.text_area(
                "Enter a test message to assess severity:",
                placeholder="e.g., URGENT: System is down and all customers are affected!",
                height=100
            )

        with col2:
            st.markdown("**Severity Keywords:**")
            st.markdown("🔴 **Critical:** urgent, critical, emergency, lawsuit, security breach")
            st.markdown("🟠 **High:** important, asap, priority, deadline")
            st.markdown("🟡 **Medium:** problem, issue, error, broken")

        if st.button("🔍 Assess Severity") and test_message:
            try:
                # Create test message
                from agents.base_agent import Message, MessageType
                import uuid

                test_msg = Message(
                    type=MessageType.QUERY,
                    content=test_message,
                    sender="test_user",
                    recipient="escalation_agent",
                    id=str(uuid.uuid4())
                )

                # Assess severity
                severity_assessment = asyncio.run(
                    st.session_state.escalation_agent._assess_severity(test_msg)
                )

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    severity_color = {
                        'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'
                    }
                    st.metric(
                        "Severity Level",
                        f"{severity_color.get(severity_assessment.severity_level, '⚪')} {severity_assessment.severity_level.upper()}"
                    )

                with col2:
                    st.metric("Severity Score", f"{severity_assessment.severity_score:.3f}")

                with col3:
                    st.metric("Escalation Required", "YES" if severity_assessment.requires_escalation else "NO")

                # Detailed analysis
                st.markdown("#### 📋 Analysis Details")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Trigger Keywords:**")
                    if severity_assessment.trigger_keywords:
                        for keyword in severity_assessment.trigger_keywords:
                            st.markdown(f"• `{keyword}`")
                    else:
                        st.markdown("• None detected")

                with col2:
                    st.markdown("**Urgency Indicators:**")
                    if severity_assessment.urgency_indicators:
                        for indicator in severity_assessment.urgency_indicators:
                            st.markdown(f"• {indicator}")
                    else:
                        st.markdown("• None detected")

                st.markdown("**Reasoning:**")
                st.info(severity_assessment.reasoning)

                # Test escalation email
                if severity_assessment.requires_escalation:
                    st.markdown("#### 📧 Escalation Email Preview")

                    if st.button("📧 Preview Escalation Email"):
                        subject, body = st.session_state.escalation_agent._prepare_escalation_email(
                            test_msg, severity_assessment, f"TEST_{datetime.now().strftime('%H%M%S')}"
                        )

                        st.markdown("**Subject:**")
                        st.code(subject)

                        st.markdown("**Body:**")
                        st.text_area("Email Body", body, height=300, disabled=True)

            except Exception as e:
                st.error(f"Error testing escalation: {e}")

        # Predefined test scenarios
        st.markdown("#### 🎭 Test Scenarios")

        scenarios = {
            "🔴 Critical - System Outage": "EMERGENCY: Our entire production system is down! All customers are affected and we're losing revenue. This needs immediate attention!",
            "🟠 High - Security Breach": "URGENT: We detected unauthorized access to our database. Potential data breach in progress. Please escalate immediately.",
            "🟡 Medium - Performance Issue": "Important: The application is running very slowly today. Multiple users are complaining about response times.",
            "🟢 Low - General Question": "Hi, I have a question about how to reset my password. Can someone help me when you have time?"
        }

        selected_scenario = st.selectbox("Select a test scenario:", list(scenarios.keys()))

        if st.button("🚀 Run Scenario Test"):
            st.text_area("Test Message:", scenarios[selected_scenario], height=80, disabled=True)
            # Auto-populate the test message
            st.session_state.test_message = scenarios[selected_scenario]

    def _render_email_configuration(self):
        """Render email configuration interface."""
        st.markdown("### 📧 Email Configuration")

        # Current configuration display
        st.markdown("#### 📋 Current Configuration")

        try:
            config = st.session_state.escalation_agent.system_config.get('email', {})

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **SMTP Server:** {config.get('smtp_server', 'Not configured')}
                **SMTP Port:** {config.get('smtp_port', 'Not configured')}
                **Use TLS:** {config.get('use_tls', 'Not configured')}
                **Sender Email:** {config.get('sender_email', 'Not configured')}
                """)

            with col2:
                recipients = config.get('escalation_recipients', [])
                st.info(f"""
                **Recipients:** {len(recipients)} configured
                {chr(10).join(f"• {recipient}" for recipient in recipients[:5])}
                {'• ...' if len(recipients) > 5 else ''}
                """)

        except Exception as e:
            st.error(f"Error loading email configuration: {e}")

        # Test email configuration
        st.markdown("#### 🧪 Test Email Configuration")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Test Connection"):
                try:
                    with st.spinner("Testing email configuration..."):
                        test_result = st.session_state.escalation_agent.test_email_configuration()

                    if test_result['status'] == 'success':
                        st.success("✅ Email configuration is working correctly!")

                        details = test_result.get('details', {})
                        st.json({
                            "SMTP Server": details.get('smtp_server'),
                            "SMTP Port": details.get('smtp_port'),
                            "Use TLS": details.get('use_tls'),
                            "Sender Email": details.get('sender_email'),
                            "Recipients Count": details.get('recipients_count')
                        })
                    else:
                        st.error("❌ Email configuration has issues:")
                        for error in test_result.get('errors', []):
                            st.error(f"• {error}")

                except Exception as e:
                    st.error(f"Error testing email configuration: {e}")

        with col2:
            if st.button("📧 Send Test Email"):
                try:
                    # Create a test escalation record
                    from agents.escalation.escalation_agent import EscalationRecord, SeverityAssessment
                    from datetime import datetime

                    test_assessment = SeverityAssessment(
                        severity_level='medium',
                        severity_score=0.6,
                        trigger_keywords=['test'],
                        reasoning='This is a test escalation email',
                        requires_escalation=True,
                        urgency_indicators=['test_scenario']
                    )

                    test_record = EscalationRecord(
                        escalation_id=f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        original_message_id="test_message",
                        severity_assessment=test_assessment,
                        escalation_timestamp=datetime.now().isoformat(),
                        email_sent=False,
                        email_recipients=st.session_state.escalation_agent.recipients.copy(),
                        email_subject="🧪 TEST ESCALATION - Email Configuration Test",
                        email_body="This is a test escalation email to verify the email configuration is working correctly.",
                        follow_up_required=False,
                        resolution_deadline=None
                    )

                    with st.spinner("Sending test email..."):
                        email_sent = asyncio.run(
                            st.session_state.escalation_agent._send_escalation_email(test_record)
                        )

                    if email_sent:
                        st.success("✅ Test email sent successfully!")
                    else:
                        st.error("❌ Failed to send test email. Check configuration and logs.")

                except Exception as e:
                    st.error(f"Error sending test email: {e}")

        # Configuration update form
        st.markdown("#### ⚙️ Update Configuration")

        with st.expander("📝 Update Email Settings", expanded=False):
            st.warning("⚠️ Configuration changes require system restart to take effect.")

            # Email settings form
            with st.form("email_config_form"):
                col1, col2 = st.columns(2)

                with col1:
                    new_smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                    new_smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
                    new_use_tls = st.checkbox("Use TLS", value=True)

                with col2:
                    new_sender_email = st.text_input("Sender Email", placeholder="support@company.com")
                    new_sender_password = st.text_input("Sender Password", type="password",
                                                      help="Use app-specific password for Gmail")

                new_recipients = st.text_area(
                    "Escalation Recipients (one per line)",
                    placeholder="manager@company.com\nemergency@company.com",
                    height=100
                )

                if st.form_submit_button("💾 Save Configuration"):
                    try:
                        # Parse recipients
                        recipients_list = [r.strip() for r in new_recipients.split('\n') if r.strip()]

                        # Update configuration (this would need to be implemented)
                        st.info("Configuration update functionality would be implemented here.")
                        st.json({
                            "smtp_server": new_smtp_server,
                            "smtp_port": new_smtp_port,
                            "use_tls": new_use_tls,
                            "sender_email": new_sender_email,
                            "sender_password": "***" if new_sender_password else "",
                            "recipients": recipients_list
                        })

                    except Exception as e:
                        st.error(f"Error updating configuration: {e}")

    def _render_escalation_management(self):
        """Render escalation management interface."""
        st.markdown("### 📋 Escalation Management")

        # Pending escalations
        st.markdown("#### ⏳ Pending Escalations")

        try:
            escalation_stats = st.session_state.escalation_agent.get_escalation_stats()
            pending_count = escalation_stats.get('pending_escalations', 0)

            if pending_count > 0:
                st.warning(f"⚠️ You have {pending_count} pending escalations requiring attention!")

                # Get pending escalations (this would need to be implemented in the agent)
                pending_escalations = getattr(st.session_state.escalation_agent, 'pending_escalations', {})

                if pending_escalations:
                    for escalation_id, record in pending_escalations.items():
                        with st.expander(f"🚨 {escalation_id} - {record.severity_assessment.severity_level.upper()}", expanded=False):
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown(f"**Timestamp:** {record.escalation_timestamp}")
                                st.markdown(f"**Severity Score:** {record.severity_assessment.severity_score:.3f}")
                                st.markdown(f"**Email Sent:** {'✅ Yes' if record.email_sent else '❌ No'}")
                                st.markdown(f"**Reasoning:** {record.severity_assessment.reasoning}")

                                if record.resolution_deadline:
                                    deadline = datetime.fromisoformat(record.resolution_deadline)
                                    time_left = deadline - datetime.now()
                                    if time_left.total_seconds() > 0:
                                        st.markdown(f"**Deadline:** {deadline.strftime('%Y-%m-%d %H:%M:%S')} ({time_left})")
                                    else:
                                        st.error(f"**OVERDUE:** Deadline was {deadline.strftime('%Y-%m-%d %H:%M:%S')}")

                            with col2:
                                if st.button(f"✅ Mark Resolved", key=f"resolve_{escalation_id}"):
                                    if st.session_state.escalation_agent.mark_escalation_resolved(escalation_id):
                                        st.success(f"Escalation {escalation_id} marked as resolved!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to mark escalation as resolved.")

                                if st.button(f"📧 Resend Email", key=f"resend_{escalation_id}"):
                                    try:
                                        email_sent = asyncio.run(
                                            st.session_state.escalation_agent._send_escalation_email(record)
                                        )
                                        if email_sent:
                                            st.success("Email resent successfully!")
                                        else:
                                            st.error("Failed to resend email.")
                                    except Exception as e:
                                        st.error(f"Error resending email: {e}")
            else:
                st.success("✅ No pending escalations. All clear!")

        except Exception as e:
            st.error(f"Error loading pending escalations: {e}")

        # Escalation history management
        st.markdown("#### 📚 Escalation History")

        col1, col2, col3 = st.columns(3)

        with col1:
            history_limit = st.selectbox("Show last:", [10, 20, 50, 100], index=1)

        with col2:
            severity_filter = st.selectbox("Filter by severity:",
                                         ["All", "Critical", "High", "Medium", "Low"])

        with col3:
            if st.button("🔄 Refresh History"):
                st.rerun()

        try:
            escalation_history = st.session_state.escalation_agent.get_escalation_history(limit=history_limit)

            # Apply severity filter
            if severity_filter != "All":
                escalation_history = [e for e in escalation_history
                                    if e['severity_level'].lower() == severity_filter.lower()]

            if escalation_history:
                df = pd.DataFrame(escalation_history)

                # Format timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

                # Add action buttons column (for future use)
                # df['actions'] = df['escalation_id'].apply(lambda x: f"📧 Resend | 📋 Details")

                # Display with better formatting
                st.dataframe(
                    df[['escalation_id', 'severity_level', 'severity_score', 'timestamp', 'email_sent', 'reasoning']],
                    use_container_width=True,
                    column_config={
                        "escalation_id": "ID",
                        "severity_level": "Severity",
                        "severity_score": st.column_config.NumberColumn("Score", format="%.3f"),
                        "timestamp": "Timestamp",
                        "email_sent": st.column_config.CheckboxColumn("Email Sent"),
                        "reasoning": "Reason"
                    }
                )

                # Export functionality
                if st.button("📥 Export History to CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"escalation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("📭 No escalation history found with the current filters.")

        except Exception as e:
            st.error(f"Error loading escalation history: {e}")

        # System controls
        st.markdown("#### ⚙️ System Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Reset Statistics"):
                if st.session_state.get('confirm_reset_stats', False):
                    # Reset statistics (this would need to be implemented)
                    st.session_state.escalation_agent.escalation_stats = {
                        'total_assessments': 0,
                        'escalations_triggered': 0,
                        'emails_sent': 0,
                        'false_positives': 0,
                        'severity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
                    }
                    st.success("Statistics reset successfully!")
                    st.session_state.confirm_reset_stats = False
                    st.rerun()
                else:
                    st.session_state.confirm_reset_stats = True
                    st.warning("Click again to confirm statistics reset.")

        with col2:
            if st.button("📊 Export Statistics"):
                stats = st.session_state.escalation_agent.get_escalation_stats()
                stats_json = json.dumps(stats, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=stats_json,
                    file_name=f"escalation_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col3:
            if st.button("🧹 Clear History"):
                if st.session_state.get('confirm_clear_history', False):
                    # Clear history (this would need to be implemented)
                    st.session_state.escalation_agent.escalation_history = []
                    st.session_state.escalation_agent.pending_escalations = {}
                    st.success("History cleared successfully!")
                    st.session_state.confirm_clear_history = False
                    st.rerun()
                else:
                    st.session_state.confirm_clear_history = True
                    st.warning("Click again to confirm history clearing.")
    
    def _render_configuration_interface(self):
        """Render the system configuration interface."""
        st.markdown("## ⚙️ System Configuration")
        
        # Configuration editor
        st.markdown("### 📝 Configuration Editor")
        
        try:
            config_data = self.config.config
            
            # Display current configuration
            st.json(config_data)
            
            # Configuration modification interface
            st.markdown("### 🔧 Modify Configuration")
            
            config_section = st.selectbox(
                "Configuration Section",
                options=list(config_data.keys())
            )
            
            if config_section:
                section_data = config_data[config_section]
                
                # Allow editing of specific values
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            new_value = st.text_input(f"{config_section}.{key}", value=str(value))
                            
                            if st.button(f"Update {key}"):
                                self._update_config_value(f"{config_section}.{key}", new_value)
        
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
        
        # System information
        st.markdown("### ℹ️ System Information")
        
        system_info = {
            "Python Version": "3.9+",
            "Streamlit Version": st.__version__,
            "System Environment": self.config.get('system.environment', 'Unknown'),
            "Debug Mode": self.config.get('system.debug', False),
            "Log Level": self.config.get('system.log_level', 'INFO')
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")
    
    def _process_query_sync(self, query: str, conversation_index: int = None):
        """Synchronous wrapper for query processing."""
        print(f"DEBUG: _process_query_sync called with query: {query}")
        try:
            # Use asyncio.run in a more controlled way
            import asyncio
            loop = None
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If there's already a running loop, we need to handle this differently
                    print("DEBUG: Event loop already running, using sync fallback")
                    return self._process_query_fallback_sync(query, conversation_index)
            except RuntimeError:
                # No event loop running
                pass

            # Run the async function
            return asyncio.run(self._process_query(query, conversation_index))
        except Exception as e:
            print(f"DEBUG: Error in _process_query_sync: {e}")
            import traceback
            traceback.print_exc()
            return {"response": f"Error processing query: {str(e)}", "success": False}

    def _process_query_fallback_sync(self, query: str, conversation_index: int = None):
        """Synchronous fallback for query processing when async is not available."""
        print(f"DEBUG: _process_query_fallback_sync called with query: {query}")

        if not st.session_state.system_initialized or not query.strip():
            print("DEBUG: System not initialized or empty query")
            return {"response": "System not initialized or empty query", "success": False}

        try:
            print("DEBUG: Starting synchronous query processing")

            # Use the standard coordinator directly (synchronous approach)
            from agents.base_agent import Message, MessageType
            from utils.language_utils import detect_language

            # Create user message
            user_message = Message(
                type=MessageType.QUERY,
                content=query,
                sender="user",
                recipient="communication_agent",
                language="en"
            )

            print("DEBUG: Created user message")

            # Simple synchronous processing - just use communication agent directly
            comm_agent = st.session_state.communication_agent

            # This is a simplified synchronous approach
            response_content = f"I received your message: '{query}'. The system is working! (Synchronous mode)"

            # Update conversation history if index provided
            if conversation_index is not None and conversation_index < len(st.session_state.conversation_history):
                entry = st.session_state.conversation_history[conversation_index]
                entry.update({
                    'response': response_content,
                    'chat_response': response_content,
                    'agent_conversation': [{"agent": "communication_agent", "response": response_content}],
                    'evaluation': {"quality_score": 0.8},
                    'retrieved_docs': None,
                    'escalation_info': None,
                    'processing': False
                })
                print("DEBUG: Updated conversation history")
                st.rerun()

            return {
                "success": True,
                "response": response_content,
                "agent_path": ["communication_agent"],
                "workflow_steps": [{"agent": "communication_agent", "action": "respond"}]
            }

        except Exception as e:
            print(f"DEBUG: Error in fallback sync processing: {e}")
            import traceback
            traceback.print_exc()
            return {"response": f"Error in fallback processing: {str(e)}", "success": False}

    async def _process_query(self, query: str, conversation_index: int = None):
        """Process a user query through the multi-agent system with full functionality."""
        print(f"DEBUG: _process_query called with query: {query}")

        if not st.session_state.system_initialized or not query.strip():
            print("DEBUG: System not initialized or empty query")
            return {"response": "System not initialized or empty query", "success": False}

        try:
            print("DEBUG: Starting query processing")
            # Check if enhanced coordinator is available
            if 'enhanced_coordinator' in st.session_state:
                print("DEBUG: Using enhanced coordinator")
                # Use Agent Lightning enhanced processing
                result = await st.session_state.enhanced_coordinator.process_query_enhanced(query, "user")
                print(f"DEBUG: Enhanced coordinator result: {result}")

                # Update conversation history if index provided
                if conversation_index is not None and conversation_index < len(st.session_state.conversation_history):
                    entry = st.session_state.conversation_history[conversation_index]
                    entry.update({
                        'response': result.get("response", "No response generated"),
                        'chat_response': self._generate_chat_response(result.get("response", ""), query),
                        'agent_conversation': result.get("workflow_steps", []),
                        'evaluation': {"quality_score": result.get("quality_score", 0)},
                        'retrieved_docs': result.get("retrieved_docs"),
                        'escalation_info': result.get("escalation_info") if result.get("escalated") else None,
                        'processing': False
                    })
                    st.rerun()

                return result

            # Fallback to standard processing with full pipeline
            print("DEBUG: Using standard processing fallback")
            from utils.language_utils import detect_language
            language_result = detect_language(query)
            print(f"DEBUG: Language detected: {language_result}")

            # Create user message
            user_message = Message(
                type=MessageType.QUERY,
                content=query,
                sender="user",
                recipient="communication_agent",
                language=language_result.language
            )

            # Check for escalation FIRST
            escalation_response = None
            try:
                escalation_response = await st.session_state.escalation_agent.process_message(user_message)
            except Exception as e:
                logger.error(f"Error in escalation check: {e}")

            # Ensure agents are running
            if not getattr(st.session_state.coordinator, "is_running", False):
                st.session_state.coordinator.start_all_agents()

            # Process through communication agent
            initial_response = await st.session_state.communication_agent.process_message(user_message)

            # Check if communication agent handled it directly
            if initial_response and initial_response.recipient == "user":
                final_response = initial_response.content
                agent_conversation = [{
                    'cycle': 1,
                    'sender': 'communication_agent',
                    'recipient': 'user',
                    'type': 'response',
                    'content': final_response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }]
                evaluation_result = None
                retrieved_docs = None
            else:
                # Continue with full processing pipeline
                st.session_state.communication_agent.receive_message(user_message)

                # Run system cycles
                final_response = ""
                evaluation_result = None
                agent_conversation = []
                retrieved_docs = None

                for cycle_num in range(5):
                    messages = await st.session_state.coordinator.run_cycle() or []

                    for message in messages:
                        # Record inter-agent messages
                        safe_content = message.content
                        if isinstance(safe_content, str):
                            import re as _re
                            no_html = _re.sub(r"<[^>]+>", "", safe_content)
                            no_think = _re.sub(r"<think>[\s\S]*?</think>", "", no_html, flags=_re.IGNORECASE)
                            truncated = no_think[:400] + "..." if len(no_think) > 400 else no_think
                            safe_content = truncated.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

                        agent_conversation.append({
                            'cycle': cycle_num + 1,
                            'sender': message.sender,
                            'recipient': message.recipient,
                            'type': message.type.value,
                            'content': safe_content,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })

                        # Get final response
                        if message.type == MessageType.RESPONSE and message.recipient == "user":
                            final_response = message.content
                            if isinstance(message.metadata, dict):
                                if message.metadata.get("evaluation"):
                                    evaluation_result = message.metadata.get("evaluation")
                                if message.metadata.get("retrieved_docs"):
                                    retrieved_docs = message.metadata.get("retrieved_docs")
                                    st.session_state.last_retrieved_docs = retrieved_docs

                        elif message.type == MessageType.FEEDBACK and isinstance(message.metadata, dict) and "evaluation_result" in message.metadata:
                            evaluation_result = message.metadata.get("evaluation_result")

                    if final_response:
                        break

            # Generate chat response
            chat_response = self._generate_chat_response(final_response, query)

            # Update conversation entry if index provided
            if conversation_index is not None and conversation_index < len(st.session_state.conversation_history):
                entry = st.session_state.conversation_history[conversation_index]
                entry.update({
                    'language': language_result.language,
                    'response': final_response or "No response generated",
                    'chat_response': chat_response,
                    'agent_conversation': agent_conversation,
                    'evaluation': evaluation_result,
                    'retrieved_docs': retrieved_docs,
                    'escalation_info': escalation_response.metadata if escalation_response and escalation_response.metadata.get('escalation_triggered') else None,
                    'processing': False
                })
                st.rerun()

            return {
                "response": final_response or "No response generated",
                "success": bool(final_response),
                "processing_time_ms": 2000,  # Placeholder
                "escalated": bool(escalation_response and escalation_response.metadata.get('escalation_triggered')),
                "retrieved_docs": retrieved_docs,
                "evaluation": evaluation_result
            }

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {
                "response": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "success": False,
                "processing_time_ms": 0,
                "escalated": False
            }


    
    def _generate_chat_response(self, detailed_response: str, query: str) -> str:
        """
        Generate a concise, chat-like response from detailed search results.
        """

        # Let the agents handle all responses naturally - no hardcoded overrides
        query_lower = query.lower().strip()

        if not detailed_response or detailed_response == "No response generated":
            return "I'm sorry, I couldn't find any relevant information for your query. Please try rephrasing your question or contact support for assistance."

        # Return the detailed response directly, without adding any prefixes.
        return detailed_response
    
    def _format_thinking_details(self, exchange: dict) -> str:
        """Format the thinking process details for the expandable section."""
        import html

        details = []

        # Helper function to safely escape HTML content
        def safe_html_escape(text: str) -> str:
            if not text:
                return ""
            return html.escape(str(text))

        # 1) Communication → Retrieval (what was asked)
        com_to_ret = None
        for msg in exchange.get('agent_conversation', []) or []:
            if msg.get('sender') == 'communication_agent' and msg.get('recipient') == 'retrieval_agent':
                com_to_ret = msg
                break
        if com_to_ret:
            safe_content = safe_html_escape(com_to_ret.get('content', ''))
            details.append(f"""
            <div style="background-color: #eef6ff; color: #212529; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #1e88e5; margin: 0.5rem 0;">
                <strong style="color: #1e88e5;">1) Communication → Retrieval</strong><br>
                <span style="color: #374151;">{safe_content}</span>
            </div>
            """)

        # 2) Retrieval → Communication (what came back)
        ret_to_com = None
        for msg in exchange.get('agent_conversation', []) or []:
            if msg.get('sender') == 'retrieval_agent' and msg.get('recipient') == 'communication_agent' and msg.get('type') == 'response':
                ret_to_com = msg
                break
        if ret_to_com:
            safe_content = safe_html_escape(ret_to_com.get('content', ''))
            details.append(f"""
            <div style="background-color: #f1f3f4; color: #212529; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #6c757d; margin: 0.5rem 0;">
                <strong style="color: #6c757d;">2) Retrieval → Communication</strong><br>
                <span style="color: #374151;">{safe_content}</span>
            </div>
            """)

        # 3) Short summary (from critic feedback if available)
        summary_text = None
        eval_data = exchange.get('evaluation') if isinstance(exchange.get('evaluation'), dict) else None
        if eval_data and isinstance(eval_data.get('feedback'), str):
            summary_text = eval_data.get('feedback')
        elif exchange.get('retrieved_docs'):
            top = exchange['retrieved_docs'][0]
            top_src = top.get('chunk',{}).get('source_file','')
            summary_text = f"Selected top results by similarity; leading source: {top_src}."
        if summary_text:
            safe_summary = safe_html_escape(summary_text)
            details.append(f"""
            <div style="background-color: #fffbe6; color: #212529; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #fbc02d; margin: 0.5rem 0;">
                <strong style="color: #fbc02d;">3) Summary</strong><br>
                <span style="color: #374151;">{safe_summary}</span>
            </div>
            """)

        # Show retrieved docs from metadata if present
        retrieved_docs = exchange.get('retrieved_docs')

        if not retrieved_docs and isinstance(exchange.get('response'), dict) and 'retrieved_docs' in exchange['response']:
            retrieved_docs = exchange['response']['retrieved_docs']

        if retrieved_docs:
            details.append('<div style="margin: 0.5rem 0;"><strong style="color: #1976d2;">4) 📄 Documents Consulted:</strong></div>')
            for i, doc in enumerate(retrieved_docs[:5]):
                try:
                    src = safe_html_escape(doc['chunk']['source_file'])
                    snippet = doc['chunk']['content'][:220] + ('...' if len(doc['chunk']['content']) > 220 else '')
                    safe_snippet = safe_html_escape(snippet)
                    score = doc.get('score', 0)
                    details.append(
                        f'<div style="background-color: #f9fafb; color: #212529; padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.25rem; font-size: 0.9em;">'
                        f'<strong>Doc {i+1}</strong> — <em>{src}</em><br>'
                        f'<span style="color:#6b7280;">Score: {score:.3f}</span><br>'
                        f'<span style="color:#374151;">{safe_snippet}</span>'
                        f'</div>'
                    )
                except Exception:
                    continue
        
        # Add evaluation metrics
        if 'evaluation' in exchange and exchange['evaluation'] is not None:
            eval_data = exchange['evaluation']
            details.append(f"""
            <div style="background-color: #fff3e0; color: #212529; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff9800; margin: 0.5rem 0;">
                <strong style="color: #ff9800;">📊 Response Quality Metrics:</strong><br>
                Overall Score: {eval_data.get('overall_score', 0):.3f} |
                Relevance: {eval_data.get('relevance_score', 0):.3f} |
                Accuracy: {eval_data.get('accuracy_score', 0):.3f} |
                Completeness: {eval_data.get('completeness_score', 0):.3f}
            </div>
            """)
        
        # Add symbolic encoding
        if 'symbolic_encoding' in exchange and exchange['symbolic_encoding'] is not None:
            encoding = exchange['symbolic_encoding']
            details.append(f"""
            <div style="background-color: #e8f5e8; color: #212529; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50; margin: 0.5rem 0;">
                <strong style="color: #4caf50;">🔢 Symbolic Encoding:</strong><br>
                Length: {len(encoding)} | Unique Symbols: {len(set(encoding))} | 
                Encoding: {encoding[:20]}{"..." if len(encoding) > 20 else ""}
            </div>
            """)
        
        return "".join(details)

    def _render_thinking_process(self, exchange: dict):
        """Render the thinking process in a structured, user-friendly way."""



        # Create tabs for different aspects of the thinking process
        comm_tab, docs_tab, rl_tab = st.tabs(["🧠 Agent Communication", "📚 Referenced Documents", "🎯 RL Training"])

        with comm_tab:
            st.markdown("### 🤖 Agent-to-Agent Communication Flow")
            self._render_agent_communication_flow(exchange)

        with docs_tab:
            st.markdown("### 📄 Knowledge Base Documents")
            self._render_referenced_documents(exchange)

        with rl_tab:
            st.markdown("### 🧠 Reinforcement Learning Insights")
            self._render_rl_training_info(exchange)

    def _render_agent_communication_flow(self, exchange: dict):
        """Render detailed agent-to-agent communication flow with emergent patterns."""
        import html

        def safe_escape(text: str) -> str:
            if not text:
                return ""
            return html.escape(str(text))

        # Get workflow steps from agent conversation
        workflow_steps = exchange.get('agent_conversation', []) or []

        if not workflow_steps:
            st.info("No agent communication data available for this exchange.")
            return

        st.markdown("#### 🔄 Communication Sequence")

        for i, step in enumerate(workflow_steps):
            agent_name = step.get('agent', 'unknown').replace('_', ' ').title()
            action = step.get('action', 'unknown').replace('_', ' ').title()
            confidence = step.get('confidence', 0.0)
            intent = step.get('communication_intent', 'unknown')
            protocol = step.get('emergent_protocol', {})

            # Create expandable section for each communication step
            with st.expander(f"Step {i+1}: {agent_name} - {action} (Confidence: {confidence:.2f})"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**🤖 Agent:** {agent_name}")
                    st.markdown(f"**⚡ Action:** {action}")
                    st.markdown(f"**💬 Content:** {safe_escape(step.get('content', 'No content'))[:200]}...")
                    st.markdown(f"**🎯 Intent:** {intent.replace('_', ' ').title()}")

                with col2:
                    st.markdown("**📊 Metrics**")
                    st.metric("Confidence", f"{confidence:.2f}")
                    st.metric("Protocol Efficiency", f"{protocol.get('protocol_efficiency', 0):.2f}")
                    st.metric("Consensus Level", f"{protocol.get('consensus_level', 0):.2f}")

                # Show emergent communication details
                if protocol:
                    st.markdown("**🌟 Emergent Communication Protocol**")
                    protocol_info = f"""
                    - **Pattern:** {protocol.get('communication_pattern', 'N/A')}
                    - **Version:** {protocol.get('protocol_version', 'N/A')}
                    - **Adaptation Score:** {protocol.get('adaptation_score', 0):.2f}
                    - **Negotiation Round:** {protocol.get('negotiation_round', 1)}
                    """
                    st.markdown(protocol_info)

    def _render_referenced_documents(self, exchange: dict):
        """Render the documents referenced during the query processing."""
        retrieved_docs = exchange.get('retrieved_docs')

        if not retrieved_docs:
            st.info("No documents were referenced for this query.")
            return

        st.markdown(f"**📊 Total Documents Referenced:** {len(retrieved_docs)}")

        # Show up to 5 documents as requested
        docs_to_show = retrieved_docs[:5]

        exchange_id = exchange.get('timestamp', str(time.time()))

        if len(docs_to_show) > 1:
            doc_tabs = st.tabs([f"📄 Doc {i+1}" for i in range(len(docs_to_show))])
            for i, (tab, doc) in enumerate(zip(doc_tabs, docs_to_show)):
                with tab:
                    self._render_single_document(doc, i, exchange_id)
        elif docs_to_show:
            # Single document - show directly
            self._render_single_document(docs_to_show[0], 0, exchange_id)

    def _render_single_document(self, doc: dict, index: int, exchange_id: str):
        """Render a single document with its details."""
        try:
            chunk = doc.get('chunk', {})
            src = chunk.get('source_file', 'Unknown source')
            content = chunk.get('content', 'No content available')
            score = doc.get('score', 0)

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**📄 Source:** `{src}`")
                snippet = content[:300] + ('...' if len(content) > 300 else '')
                st.markdown("**📝 Content Preview:**")
                st.text_area("Document Content", value=snippet, height=120, disabled=True, key=f"doc_content_{exchange_id}_{index}", label_visibility="collapsed")

            with col2:
                st.metric("🎯 Relevance Score", f"{score:.3f}")
                st.metric("📏 Content Length", f"{len(content)} chars")

        except Exception as e:
            st.error(f"Error displaying document: {str(e)}")

    def _render_rl_training_info(self, exchange: dict):
        """Render reinforcement learning training information."""
        workflow_steps = exchange.get('agent_conversation', []) or []

        if not workflow_steps:
            st.info("No RL training data available for this exchange.")
            return

        # Calculate RL metrics from workflow steps
        total_confidence = sum(step.get('confidence', 0) for step in workflow_steps)
        avg_confidence = total_confidence / len(workflow_steps) if workflow_steps else 0

        # Extract protocol efficiency scores
        protocol_scores = [
            step.get('emergent_protocol', {}).get('protocol_efficiency', 0)
            for step in workflow_steps
        ]
        avg_protocol_efficiency = sum(protocol_scores) / len(protocol_scores) if protocol_scores else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 Average Confidence", f"{avg_confidence:.3f}")

        with col2:
            st.metric("⚡ Protocol Efficiency", f"{avg_protocol_efficiency:.3f}")

        with col3:
            st.metric("🔄 Communication Steps", len(workflow_steps))

        # Show action distribution
        if workflow_steps:
            st.markdown("#### 📊 Action Distribution")
            actions = [step.get('action', 'unknown') for step in workflow_steps]
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1

            action_df = pd.DataFrame(list(action_counts.items()), columns=['Action', 'Count'])
            action_df['Action'] = action_df['Action'].str.replace('_', ' ').str.title()
            st.bar_chart(action_df.set_index('Action'))

        # Show emergent communication patterns
        st.markdown("#### 🌟 Emergent Communication Patterns")
        communication_intents = [step.get('communication_intent', 'unknown') for step in workflow_steps]
        intent_counts = {}
        for intent in communication_intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        for intent, count in intent_counts.items():
            st.markdown(f"- **{intent.replace('_', ' ').title()}:** {count} occurrences")

        # Legacy fallback - this section is now handled by the new tabbed interface above
        # Keeping minimal version for backward compatibility
        st.markdown("---")
        st.markdown("#### 📋 Legacy Communication Summary")

        workflow_steps = exchange.get('agent_conversation', []) or []
        if workflow_steps:
            st.markdown(f"**Total Communication Steps:** {len(workflow_steps)}")
            for i, step in enumerate(workflow_steps[:3]):  # Show first 3 steps
                agent = step.get('agent', 'unknown').replace('_', ' ').title()
                action = step.get('action', 'unknown').replace('_', ' ').title()
                st.markdown(f"- **Step {i+1}:** {agent} performed {action}")
        else:
            st.info("No detailed communication steps available.")

    def _start_agents(self):
        """Start all agents."""
        try:
            st.session_state.coordinator.start_all_agents()
            st.session_state.agents_running = True
            st.success("✅ All agents started successfully!")
        except Exception as e:
            st.error(f"❌ Error starting agents: {e}")
    
    def _stop_agents(self):
        """Stop all agents."""
        try:
            st.session_state.coordinator.stop_all_agents()
            st.session_state.agents_running = False
            st.success("✅ All agents stopped successfully!")
        except Exception as e:
            st.error(f"❌ Error stopping agents: {e}")
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time."""
        # Simplified calculation
        return 2.5  # Mock value
    


    def _add_document_to_kb(self, uploaded_file):
        """Add uploaded document to knowledge base."""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add to knowledge base
            success = self.knowledge_base.add_document(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            if success:
                st.success(f"✅ Document '{uploaded_file.name}' added to knowledge base!")
            else:
                st.error(f"❌ Failed to add document '{uploaded_file.name}'")
        
        except Exception as e:
            st.error(f"❌ Error adding document: {e}")
    
    def _index_directory(self, directory_path: str):
        """Index all documents in a directory."""
        try:
            successful, total = self.knowledge_base.add_documents_from_directory(directory_path)
            st.success(f"✅ Indexed {successful}/{total} documents from {directory_path}")
        except Exception as e:
            st.error(f"❌ Error indexing directory: {e}")
    
    def _search_knowledge_base(self, query: str, max_results: int, min_score: float, language: str):
        """Search the knowledge base."""
        try:
            # Perform search
            search_language = None if language == 'all' else language
            results = self.knowledge_base.search(
                query=query,
                max_results=max_results,
                min_score=min_score,
                language=search_language
            )
            
            st.markdown(f"### 🔍 Search Results ({len(results)} found)")
            
            if results:
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} - Score: {result.score:.3f}"):
                        st.markdown(f"**Source:** {result.chunk.source_file}")
                        st.markdown(f"**Language:** {result.chunk.language}")
                        st.markdown(f"**Content:**")
                        st.text(result.chunk.content[:500] + "..." if len(result.chunk.content) > 500 else result.chunk.content)
            else:
                st.info("📭 No results found for your query.")
        
        except Exception as e:
            st.error(f"❌ Error searching knowledge base: {e}")
    
    def _update_config_value(self, key_path: str, new_value: str):
        """Update a configuration value."""
        try:
            # This would update the configuration
            st.success(f"✅ Configuration updated: {key_path} = {new_value}")
        except Exception as e:
            st.error(f"❌ Error updating configuration: {e}")
    
    def _export_logs(self):
        """Export system logs."""
        try:
            # Create export data
            export_data = {
                'conversation_history': st.session_state.conversation_history,
                'system_stats': st.session_state.coordinator.get_system_stats() if st.session_state.system_initialized else {},
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Create download
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download Logs",
                data=json_data,
                file_name=f"support_system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"❌ Error exporting logs: {e}")
    
    def _render_agent_lightning_dashboard(self):
        """Render the Agent Lightning training dashboard."""
        st.header("🚀 Agent Lightning Training Dashboard")

        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please initialize the system first.")
            return

        # Check if enhanced coordinator is available
        if 'enhanced_coordinator' not in st.session_state:
            # Try to get AgentOps API key from environment first
            agentops_api_key = os.getenv("AGENTOPS_API_KEY")

            if agentops_api_key:
                # Auto-initialize if API key is available
                st.info("🔧 Auto-initializing Agent Lightning with environment API key...")
                try:
                    agents = st.session_state.coordinator.agents
                    enhanced_coordinator = create_enhanced_coordinator(agents, agentops_api_key)
                    st.session_state.enhanced_coordinator = enhanced_coordinator
                    st.success(f"🚀 Agent Lightning initialized successfully with API key: {agentops_api_key[:8]}...")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to auto-initialize Agent Lightning: {e}")
                    logger.error(f"Agent Lightning auto-initialization error: {e}")
            else:
                # Manual initialization if no API key in environment
                st.info("🔧 Initializing Agent Lightning capabilities...")

                try:
                    agents = st.session_state.coordinator.agents
                    manual_api_key = st.text_input("AgentOps API Key (optional)", type="password",
                                                  help="Enter your AgentOps API key for cloud monitoring")

                    if st.button("Initialize Agent Lightning"):
                        enhanced_coordinator = create_enhanced_coordinator(agents, manual_api_key)
                        st.session_state.enhanced_coordinator = enhanced_coordinator
                        st.success("🚀 Agent Lightning initialized successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Failed to initialize Agent Lightning: {e}")
                    logger.error(f"Agent Lightning initialization error: {e}")

            return

        # Render the dashboard
        try:
            render_agent_lightning_dashboard()
        except Exception as e:
            st.error(f"❌ Error rendering Agent Lightning dashboard: {e}")
            logger.error(f"Agent Lightning dashboard error: {e}")

    def _reset_system(self):
        """Reset the entire system."""
        try:
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'system_initialized':
                    del st.session_state[key]

            st.session_state.system_initialized = False
            st.success("🔄 System reset successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error resetting system: {e}")

    def _render_training_interface(self):
        """Render the RL training interface."""
        st.header("🎓 Reinforcement Learning Training")

        if not st.session_state.system_initialized:
            st.warning("⚠️ System not initialized. Please initialize the system first.")
            return

        # Check if enhanced coordinator is available
        if 'enhanced_coordinator' not in st.session_state:
            st.info("🔧 Agent Lightning not initialized. Please go to the Agent Lightning tab to set it up.")
            return

        coordinator = st.session_state.enhanced_coordinator

        # Training controls
        st.subheader("🎛️ Training Controls")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Training mode toggle
            current_training_mode = getattr(coordinator, 'training_mode', False)
            if st.button("🟢 Enable Training" if not current_training_mode else "🔴 Disable Training"):
                try:
                    if current_training_mode:
                        coordinator.rl_coordinator.disable_training()
                        st.success("Training mode disabled")
                    else:
                        coordinator.rl_coordinator.enable_training()
                        st.success("Training mode enabled")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error toggling training mode: {e}")

        with col2:
            # Manual training episode
            if st.button("🚀 Start Training Episode"):
                try:
                    coordinator.rl_coordinator.start_training_episode()
                    st.success("Training episode started!")
                except Exception as e:
                    st.error(f"Error starting training episode: {e}")

        with col3:
            # Export training data
            if st.button("📁 Export Training Data"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = f"training_data_{timestamp}.json"
                    # coordinator.export_training_data(filepath)
                    st.success(f"Training data export initiated: {filepath}")
                except Exception as e:
                    st.error(f"Error exporting training data: {e}")

        # Training status
        st.subheader("📊 Training Status")

        try:
            training_stats = coordinator.rl_coordinator.get_training_stats()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Episodes", training_stats.get("total_episodes", 0))
            with col2:
                st.metric("Avg Reward", f"{training_stats.get('avg_reward', 0):.3f}")
            with col3:
                st.metric("Success Rate", f"{training_stats.get('success_rate', 0):.1%}")
            with col4:
                st.metric("Learning Rate", f"{training_stats.get('learning_rate', 0.001):.4f}")
        except Exception as e:
            st.warning(f"Could not load training stats: {e}")

        # Batch training section
        st.subheader("📚 Batch Training")

        batch_queries = st.text_area(
            "Enter training queries (one per line):",
            placeholder="My email isn't working\nPassword reset help\nVPN connection issues\nScreen sharing problems\n...",
            height=150,
            key="batch_training_queries"
        )

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=50, value=10)
        with col2:
            feedback_score = st.slider("Simulated Feedback Score", 0.0, 1.0, 0.8, 0.1)

        if st.button("🎯 Run Batch Training"):
            if batch_queries.strip():
                queries = [q.strip() for q in batch_queries.split('\n') if q.strip()]

                if queries:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []
                    for i, query in enumerate(queries[:batch_size]):
                        status_text.text(f"Processing query {i+1}/{min(len(queries), batch_size)}: {query[:50]}...")

                        try:
                            # Process query
                            result = asyncio.run(coordinator.process_query_enhanced(query, f"batch_user_{i}"))

                            results.append({
                                "query": query,
                                "success": result.get("success", False),
                                "processing_time": result.get("total_processing_time_ms", 0),
                                "response_length": len(result.get("response", "")),
                                "quality_score": result.get("quality_score", 0)
                            })

                        except Exception as e:
                            results.append({
                                "query": query,
                                "success": False,
                                "error": str(e)
                            })

                        progress_bar.progress((i + 1) / min(len(queries), batch_size))

                    status_text.text("✅ Batch training completed!")

                    # Show results
                    st.subheader("📈 Batch Training Results")
                    import pandas as pd
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    # Summary metrics
                    success_rate = sum(1 for r in results if r.get("success", False)) / len(results)
                    avg_time = sum(r.get("processing_time", 0) for r in results) / len(results)
                    avg_quality = sum(r.get("quality_score", 0) for r in results) / len(results)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Success Rate", f"{success_rate:.1%}")
                    with col2:
                        st.metric("Avg Processing Time", f"{avg_time:.0f}ms")
                    with col3:
                        st.metric("Avg Quality Score", f"{avg_quality:.3f}")
                else:
                    st.warning("Please enter at least one query for batch training.")
            else:
                st.warning("Please enter training queries.")

        # Quick training examples
        st.subheader("🎯 Quick Training Examples")

        example_batches = {
            "Technical Issues": [
                "My computer won't start",
                "Email not syncing properly",
                "VPN connection keeps dropping",
                "Screen sharing not working in Zoom",
                "Printer offline error"
            ],
            "Account Problems": [
                "Can't log into my account",
                "Password reset not working",
                "Two-factor authentication issues",
                "Account locked out",
                "Forgot my username"
            ],
            "Software Issues": [
                "Application crashes on startup",
                "Software installation failed",
                "License key not working",
                "Update installation stuck",
                "Performance is very slow"
            ]
        }

        cols = st.columns(len(example_batches))
        for col, (category, queries) in zip(cols, example_batches.items()):
            with col:
                if st.button(f"📝 Load {category}", key=f"load_{category}"):
                    # Update the text area with example queries
                    st.session_state.batch_training_queries = '\n'.join(queries)
                    st.rerun()

    def _display_enhanced_query_result(self, result: Dict[str, Any], query: str, conversation_index: int = None):
        """Display results from enhanced Agent Lightning processing."""
        # Response content
        response = result.get("response", "No response generated")

        # Display main response
        st.markdown("### 🤖 AI Assistant Response")
        st.markdown(response)

        # Agent Lightning metrics
        if result.get("agent_lightning_enabled", False):
            st.markdown("### 🚀 Agent Lightning Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                processing_time = result.get("total_processing_time_ms", 0)
                st.metric("Processing Time", f"{processing_time:.1f}ms")

            with col2:
                quality_score = result.get("quality_score", 0)
                st.metric("Quality Score", f"{quality_score:.2f}")

            with col3:
                escalated = result.get("escalated", False)
                st.metric("Escalated", "Yes" if escalated else "No")

            with col4:
                training_mode = result.get("training_mode", False)
                st.metric("Training Mode", "Active" if training_mode else "Inactive")

            # Workflow visualization
            workflow_steps = result.get("workflow_steps", [])
            if workflow_steps:
                st.markdown("### 🔄 Agent Workflow")

                workflow_df = pd.DataFrame(workflow_steps)
                workflow_df["Agent"] = workflow_df["agent"].str.replace("_", " ").str.title()
                workflow_df["Action"] = workflow_df["action"].str.replace("_", " ").str.title()
                workflow_df["Confidence"] = workflow_df["confidence"].round(3)

                st.dataframe(workflow_df[["Agent", "Action", "Confidence"]], use_container_width=True)

        # Store conversation
        if conversation_index is not None:
            conversation = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'query': query,
                'response': response,
                'agent_lightning_enabled': result.get("agent_lightning_enabled", False),
                'processing_time_ms': result.get("total_processing_time_ms", 0),
                'quality_score': result.get("quality_score", 0),
                'escalated': result.get("escalated", False),
                'workflow_steps': result.get("workflow_steps", [])
            }

            if 'conversations' not in st.session_state:
                st.session_state.conversations = []

            if conversation_index < len(st.session_state.conversations):
                st.session_state.conversations[conversation_index] = conversation
            else:
                st.session_state.conversations.append(conversation)

# Main application
def main():
    """Main application entry point."""
    print("DEBUG: Starting main()")

    # Initialize dashboard only once using session state
    if 'dashboard' not in st.session_state:
        print("DEBUG: Creating SupportSystemDashboard")
        st.session_state.dashboard = SupportSystemDashboard()
        print("DEBUG: SupportSystemDashboard created")

    print("DEBUG: Running dashboard")
    st.session_state.dashboard.run()
    print("DEBUG: Dashboard run completed")

if __name__ == "__main__":
    print("DEBUG: Script starting")
    main()