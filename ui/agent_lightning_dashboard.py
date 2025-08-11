"""
Agent Lightning Training Dashboard.
Streamlit interface for monitoring RL training and emergent communication.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, Any, List

# Import Agent Lightning components
from rl.agent_lightning.enhanced_coordinator import EnhancedAgentCoordinator
from rl.agent_lightning.agentops_integration import get_monitor
from rl.agent_lightning.reinforcement_learning import get_rl_coordinator
from rl.agent_lightning.agent_tools import get_tool_coordinator
from rl.agent_lightning.reward_system import get_reward_system

def render_agent_lightning_dashboard():
    """Render the Agent Lightning training dashboard."""
    st.markdown("Monitor reinforcement learning progress and emergent communication patterns")
    
    # Get system components
    monitor = get_monitor()
    rl_coordinator = get_rl_coordinator()
    tool_coordinator = get_tool_coordinator()
    reward_system = get_reward_system()
    
    # Sidebar controls
    st.sidebar.header("Training Controls")
    
    # Training mode toggle
    if 'enhanced_coordinator' in st.session_state:
        coordinator = st.session_state.enhanced_coordinator
        
        current_training_mode = getattr(coordinator, 'training_mode', False)
        training_mode = st.sidebar.checkbox("Enable Training Mode", value=current_training_mode)
        
        if training_mode != current_training_mode:
            if training_mode:
                coordinator.enable_training_mode()
                st.sidebar.success("Training mode enabled!")
            else:
                coordinator.disable_training_mode()
                st.sidebar.info("Training mode disabled")
    
    # Export controls
    st.sidebar.header("Data Export")
    if st.sidebar.button("Export Training Data"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"training_data_{timestamp}.json"
        
        if 'enhanced_coordinator' in st.session_state:
            st.session_state.enhanced_coordinator.export_training_data(filepath)
            st.sidebar.success(f"Data exported to {filepath}")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 RL Performance", 
        "💬 Communication Patterns", 
        "🔧 Tool Usage", 
        "🎯 Reward Analysis",
        "📊 System Overview"
    ])
    
    with tab1:
        render_rl_performance_tab(rl_coordinator)
    
    with tab2:
        render_communication_patterns_tab(monitor)
    
    with tab3:
        render_tool_usage_tab(tool_coordinator)
    
    with tab4:
        render_reward_analysis_tab(reward_system)
    
    with tab5:
        render_system_overview_tab(monitor, rl_coordinator, tool_coordinator, reward_system)

def render_rl_performance_tab(rl_coordinator):
    """Render RL performance monitoring tab."""
    st.header("🧠 Reinforcement Learning Performance")
    
    # Get RL statistics
    rl_stats = rl_coordinator.get_training_statistics()
    
    # Training overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Episodes", rl_stats.get("total_episodes", 0))
    
    with col2:
        training_active = rl_stats.get("training_active", False)
        st.metric("Training Status", "Active" if training_active else "Inactive")
    
    with col3:
        agent_count = len(rl_stats.get("agent_statistics", {}))
        st.metric("RL Agents", agent_count)
    
    with col4:
        recent_perf = rl_stats.get("recent_performance", [])
        avg_reward = np.mean([ep.get("agent_performance", {}).get("communication_agent", {}).get("average_reward", 0) 
                             for ep in recent_perf[-10:]]) if recent_perf else 0.0
        st.metric("Avg Reward (Recent)", f"{avg_reward:.3f}")
    
    # Agent performance comparison
    st.subheader("Agent Performance Comparison")
    
    agent_stats = rl_stats.get("agent_statistics", {})
    if agent_stats:
        # Create performance DataFrame
        perf_data = []
        for agent_id, stats in agent_stats.items():
            perf_data.append({
                "Agent": agent_id.replace("_", " ").title(),
                "Success Rate": stats.get("success_rate", 0),
                "Average Reward": stats.get("average_reward", 0),
                "Total Actions": stats.get("total_actions", 0),
                "Epsilon": stats.get("epsilon", 0)
            })
        
        df = pd.DataFrame(perf_data)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_success = px.bar(df, x="Agent", y="Success Rate", 
                               title="Success Rate by Agent",
                               color="Success Rate",
                               color_continuous_scale="viridis")
            st.plotly_chart(fig_success, use_container_width=True)
        
        with col2:
            fig_reward = px.bar(df, x="Agent", y="Average Reward",
                              title="Average Reward by Agent",
                              color="Average Reward",
                              color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_reward, use_container_width=True)
        
        # Detailed agent statistics
        st.subheader("Detailed Agent Statistics")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No RL training data available yet. Start interacting with the system to generate training data.")

def render_communication_patterns_tab(monitor):
    """Render communication patterns analysis tab."""
    st.header("💬 Emergent Communication Patterns")
    
    # Get communication statistics
    comm_patterns = monitor.get_communication_patterns()
    
    # Communication overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", comm_patterns.get("total_events", 0))
    
    with col2:
        success_rate = comm_patterns.get("success_rate", 0)
        st.metric("Success Rate", f"{success_rate:.1%}")
    
    with col3:
        avg_latency = comm_patterns.get("average_latency", 0)
        st.metric("Avg Latency", f"{avg_latency:.1f}ms")
    
    with col4:
        comm_types = len(comm_patterns.get("communication_types", {}))
        st.metric("Communication Types", comm_types)
    
    # Communication flow visualization
    st.subheader("Communication Flow")
    
    sender_dist = comm_patterns.get("sender_distribution", {})
    receiver_dist = comm_patterns.get("receiver_distribution", {})
    
    if sender_dist and receiver_dist:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sender distribution
            sender_df = pd.DataFrame(list(sender_dist.items()), columns=["Agent", "Messages Sent"])
            fig_sender = px.pie(sender_df, values="Messages Sent", names="Agent",
                              title="Messages Sent by Agent")
            st.plotly_chart(fig_sender, use_container_width=True)
        
        with col2:
            # Receiver distribution
            receiver_df = pd.DataFrame(list(receiver_dist.items()), columns=["Agent", "Messages Received"])
            fig_receiver = px.pie(receiver_df, values="Messages Received", names="Agent",
                                title="Messages Received by Agent")
            st.plotly_chart(fig_receiver, use_container_width=True)
        
        # Communication types breakdown
        st.subheader("Communication Types")
        comm_types = comm_patterns.get("communication_types", {})
        if comm_types:
            types_df = pd.DataFrame(list(comm_types.items()), columns=["Type", "Count"])
            fig_types = px.bar(types_df, x="Type", y="Count",
                             title="Communication Types Distribution")
            st.plotly_chart(fig_types, use_container_width=True)
    else:
        st.info("No communication data available yet. Agents will start communicating as they process queries.")

def render_tool_usage_tab(tool_coordinator):
    """Render tool usage analysis tab."""
    st.header("🔧 Agent Tool Usage Analysis")
    
    # Get tool statistics
    tool_stats = tool_coordinator.get_coordination_statistics()
    registry_stats = tool_stats.get("registry_stats", {})
    
    # Tool overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_tools = registry_stats.get("total_tools", 0)
        st.metric("Available Tools", total_tools)
    
    with col2:
        total_calls = tool_stats.get("total_tool_calls", 0)
        st.metric("Total Tool Calls", total_calls)
    
    with col3:
        tool_performance = registry_stats.get("tool_performance", {})
        avg_success = np.mean([perf.get("success_rate", 0) for perf in tool_performance.values()]) if tool_performance else 0
        st.metric("Avg Success Rate", f"{avg_success:.1%}")
    
    # Tool usage statistics
    st.subheader("Tool Usage Statistics")
    
    usage_stats = registry_stats.get("tool_usage_stats", {})
    if usage_stats:
        usage_df = pd.DataFrame(list(usage_stats.items()), columns=["Tool", "Usage Count"])
        fig_usage = px.bar(usage_df, x="Tool", y="Usage Count",
                          title="Tool Usage Frequency")
        st.plotly_chart(fig_usage, use_container_width=True)
        
        # Tool performance details
        st.subheader("Tool Performance Details")
        if tool_performance:
            perf_data = []
            for tool_name, perf in tool_performance.items():
                perf_data.append({
                    "Tool": tool_name,
                    "Success Rate": f"{perf.get('success_rate', 0):.1%}",
                    "Avg Processing Time": f"{perf.get('average_processing_time_ms', 0):.1f}ms",
                    "Total Calls": perf.get('call_count', 0)
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
    else:
        st.info("No tool usage data available yet. Tools will be used as agents interact.")

def render_reward_analysis_tab(reward_system):
    """Render reward analysis tab."""
    st.header("🎯 Reward System Analysis")
    
    # Get reward statistics
    reward_stats = reward_system.get_system_statistics()
    
    # Reward overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feedback_count = reward_stats.get("feedback_history_count", 0)
        st.metric("Feedback Signals", feedback_count)
    
    with col2:
        signal_count = reward_stats.get("signal_history_count", 0)
        st.metric("Total Signals", signal_count)
    
    with col3:
        smoothing_enabled = reward_stats.get("reward_smoothing", False)
        st.metric("Reward Smoothing", "Enabled" if smoothing_enabled else "Disabled")
    
    # Learning curves
    st.subheader("Learning Curves")
    
    learning_curves = reward_stats.get("learning_curves", {})
    if learning_curves:
        # Create learning curve visualization
        fig = go.Figure()
        
        for agent_id, curve_data in learning_curves.items():
            recent_avg = curve_data.get("recent_average", 0)
            overall_avg = curve_data.get("overall_average", 0)
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[overall_avg, recent_avg],
                mode='lines+markers',
                name=agent_id.replace("_", " ").title(),
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Learning Progress: Overall vs Recent Performance",
            xaxis_title="Time Period",
            yaxis_title="Average Reward",
            xaxis=dict(tickvals=[0, 1], ticktext=["Overall", "Recent"])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Learning curve details
        st.subheader("Learning Curve Details")
        curve_df = pd.DataFrame([
            {
                "Agent": agent_id.replace("_", " ").title(),
                "Data Points": curve_data.get("data_points", 0),
                "Recent Average": f"{curve_data.get('recent_average', 0):.3f}",
                "Overall Average": f"{curve_data.get('overall_average', 0):.3f}"
            }
            for agent_id, curve_data in learning_curves.items()
        ])
        st.dataframe(curve_df, use_container_width=True)
    else:
        st.info("No learning curve data available yet. Learning curves will appear as agents receive rewards.")

def render_system_overview_tab(monitor, rl_coordinator, tool_coordinator, reward_system):
    """Render system overview tab."""
    st.header("📊 Agent Lightning System Overview")
    
    # System health indicators
    st.subheader("System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Monitor health
    with col1:
        monitor_active = monitor.session_active
        st.metric("Monitoring", "Active" if monitor_active else "Inactive",
                 delta="🟢" if monitor_active else "🔴")
    
    # RL health
    with col2:
        rl_stats = rl_coordinator.get_training_statistics()
        rl_agents = len(rl_stats.get("agent_statistics", {}))
        st.metric("RL Agents", rl_agents, delta="🟢" if rl_agents > 0 else "🔴")
    
    # Tool health
    with col3:
        tool_stats = tool_coordinator.get_coordination_statistics()
        available_tools = tool_stats.get("registry_stats", {}).get("total_tools", 0)
        st.metric("Available Tools", available_tools, delta="🟢" if available_tools > 0 else "🔴")
    
    # Reward system health
    with col4:
        reward_stats = reward_system.get_system_statistics()
        signal_count = reward_stats.get("signal_history_count", 0)
        st.metric("Reward Signals", signal_count, delta="🟢" if signal_count > 0 else "🔴")
    
    # Recent activity timeline
    st.subheader("Recent Activity")
    
    # Create mock timeline data (in real implementation, this would come from actual logs)
    timeline_data = []
    
    # Add RL episodes
    recent_episodes = rl_stats.get("recent_performance", [])
    for episode in recent_episodes[-5:]:
        timeline_data.append({
            "Time": episode.get("timestamp", ""),
            "Event": f"Training Episode {episode.get('episode', 'N/A')}",
            "Type": "RL Training",
            "Status": episode.get("outcome", "unknown")
        })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
    else:
        st.info("No recent activity to display. Activity will appear as the system processes queries.")
    
    # Configuration summary
    st.subheader("Configuration Summary")
    
    config_data = {
        "Component": ["Monitoring", "RL Training", "Tool Coordination", "Reward System"],
        "Status": [
            "Initialized" if monitor else "Not Available",
            f"{len(rl_coordinator.rl_agents)} agents registered" if rl_coordinator else "Not Available",
            f"{tool_coordinator.registry.get_registry_statistics().get('total_tools', 0)} tools" if tool_coordinator else "Not Available",
            "Active" if reward_system else "Not Available"
        ],
        "Configuration": [
            f"Session: {'Active' if monitor.session_active else 'Inactive'}",
            f"Training: {'Active' if rl_stats.get('training_active', False) else 'Inactive'}",
            f"Auto-registration: Enabled",
            f"Smoothing: {'Enabled' if reward_stats.get('reward_smoothing', False) else 'Disabled'}"
        ]
    }
    
    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)
    
    # Performance recommendations
    st.subheader("Performance Recommendations")
    
    recommendations = []
    
    # Check RL performance
    for agent_id, stats in rl_stats.get("agent_statistics", {}).items():
        success_rate = stats.get("success_rate", 0)
        if success_rate < 0.6:
            recommendations.append(f"🔄 {agent_id}: Low success rate ({success_rate:.1%}) - consider adjusting learning parameters")
        
        total_actions = stats.get("total_actions", 0)
        if total_actions < 50:
            recommendations.append(f"📈 {agent_id}: Needs more training data ({total_actions} actions) - increase interaction volume")
    
    # Check communication patterns
    comm_patterns = monitor.get_communication_patterns()
    success_rate = comm_patterns.get("success_rate", 0)
    if success_rate < 0.8:
        recommendations.append(f"💬 Communication success rate is low ({success_rate:.1%}) - review agent communication protocols")
    
    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.success("🎉 All systems performing well! No immediate recommendations.")

def render_tool_usage_tab(tool_coordinator):
    """Render tool usage analysis tab."""
    st.header("🔧 Agent Tool Usage Analysis")

    # Get tool statistics
    tool_stats = tool_coordinator.get_coordination_statistics()
    registry_stats = tool_stats.get("registry_stats", {})

    # Tool overview
    col1, col2, col3 = st.columns(3)

    with col1:
        total_tools = registry_stats.get("total_tools", 0)
        st.metric("Available Tools", total_tools)

    with col2:
        total_calls = tool_stats.get("total_tool_calls", 0)
        st.metric("Total Tool Calls", total_calls)

    with col3:
        tool_performance = registry_stats.get("tool_performance", {})
        avg_success = np.mean([perf.get("success_rate", 0) for perf in tool_performance.values()]) if tool_performance else 0
        st.metric("Avg Success Rate", f"{avg_success:.1%}")

    # Tool efficiency matrix
    st.subheader("Tool Efficiency Matrix")

    efficiency_matrix = tool_stats.get("tool_efficiency_matrix", {})
    if efficiency_matrix:
        # Convert to DataFrame for visualization
        matrix_data = []
        for tool_name, agent_scores in efficiency_matrix.items():
            for agent_id, score in agent_scores.items():
                matrix_data.append({
                    "Tool": tool_name,
                    "Agent": agent_id.replace("_", " ").title(),
                    "Efficiency": score
                })

        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
            fig_matrix = px.scatter(matrix_df, x="Tool", y="Agent", size="Efficiency", color="Efficiency",
                                  title="Tool-Agent Efficiency Matrix",
                                  color_continuous_scale="viridis")
            st.plotly_chart(fig_matrix, use_container_width=True)
    else:
        st.info("No tool efficiency data available yet. Efficiency metrics will appear as agents use tools.")

def render_reward_analysis_tab(reward_system):
    """Render reward analysis tab."""
    st.header("🎯 Reward System Analysis")

    # Get reward statistics
    reward_stats = reward_system.get_system_statistics()

    # Reward overview
    col1, col2, col3 = st.columns(3)

    with col1:
        feedback_count = reward_stats.get("feedback_history_count", 0)
        st.metric("Feedback Signals", feedback_count)

    with col2:
        signal_count = reward_stats.get("signal_history_count", 0)
        st.metric("Total Signals", signal_count)

    with col3:
        smoothing_enabled = reward_stats.get("reward_smoothing", False)
        st.metric("Reward Smoothing", "Enabled" if smoothing_enabled else "Disabled")

    # Reward trends
    st.subheader("Reward Trends by Agent")

    learning_curves = reward_stats.get("learning_curves", {})
    if learning_curves:
        trend_data = []
        for agent_id, curve_info in learning_curves.items():
            trend_data.append({
                "Agent": agent_id.replace("_", " ").title(),
                "Recent Performance": curve_info.get("recent_average", 0),
                "Overall Performance": curve_info.get("overall_average", 0),
                "Data Points": curve_info.get("data_points", 0)
            })

        trend_df = pd.DataFrame(trend_data)

        # Performance comparison chart
        fig_trends = go.Figure()

        fig_trends.add_trace(go.Bar(
            name="Recent Performance",
            x=trend_df["Agent"],
            y=trend_df["Recent Performance"],
            marker_color="lightblue"
        ))

        fig_trends.add_trace(go.Bar(
            name="Overall Performance",
            x=trend_df["Agent"],
            y=trend_df["Overall Performance"],
            marker_color="darkblue"
        ))

        fig_trends.update_layout(
            title="Performance Comparison: Recent vs Overall",
            xaxis_title="Agent",
            yaxis_title="Average Reward",
            barmode="group"
        )

        st.plotly_chart(fig_trends, use_container_width=True)

        # Detailed trends table
        st.dataframe(trend_df, use_container_width=True)
    else:
        st.info("No reward trend data available yet. Trends will appear as agents receive feedback.")

def render_training_controls():
    """Render training control panel."""
    st.sidebar.header("🎛️ Training Controls")
    
    # Manual training episode
    if st.sidebar.button("Start Training Episode"):
        if 'enhanced_coordinator' in st.session_state:
            coordinator = st.session_state.enhanced_coordinator
            coordinator.rl_coordinator.start_training_episode()
            st.sidebar.success("Training episode started!")
    
    # Reward system configuration
    st.sidebar.subheader("Reward Configuration")
    
    reward_smoothing = st.sidebar.checkbox("Enable Reward Smoothing", value=True)
    adaptive_shaping = st.sidebar.checkbox("Enable Adaptive Shaping", value=True)
    
    if st.sidebar.button("Apply Reward Settings"):
        reward_system = get_reward_system()
        reward_system.reward_smoothing = reward_smoothing
        reward_system.adaptive_shaping_enabled = adaptive_shaping
        st.sidebar.success("Reward settings updated!")
    
    # Model persistence
    st.sidebar.subheader("Model Management")
    
    if st.sidebar.button("Save RL Models"):
        rl_coordinator = get_rl_coordinator()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"rl_models_{timestamp}"
        rl_coordinator.save_all_models(model_dir)
        st.sidebar.success(f"Models saved to {model_dir}")
    
    # Load models
    model_dir = st.sidebar.text_input("Model Directory to Load")
    if st.sidebar.button("Load RL Models") and model_dir:
        try:
            rl_coordinator = get_rl_coordinator()
            rl_coordinator.load_all_models(model_dir)
            st.sidebar.success(f"Models loaded from {model_dir}")
        except Exception as e:
            st.sidebar.error(f"Failed to load models: {e}")

# Main dashboard function
def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Agent Lightning Dashboard",
        page_icon="🚀",
        layout="wide"
    )
    
    # Render training controls in sidebar
    render_training_controls()
    
    # Render main dashboard
    render_agent_lightning_dashboard()

if __name__ == "__main__":
    main()
