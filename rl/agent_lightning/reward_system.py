"""
Advanced Reward System for Agent Lightning Framework.
Implements sophisticated reward calculation based on multiple outcome signals.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from agents.base_agent import Message, MessageType
from rl.agent_lightning.reinforcement_learning import RLAction, ActionType
from rl.agent_lightning.agentops_integration import get_monitor

logger = logging.getLogger(__name__)

class RewardSignalType(Enum):
    """Types of reward signals."""
    USER_FEEDBACK = "user_feedback"
    RESPONSE_QUALITY = "response_quality"
    PROCESSING_EFFICIENCY = "processing_efficiency"
    ESCALATION_ACCURACY = "escalation_accuracy"
    COMMUNICATION_EFFECTIVENESS = "communication_effectiveness"
    TOOL_USAGE_EFFICIENCY = "tool_usage_efficiency"

@dataclass
class RewardSignal:
    """Represents a reward signal for RL training."""
    signal_type: RewardSignalType
    value: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # Where the signal came from
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class UserFeedbackCollector:
    """Collects and processes user feedback for reward signals."""
    
    def __init__(self):
        self.feedback_history: List[Dict[str, Any]] = []
        self.satisfaction_trends: Dict[str, List[float]] = {}  # agent_id -> satisfaction scores
        
    def collect_explicit_feedback(self, 
                                user_id: str,
                                response_id: str,
                                satisfaction_score: float,
                                feedback_text: Optional[str] = None) -> RewardSignal:
        """Collect explicit user feedback (thumbs up/down, ratings)."""
        feedback_data = {
            "user_id": user_id,
            "response_id": response_id,
            "satisfaction_score": satisfaction_score,
            "feedback_text": feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback_data)
        
        # Normalize satisfaction score to reward range
        reward_value = (satisfaction_score - 0.5) * 2  # Convert 0-1 to -1 to 1
        
        return RewardSignal(
            signal_type=RewardSignalType.USER_FEEDBACK,
            value=reward_value,
            confidence=0.9,  # High confidence for explicit feedback
            source="user_explicit",
            context=feedback_data
        )
    
    def infer_implicit_feedback(self, 
                              user_behavior: Dict[str, Any]) -> Optional[RewardSignal]:
        """Infer feedback from user behavior patterns."""
        # Analyze implicit signals
        session_duration = user_behavior.get("session_duration_seconds", 0)
        follow_up_questions = user_behavior.get("follow_up_questions", 0)
        task_completion = user_behavior.get("task_completed", False)
        
        # Calculate implicit satisfaction
        satisfaction = 0.5  # Neutral baseline
        
        # Longer sessions might indicate engagement or confusion
        if session_duration > 300:  # 5 minutes
            satisfaction += 0.1 if follow_up_questions == 0 else -0.1
        
        # Task completion is positive
        if task_completion:
            satisfaction += 0.3
        
        # Multiple follow-ups might indicate confusion
        if follow_up_questions > 2:
            satisfaction -= 0.2
        
        # Normalize to -1 to 1 range
        reward_value = (satisfaction - 0.5) * 2
        
        return RewardSignal(
            signal_type=RewardSignalType.USER_FEEDBACK,
            value=reward_value,
            confidence=0.6,  # Lower confidence for implicit feedback
            source="user_implicit",
            context=user_behavior
        )

class QualityAssessmentReward:
    """Assesses response quality for reward calculation."""
    
    def __init__(self):
        self.quality_metrics = {
            "relevance": 0.3,
            "accuracy": 0.3,
            "completeness": 0.2,
            "clarity": 0.2
        }
    
    def assess_response_quality(self, 
                              query: str,
                              response: str,
                              knowledge_base_match: Optional[Dict[str, Any]] = None) -> RewardSignal:
        """Assess response quality and generate reward signal."""
        quality_scores = {}
        
        # Relevance assessment
        relevance = self._assess_relevance(query, response)
        quality_scores["relevance"] = relevance
        
        # Accuracy assessment (if knowledge base match available)
        if knowledge_base_match:
            accuracy = self._assess_accuracy(response, knowledge_base_match)
            quality_scores["accuracy"] = accuracy
        else:
            quality_scores["accuracy"] = 0.7  # Default when no reference
        
        # Completeness assessment
        completeness = self._assess_completeness(query, response)
        quality_scores["completeness"] = completeness
        
        # Clarity assessment
        clarity = self._assess_clarity(response)
        quality_scores["clarity"] = clarity
        
        # Calculate weighted quality score
        total_quality = sum(
            score * self.quality_metrics[metric]
            for metric, score in quality_scores.items()
        )
        
        # Convert to reward range (-1 to 1)
        reward_value = (total_quality - 0.5) * 2
        
        return RewardSignal(
            signal_type=RewardSignalType.RESPONSE_QUALITY,
            value=reward_value,
            confidence=0.8,
            source="quality_assessment",
            context={
                "quality_scores": quality_scores,
                "total_quality": total_quality
            }
        )
    
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess how relevant the response is to the query."""
        # Simple keyword overlap assessment
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words.intersection(response_words))
        relevance = min(1.0, overlap / len(query_words))
        
        # Boost for specific technical terms
        technical_terms = ["error", "bug", "issue", "problem", "solution", "fix"]
        if any(term in query.lower() for term in technical_terms):
            if any(term in response.lower() for term in technical_terms):
                relevance += 0.2
        
        return min(1.0, relevance)
    
    def _assess_accuracy(self, response: str, knowledge_base_match: Dict[str, Any]) -> float:
        """Assess accuracy against knowledge base."""
        # Check if response contains key information from knowledge base
        kb_content = knowledge_base_match.get("content", "")
        
        if not kb_content:
            return 0.5
        
        # Simple content similarity check
        response_lower = response.lower()
        kb_lower = kb_content.lower()
        
        # Check for key phrases
        kb_phrases = [phrase.strip() for phrase in kb_lower.split('.') if len(phrase.strip()) > 10]
        matches = sum(1 for phrase in kb_phrases if phrase in response_lower)
        
        if kb_phrases:
            accuracy = min(1.0, matches / len(kb_phrases))
        else:
            accuracy = 0.5
        
        return accuracy
    
    def _assess_completeness(self, query: str, response: str) -> float:
        """Assess if response completely addresses the query."""
        # Check response length appropriateness
        query_complexity = len(query.split())
        response_length = len(response.split())
        
        # Expect longer responses for complex queries
        expected_length = max(20, query_complexity * 2)
        length_ratio = min(1.0, response_length / expected_length)
        
        # Check for solution indicators
        solution_indicators = ["solution", "fix", "resolve", "try", "steps", "follow"]
        has_solution = any(indicator in response.lower() for indicator in solution_indicators)
        
        completeness = (length_ratio * 0.7) + (0.3 if has_solution else 0.0)
        return min(1.0, completeness)
    
    def _assess_clarity(self, response: str) -> float:
        """Assess clarity and readability of response."""
        # Simple clarity metrics
        sentences = response.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Prefer moderate sentence lengths (10-20 words)
        if 10 <= avg_sentence_length <= 20:
            length_score = 1.0
        else:
            length_score = max(0.3, 1.0 - abs(avg_sentence_length - 15) / 15)
        
        # Check for structure indicators
        structure_indicators = ["first", "second", "next", "finally", "step", "•", "-"]
        has_structure = any(indicator in response.lower() for indicator in structure_indicators)
        
        clarity = (length_score * 0.7) + (0.3 if has_structure else 0.0)
        return min(1.0, clarity)

class AdvancedRewardCalculator:
    """Advanced reward calculator with multiple signal integration."""
    
    def __init__(self):
        self.feedback_collector = UserFeedbackCollector()
        self.quality_assessor = QualityAssessmentReward()
        self.monitor = get_monitor()
        
        # Reward signal weights
        self.signal_weights = {
            RewardSignalType.USER_FEEDBACK: 0.4,
            RewardSignalType.RESPONSE_QUALITY: 0.3,
            RewardSignalType.PROCESSING_EFFICIENCY: 0.1,
            RewardSignalType.ESCALATION_ACCURACY: 0.1,
            RewardSignalType.COMMUNICATION_EFFECTIVENESS: 0.05,
            RewardSignalType.TOOL_USAGE_EFFICIENCY: 0.05
        }
        
        # Signal history for trend analysis
        self.signal_history: List[RewardSignal] = []
    
    def calculate_comprehensive_reward(self,
                                     agent_id: str,
                                     action: RLAction,
                                     outcome: Dict[str, Any],
                                     user_feedback: Optional[Dict[str, Any]] = None) -> float:
        """Calculate comprehensive reward from multiple signals."""
        signals = []
        
        # 1. User feedback signal
        if user_feedback:
            feedback_signal = self.feedback_collector.collect_explicit_feedback(
                user_id=user_feedback.get("user_id", "unknown"),
                response_id=outcome.get("response_id", "unknown"),
                satisfaction_score=user_feedback.get("satisfaction_score", 0.5),
                feedback_text=user_feedback.get("feedback_text")
            )
            signals.append(feedback_signal)
        
        # 2. Response quality signal
        if "query" in outcome and "response" in outcome:
            quality_signal = self.quality_assessor.assess_response_quality(
                query=outcome["query"],
                response=outcome["response"],
                knowledge_base_match=outcome.get("knowledge_base_match")
            )
            signals.append(quality_signal)
        
        # 3. Processing efficiency signal
        processing_time = outcome.get("response_time_ms", 5000)
        efficiency_reward = max(-0.5, 1.0 - (processing_time / 10000))  # Faster is better
        efficiency_signal = RewardSignal(
            signal_type=RewardSignalType.PROCESSING_EFFICIENCY,
            value=efficiency_reward,
            confidence=0.9,
            source="timing_analysis",
            context={"processing_time_ms": processing_time}
        )
        signals.append(efficiency_signal)
        
        # 4. Escalation accuracy signal
        if action.action_type == ActionType.ESCALATE_IMMEDIATELY:
            escalation_needed = outcome.get("escalation_was_needed", True)
            escalation_reward = 0.8 if escalation_needed else -0.6
            escalation_signal = RewardSignal(
                signal_type=RewardSignalType.ESCALATION_ACCURACY,
                value=escalation_reward,
                confidence=0.85,
                source="escalation_analysis",
                context={"escalation_needed": escalation_needed}
            )
            signals.append(escalation_signal)
        
        # 5. Communication effectiveness signal
        comm_effectiveness = outcome.get("communication_effectiveness", 0.5)
        comm_signal = RewardSignal(
            signal_type=RewardSignalType.COMMUNICATION_EFFECTIVENESS,
            value=(comm_effectiveness - 0.5) * 2,
            confidence=0.7,
            source="communication_analysis",
            context={"effectiveness_score": comm_effectiveness}
        )
        signals.append(comm_signal)
        
        # 6. Tool usage efficiency signal
        if action.tool_name:
            tool_efficiency = outcome.get("tool_efficiency", 0.5)
            tool_signal = RewardSignal(
                signal_type=RewardSignalType.TOOL_USAGE_EFFICIENCY,
                value=(tool_efficiency - 0.5) * 2,
                confidence=0.75,
                source="tool_analysis",
                context={"tool_name": action.tool_name, "efficiency": tool_efficiency}
            )
            signals.append(tool_signal)
        
        # Calculate weighted reward
        total_reward = 0.0
        total_weight = 0.0
        
        for signal in signals:
            weight = self.signal_weights[signal.signal_type]
            confidence_adjusted_weight = weight * signal.confidence
            total_reward += signal.value * confidence_adjusted_weight
            total_weight += confidence_adjusted_weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_reward = total_reward / total_weight
        else:
            final_reward = 0.0
        
        # Store signals for analysis
        self.signal_history.extend(signals)
        
        # Keep only recent signals (last 1000)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        # Record comprehensive reward
        self.monitor.record_reward_signal(
            agent_id=agent_id,
            reward=final_reward,
            context={
                "action_type": action.action_type.value,
                "signals": [
                    {
                        "type": s.signal_type.value,
                        "value": s.value,
                        "confidence": s.confidence,
                        "source": s.source
                    }
                    for s in signals
                ],
                "total_weight": total_weight
            }
        )
        
        logger.debug(f"Calculated comprehensive reward for {agent_id}: {final_reward:.3f} from {len(signals)} signals")
        return final_reward
    
    def get_reward_trends(self, agent_id: str, days: int = 7) -> Dict[str, Any]:
        """Get reward trends for an agent over specified days."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter recent signals
        recent_signals = [
            s for s in self.signal_history
            if datetime.fromisoformat(s.timestamp) > cutoff_time
        ]
        
        if not recent_signals:
            return {"trend": "no_data", "average_reward": 0.0}
        
        # Group by signal type
        signal_groups = {}
        for signal in recent_signals:
            signal_type = signal.signal_type.value
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal.value)
        
        # Calculate trends
        trends = {}
        for signal_type, values in signal_groups.items():
            if len(values) >= 2:
                # Simple linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                trends[signal_type] = {
                    "slope": slope,
                    "average": np.mean(values),
                    "count": len(values)
                }
        
        overall_average = np.mean([s.value for s in recent_signals])
        
        return {
            "agent_id": agent_id,
            "period_days": days,
            "signal_trends": trends,
            "overall_average": overall_average,
            "total_signals": len(recent_signals)
        }

class AdaptiveRewardShaping:
    """Implements adaptive reward shaping based on learning progress."""
    
    def __init__(self):
        self.agent_learning_curves: Dict[str, List[float]] = {}
        self.reward_adjustments: Dict[str, Dict[str, float]] = {}
        
    def update_learning_curve(self, agent_id: str, performance_score: float):
        """Update learning curve for an agent."""
        if agent_id not in self.agent_learning_curves:
            self.agent_learning_curves[agent_id] = []
        
        self.agent_learning_curves[agent_id].append(performance_score)
        
        # Keep only recent performance (last 100 episodes)
        if len(self.agent_learning_curves[agent_id]) > 100:
            self.agent_learning_curves[agent_id] = self.agent_learning_curves[agent_id][-100:]
    
    def get_reward_shaping_factor(self, agent_id: str, action_type: ActionType) -> float:
        """Get reward shaping factor based on learning progress."""
        if agent_id not in self.agent_learning_curves:
            return 1.0  # No adjustment
        
        curve = self.agent_learning_curves[agent_id]
        if len(curve) < 10:
            return 1.0  # Need more data
        
        # Analyze learning progress
        recent_performance = np.mean(curve[-10:])
        overall_performance = np.mean(curve)
        
        # If agent is struggling, provide more encouragement
        if recent_performance < 0.3:
            return 1.2  # Boost rewards
        elif recent_performance > 0.8:
            return 0.9  # Slightly reduce rewards to encourage exploration
        
        return 1.0  # No adjustment
    
    def suggest_reward_adjustments(self, agent_id: str) -> Dict[str, str]:
        """Suggest reward system adjustments based on learning patterns."""
        if agent_id not in self.agent_learning_curves:
            return {"suggestion": "No learning data available"}
        
        curve = self.agent_learning_curves[agent_id]
        if len(curve) < 20:
            return {"suggestion": "Need more training data"}
        
        # Analyze learning patterns
        recent_trend = np.polyfit(range(len(curve[-20:])), curve[-20:], 1)[0]
        
        suggestions = []
        
        if recent_trend < -0.01:
            suggestions.append("Learning is declining - consider reducing exploration rate")
        elif recent_trend > 0.01:
            suggestions.append("Good learning progress - maintain current settings")
        else:
            suggestions.append("Learning has plateaued - consider adjusting reward weights")
        
        if np.std(curve[-20:]) > 0.3:
            suggestions.append("High variance in performance - consider reward smoothing")
        
        return {
            "agent_id": agent_id,
            "recent_trend": recent_trend,
            "suggestions": suggestions
        }

class RewardSystemManager:
    """Main manager for the reward system."""
    
    def __init__(self):
        self.feedback_collector = UserFeedbackCollector()
        self.quality_assessor = QualityAssessmentReward()
        self.reward_calculator = AdvancedRewardCalculator()
        self.adaptive_shaping = AdaptiveRewardShaping()
        self.monitor = get_monitor()
        
        # System configuration
        self.reward_smoothing = True
        self.adaptive_shaping_enabled = True
        
    async def process_outcome_and_calculate_reward(self,
                                                 agent_id: str,
                                                 action: RLAction,
                                                 outcome: Dict[str, Any],
                                                 user_feedback: Optional[Dict[str, Any]] = None) -> float:
        """Process outcome and calculate final reward."""
        # Calculate base reward
        base_reward = self.reward_calculator.calculate_comprehensive_reward(
            agent_id=agent_id,
            action=action,
            outcome=outcome,
            user_feedback=user_feedback
        )
        
        # Apply adaptive shaping if enabled
        if self.adaptive_shaping_enabled:
            shaping_factor = self.adaptive_shaping.get_reward_shaping_factor(agent_id, action.action_type)
            shaped_reward = base_reward * shaping_factor
        else:
            shaped_reward = base_reward
        
        # Apply smoothing if enabled
        if self.reward_smoothing:
            # Simple exponential smoothing
            if agent_id in self.reward_calculator.feedback_collector.satisfaction_trends:
                recent_rewards = self.reward_calculator.feedback_collector.satisfaction_trends[agent_id]
                if recent_rewards:
                    smoothed_reward = 0.7 * shaped_reward + 0.3 * recent_rewards[-1]
                else:
                    smoothed_reward = shaped_reward
            else:
                smoothed_reward = shaped_reward
        else:
            smoothed_reward = shaped_reward
        
        # Update learning curve
        self.adaptive_shaping.update_learning_curve(agent_id, smoothed_reward)
        
        # Clamp to valid range
        final_reward = max(-1.0, min(1.0, smoothed_reward))
        
        logger.debug(f"Final reward for {agent_id}: {final_reward:.3f} (base: {base_reward:.3f})")
        return final_reward
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics."""
        return {
            "feedback_history_count": len(self.feedback_collector.feedback_history),
            "signal_history_count": len(self.reward_calculator.signal_history),
            "learning_curves": {
                agent_id: {
                    "data_points": len(curve),
                    "recent_average": np.mean(curve[-10:]) if len(curve) >= 10 else 0.0,
                    "overall_average": np.mean(curve) if curve else 0.0
                }
                for agent_id, curve in self.adaptive_shaping.agent_learning_curves.items()
            },
            "reward_smoothing": self.reward_smoothing,
            "adaptive_shaping_enabled": self.adaptive_shaping_enabled
        }

# Global reward system instance
_reward_system: Optional[RewardSystemManager] = None

def get_reward_system() -> RewardSystemManager:
    """Get the global reward system instance."""
    global _reward_system
    if _reward_system is None:
        _reward_system = RewardSystemManager()
    return _reward_system
