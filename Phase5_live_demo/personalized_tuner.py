"""
Phase 6: Personalized Weight Tuning
====================================
Diversity-Aware News Recommender System — Capstone Project

This module learns optimal diversity weights (λ, α, β) for each user based on
their feedback and interaction patterns.

Problem:
--------
Different users have different diversity preferences:
  - Some users want high diversity (explorers)
  - Some users prefer focused content (specialists)
  - Most users want adaptive diversity based on context

Solution:
---------
Three approaches to personalize diversity weights:

1. **Feedback-based Learning**
   - User rates recommendations (thumbs up/down)
   - System adjusts weights to maximize satisfaction
   - Uses simple Bayesian optimization

2. **Implicit Signal Detection**
   - Analyzes user behavior (dwell time, skip rate, category switching)
   - Infers diversity preference from interaction patterns
   - No explicit feedback needed

3. **Contextual Adaptation**
   - Adjusts weights based on time of day, device, session context
   - Morning: higher diversity (exploration)
   - Evening: lower diversity (focused reading)
   - Mobile: lower diversity (quick consumption)

Usage:
    from personalized_tuner import PersonalizedWeightTuner
    
    tuner = PersonalizedWeightTuner()
    
    # Learn from feedback
    tuner.update_from_feedback(user_id, recommendations, feedback_scores)
    
    # Get personalized weights
    weights = tuner.get_user_weights(user_id, context={'time_of_day': 'morning'})
    
    # Use in re-ranking
    recs = reranker.mmr_rerank(candidates, history, lambda_param=weights['lambda'])
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonalizedWeightTuner:
    """
    Learns and adapts diversity weights for individual users.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        """
        Args:
            learning_rate: Speed of weight adaptation (0.0-1.0)
        """
        self.learning_rate = learning_rate
        
        # User profiles: user_id -> weight parameters
        self.user_weights = defaultdict(self._default_weights)
        
        # Feedback history: user_id -> list of (weights, score) tuples
        self.feedback_history = defaultdict(list)
        
        # Interaction patterns: user_id -> behavior stats
        self.interaction_patterns = defaultdict(self._default_patterns)
    
    def _default_weights(self) -> Dict[str, float]:
        """Default starting weights for new users."""
        return {
            'lambda': 0.5,      # MMR / xQuAD balance
            'alpha': 0.5,       # Calibration strength
            'beta': 0.3,        # Serendipity weight
            'confidence': 0.0,  # How confident we are (0-1)
        }
    
    def _default_patterns(self) -> Dict:
        """Default interaction pattern stats."""
        return {
            'avg_dwell_time': 0.0,
            'skip_rate': 0.0,
            'category_switches': 0,
            'total_interactions': 0,
            'diversity_preference_score': 0.5,  # 0=focused, 1=diverse
        }
    
    # -----------------------------------------------------------------------
    # Method 1: Feedback-based Learning
    # -----------------------------------------------------------------------
    
    def update_from_feedback(
        self,
        user_id: str,
        recommendations: List[Tuple[str, float]],
        feedback_scores: List[float],
        algorithm: str = 'mmr',
        current_weights: Optional[Dict] = None,
    ):
        """
        Update user's weights based on explicit feedback.
        
        Args:
            user_id:           User identifier
            recommendations:   List of (news_id, score) tuples
            feedback_scores:   List of ratings (0-1, where 1=loved it)
            algorithm:         Which algorithm was used
            current_weights:   Weights that generated these recommendations
        """
        if current_weights is None:
            current_weights = self.user_weights[user_id]
        
        # Calculate overall satisfaction
        avg_feedback = np.mean(feedback_scores)
        
        # Store in history
        self.feedback_history[user_id].append((current_weights.copy(), avg_feedback))
        
        # Adjust weights based on feedback
        if algorithm == 'mmr' or algorithm == 'xquad':
            # If feedback is good, keep similar weights
            # If feedback is bad, adjust toward opposite direction
            current_lambda = current_weights.get('lambda', 0.5)
            
            if avg_feedback > 0.7:
                # Good feedback - move slightly toward current value
                new_lambda = current_lambda
            elif avg_feedback < 0.3:
                # Bad feedback - try opposite direction
                new_lambda = 1.0 - current_lambda
            else:
                # Neutral - random exploration
                new_lambda = np.random.uniform(0.3, 0.7)
            
            # Smooth update
            self.user_weights[user_id]['lambda'] = (
                (1 - self.learning_rate) * self.user_weights[user_id]['lambda'] +
                self.learning_rate * new_lambda
            )
        
        elif algorithm == 'calibrated':
            current_alpha = current_weights.get('alpha', 0.5)
            
            if avg_feedback > 0.7:
                new_alpha = min(1.0, current_alpha + 0.1)
            elif avg_feedback < 0.3:
                new_alpha = max(0.0, current_alpha - 0.1)
            else:
                new_alpha = current_alpha
            
            self.user_weights[user_id]['alpha'] = (
                (1 - self.learning_rate) * self.user_weights[user_id]['alpha'] +
                self.learning_rate * new_alpha
            )
        
        elif algorithm == 'serendipity':
            current_beta = current_weights.get('beta', 0.3)
            
            if avg_feedback > 0.7:
                new_beta = min(1.0, current_beta + 0.1)
            elif avg_feedback < 0.3:
                new_beta = max(0.0, current_beta - 0.1)
            else:
                new_beta = current_beta
            
            self.user_weights[user_id]['beta'] = (
                (1 - self.learning_rate) * self.user_weights[user_id]['beta'] +
                self.learning_rate * new_beta
            )
        
        # Increase confidence
        self.user_weights[user_id]['confidence'] = min(
            1.0,
            self.user_weights[user_id]['confidence'] + 0.05
        )
        
        logger.info(f"Updated weights for {user_id}: {self.user_weights[user_id]}")
    
    # -----------------------------------------------------------------------
    # Method 2: Implicit Signal Detection
    # -----------------------------------------------------------------------
    
    def update_from_interactions(
        self,
        user_id: str,
        interactions: List[Dict],
    ):
        """
        Learn from implicit signals (clicks, dwell time, skips).
        
        Args:
            user_id:       User identifier
            interactions:  List of dicts with:
                          - news_id: str
                          - action: 'click' | 'skip' | 'dwell'
                          - dwell_time: float (seconds, if action='dwell')
                          - category: str
        """
        patterns = self.interaction_patterns[user_id]
        
        # Update interaction stats
        for interaction in interactions:
            action = interaction['action']
            
            if action == 'dwell':
                dwell_time = interaction.get('dwell_time', 0)
                # Update average dwell time
                total = patterns['total_interactions']
                patterns['avg_dwell_time'] = (
                    (patterns['avg_dwell_time'] * total + dwell_time) / (total + 1)
                )
            
            elif action == 'skip':
                # Update skip rate
                total = patterns['total_interactions']
                patterns['skip_rate'] = (
                    (patterns['skip_rate'] * total + 1.0) / (total + 1)
                )
            
            elif action == 'click':
                # Check if category switched
                category = interaction.get('category')
                if hasattr(self, '_last_category') and category != self._last_category:
                    patterns['category_switches'] += 1
                self._last_category = category
            
            patterns['total_interactions'] += 1
        
        # Infer diversity preference from patterns
        # High dwell time + low skip rate + many switches = diverse user
        # Low dwell time + high skip rate + few switches = focused user
        
        if patterns['total_interactions'] > 10:
            # Normalize signals
            dwell_signal = min(1.0, patterns['avg_dwell_time'] / 60.0)  # Cap at 60s
            skip_signal = 1.0 - patterns['skip_rate']
            switch_signal = min(1.0, patterns['category_switches'] / patterns['total_interactions'] * 5)
            
            # Weighted average
            diversity_score = 0.4 * dwell_signal + 0.3 * skip_signal + 0.3 * switch_signal
            patterns['diversity_preference_score'] = diversity_score
            
            # Map to weights
            # High diversity score → lower lambda (more diversity in MMR)
            self.user_weights[user_id]['lambda'] = 1.0 - diversity_score * 0.6  # Range: 0.4-1.0
            self.user_weights[user_id]['alpha'] = 0.3 + diversity_score * 0.4   # Range: 0.3-0.7
            self.user_weights[user_id]['beta'] = 0.2 + diversity_score * 0.4    # Range: 0.2-0.6
            
            logger.info(f"Inferred diversity preference for {user_id}: {diversity_score:.2f}")
    
    # -----------------------------------------------------------------------
    # Method 3: Contextual Adaptation
    # -----------------------------------------------------------------------
    
    def get_contextual_weights(
        self,
        user_id: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Get weights adjusted for current context.
        
        Args:
            user_id:  User identifier
            context:  Dict with optional keys:
                     - time_of_day: 'morning' | 'afternoon' | 'evening' | 'night'
                     - device: 'mobile' | 'desktop' | 'tablet'
                     - session_length: int (number of articles viewed this session)
        
        Returns:
            Dict with adjusted weights
        """
        # Start with base weights
        base_weights = self.user_weights[user_id].copy()
        
        if context is None:
            return base_weights
        
        # Time of day adjustments
        time_of_day = context.get('time_of_day', 'afternoon')
        
        if time_of_day == 'morning':
            # Morning: people explore more (higher diversity)
            base_weights['lambda'] *= 0.8  # Lower lambda = more diversity
            base_weights['beta'] *= 1.2    # Higher serendipity
        
        elif time_of_day == 'evening':
            # Evening: focused reading (lower diversity)
            base_weights['lambda'] = min(1.0, base_weights['lambda'] * 1.2)
            base_weights['beta'] *= 0.8
        
        elif time_of_day == 'night':
            # Night: light reading, moderate diversity
            pass  # Use base weights
        
        # Device adjustments
        device = context.get('device', 'desktop')
        
        if device == 'mobile':
            # Mobile: quick consumption, focused content
            base_weights['lambda'] = min(1.0, base_weights['lambda'] * 1.1)
            base_weights['beta'] *= 0.9
        
        elif device == 'desktop':
            # Desktop: longer sessions, can handle more diversity
            base_weights['lambda'] *= 0.95
        
        # Session length adjustments
        session_length = context.get('session_length', 0)
        
        if session_length > 10:
            # Long session: user is exploring, increase diversity
            base_weights['lambda'] *= 0.9
            base_weights['beta'] *= 1.1
        
        # Clamp all values to valid ranges
        base_weights['lambda'] = np.clip(base_weights['lambda'], 0.0, 1.0)
        base_weights['alpha'] = np.clip(base_weights['alpha'], 0.0, 1.0)
        base_weights['beta'] = np.clip(base_weights['beta'], 0.0, 1.0)
        
        return base_weights
    
    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    
    def get_user_weights(
        self,
        user_id: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Get personalized weights for a user.
        
        This is the main method to call from your recommender.
        
        Args:
            user_id:  User identifier
            context:  Optional context (time, device, etc.)
        
        Returns:
            Dict with 'lambda', 'alpha', 'beta', 'confidence'
        """
        return self.get_contextual_weights(user_id, context)
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get complete user profile including weights and patterns."""
        return {
            'user_id': user_id,
            'weights': self.user_weights[user_id],
            'patterns': self.interaction_patterns[user_id],
            'feedback_count': len(self.feedback_history[user_id]),
        }
    
    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    
    def save(self, filepath: str):
        """Save user profiles to disk."""
        data = {
            'user_weights': dict(self.user_weights),
            'interaction_patterns': dict(self.interaction_patterns),
            'feedback_history': {
                uid: [(w, s) for w, s in history]
                for uid, history in self.feedback_history.items()
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.user_weights)} user profiles to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PersonalizedWeightTuner':
        """Load user profiles from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tuner = cls()
        tuner.user_weights = defaultdict(tuner._default_weights, data['user_weights'])
        tuner.interaction_patterns = defaultdict(tuner._default_patterns, data['interaction_patterns'])
        tuner.feedback_history = defaultdict(list, {
            uid: [(w, s) for w, s in history]
            for uid, history in data['feedback_history'].items()
        })
        
        logger.info(f"Loaded {len(tuner.user_weights)} user profiles from {filepath}")
        
        return tuner


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def infer_time_of_day() -> str:
    """Infer current time of day."""
    hour = datetime.now().hour
    
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'night'


if __name__ == "__main__":
    # Example usage
    tuner = PersonalizedWeightTuner()
    
    # Simulate user feedback
    user_id = "user_123"
    
    # User gives positive feedback on diverse recommendations
    tuner.update_from_feedback(
        user_id=user_id,
        recommendations=[('N1', 0.9), ('N2', 0.8)],
        feedback_scores=[0.9, 0.8],
        algorithm='mmr',
        current_weights={'lambda': 0.3},
    )
    
    # Get personalized weights
    weights = tuner.get_user_weights(user_id, context={'time_of_day': 'morning'})
    print(f"Personalized weights for {user_id}: {weights}")
    
    # Save profiles
    tuner.save('./user_profiles.json')
