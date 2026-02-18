"""
Phase 7: LLM-Adaptive Content Reframing
========================================
Diversity-Aware News Recommender System — Capstone Project

This module uses Large Language Models (Claude, GPT-4, or local models) to
reduce bias and improve content diversity through adaptive reframing.

Problem:
--------
Even with diverse recommendations, the CONTENT ITSELF may:
  - Contain bias or one-sided framing
  - Use polarizing language
  - Lack alternative perspectives
  - Reinforce filter bubbles through phrasing

Solution:
---------
LLM-powered content reframing:

1. **Bias Detection**
   - Analyze article text for biased language
   - Detect one-sided framing
   - Identify missing perspectives

2. **Multi-Perspective Reframing**
   - Rewrite article summary from multiple viewpoints
   - Generate "steel-man" counterarguments
   - Add context and nuance

3. **Neutralization**
   - Remove inflammatory language
   - Replace loaded terms with neutral equivalents
   - Maintain factual accuracy while reducing polarization

4. **Adaptive Summarization**
   - Tailor summary length to user preference
   - Highlight aspects user hasn't seen before
   - Connect to user's broader interests

Usage:
    from llm_reframer import LLMContentReframer
    
    reframer = LLMContentReframer(provider='anthropic')  # or 'openai', 'local'
    
    # Detect bias
    bias_score = reframer.detect_bias(article_text)
    
    # Reframe with multiple perspectives
    reframed = reframer.reframe_multi_perspective(
        title="Senate Passes Climate Bill",
        abstract="Democrats celebrate...",
        target_perspectives=['progressive', 'conservative', 'neutral']
    )
    
    # Neutralize polarizing content
    neutral = reframer.neutralize(article_text)
"""

import json
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMContentReframer:
    """
    Uses LLMs to reframe news content for reduced bias and increased diversity.
    """
    
    def __init__(
        self,
        provider: str = 'anthropic',
        model: str = 'claude-sonnet-4-20250514',
        api_key: Optional[str] = None,
    ):
        """
        Args:
            provider:  'anthropic', 'openai', or 'local'
            model:     Model identifier
            api_key:   API key (if None, reads from environment)
        """
        self.provider = provider
        self.model = model
        
        # Initialize API client
        if provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"✓ Anthropic client initialized (model: {model})")
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                self.client = None
        
        elif provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"✓ OpenAI client initialized (model: {model})")
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                self.client = None
        
        elif provider == 'local':
            # For local models (Ollama, llama.cpp, etc.)
            logger.info("Using local LLM (implement your own client)")
            self.client = None
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    # -----------------------------------------------------------------------
    # Core LLM Interaction
    # -----------------------------------------------------------------------
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call the LLM with a prompt."""
        if self.client is None:
            return "[LLM not available - install anthropic or openai package]"
        
        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            
            else:
                return "[Local LLM not implemented]"
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[Error: {str(e)}]"
    
    # -----------------------------------------------------------------------
    # Feature 1: Bias Detection
    # -----------------------------------------------------------------------
    
    def detect_bias(self, text: str) -> Dict:
        """
        Analyze text for bias indicators.
        
        Returns:
            Dict with:
              - bias_score: float (0-1, where 1=highly biased)
              - bias_types: list of detected bias types
              - flagged_phrases: list of problematic phrases
              - explanation: str
        """
        prompt = f"""Analyze the following news article for bias. Identify:
1. Political bias (left/right leaning language)
2. Loaded language (emotionally charged words)
3. One-sided framing (missing alternative perspectives)
4. Logical fallacies

Article:
{text[:1000]}

Respond in JSON format:
{{
  "bias_score": 0.0-1.0,
  "bias_types": ["political", "emotional", "one-sided"],
  "flagged_phrases": ["phrase1", "phrase2"],
  "explanation": "brief analysis"
}}
"""
        
        response = self._call_llm(prompt, max_tokens=500)
        
        try:
            # Parse JSON response
            result = json.loads(response.strip('```json\n').strip('```\n'))
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                'bias_score': 0.5,
                'bias_types': ['unknown'],
                'flagged_phrases': [],
                'explanation': response,
            }
    
    # -----------------------------------------------------------------------
    # Feature 2: Multi-Perspective Reframing
    # -----------------------------------------------------------------------
    
    def reframe_multi_perspective(
        self,
        title: str,
        abstract: str,
        target_perspectives: List[str] = None,
    ) -> Dict[str, str]:
        """
        Rewrite article summary from multiple viewpoints.
        
        Args:
            title:                Article title
            abstract:             Article abstract
            target_perspectives:  List of perspectives to generate
                                 e.g. ['progressive', 'conservative', 'neutral']
        
        Returns:
            Dict mapping perspective → reframed summary
        """
        if target_perspectives is None:
            target_perspectives = ['progressive', 'conservative', 'neutral', 'international']
        
        results = {}
        
        for perspective in target_perspectives:
            prompt = f"""Rewrite this news summary from a {perspective} perspective, 
maintaining factual accuracy but adjusting tone and emphasis.

Original Title: {title}
Original Abstract: {abstract}

Rewritten from {perspective} perspective (1-2 sentences):"""
            
            response = self._call_llm(prompt, max_tokens=200)
            results[perspective] = response.strip()
        
        return results
    
    # -----------------------------------------------------------------------
    # Feature 3: Neutralization
    # -----------------------------------------------------------------------
    
    def neutralize(self, text: str) -> str:
        """
        Remove polarizing language while maintaining factual content.
        
        Args:
            text: Original article text
        
        Returns:
            Neutralized version
        """
        prompt = f"""Rewrite this news article to be more neutral and balanced:

1. Replace loaded language with neutral terms
2. Remove inflammatory adjectives
3. Present facts without editorializing
4. Include multiple perspectives if discussing controversial topics
5. Maintain all factual information

Original:
{text[:800]}

Neutralized version:"""
        
        response = self._call_llm(prompt, max_tokens=1000)
        return response.strip()
    
    # -----------------------------------------------------------------------
    # Feature 4: Adaptive Summarization
    # -----------------------------------------------------------------------
    
    def adaptive_summarize(
        self,
        text: str,
        user_history_categories: List[str],
        target_length: str = 'medium',
    ) -> str:
        """
        Generate personalized summary highlighting novel aspects.
        
        Args:
            text:                     Article text
            user_history_categories:  Categories user has read before
            target_length:            'short' (1 sent), 'medium' (2-3), 'long' (4-5)
        
        Returns:
            Personalized summary
        """
        length_map = {
            'short': '1 sentence',
            'medium': '2-3 sentences',
            'long': '4-5 sentences',
        }
        
        prompt = f"""Summarize this article in {length_map[target_length]}.

The user has previously read articles about: {', '.join(user_history_categories)}

Focus on aspects that are NEW or DIFFERENT from what the user typically reads,
while still capturing the main point.

Article:
{text[:1000]}

Summary ({target_length}):"""
        
        response = self._call_llm(prompt, max_tokens=300)
        return response.strip()
    
    # -----------------------------------------------------------------------
    # Feature 5: Counterargument Generation
    # -----------------------------------------------------------------------
    
    def generate_counterarguments(self, text: str, num_arguments: int = 3) -> List[str]:
        """
        Generate steel-man counterarguments to the article's main claim.
        
        Args:
            text:           Article text
            num_arguments:  Number of counterarguments to generate
        
        Returns:
            List of counterarguments
        """
        prompt = f"""Read this article and identify its main argument.
Then generate {num_arguments} STRONG counterarguments that someone might reasonably make.
These should be "steel-man" arguments (the best possible version), not strawman.

Article:
{text[:800]}

Counterarguments (numbered list):"""
        
        response = self._call_llm(prompt, max_tokens=500)
        
        # Parse numbered list
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        counterarguments = [
            line.split('.', 1)[1].strip() if '.' in line else line
            for line in lines
            if any(char.isdigit() for char in line[:3])
        ]
        
        return counterarguments[:num_arguments]
    
    # -----------------------------------------------------------------------
    # Feature 6: Missing Context Detection
    # -----------------------------------------------------------------------
    
    def identify_missing_context(self, text: str) -> Dict:
        """
        Identify what important context might be missing from the article.
        
        Returns:
            Dict with:
              - missing_background: str
              - unanswered_questions: list[str]
              - suggested_additions: list[str]
        """
        prompt = f"""Analyze this news article and identify:
1. Important background information that's missing
2. Key questions the article doesn't answer
3. Alternative perspectives or viewpoints not mentioned

Article:
{text[:800]}

Respond in JSON format:
{{
  "missing_background": "brief description",
  "unanswered_questions": ["question1", "question2"],
  "suggested_additions": ["addition1", "addition2"]
}}
"""
        
        response = self._call_llm(prompt, max_tokens=400)
        
        try:
            result = json.loads(response.strip('```json\n').strip('```\n'))
            return result
        except json.JSONDecodeError:
            return {
                'missing_background': '',
                'unanswered_questions': [],
                'suggested_additions': [],
                'explanation': response,
            }
    
    # -----------------------------------------------------------------------
    # Integrated Workflow
    # -----------------------------------------------------------------------
    
    def enhance_article(
        self,
        title: str,
        abstract: str,
        full_text: Optional[str] = None,
        user_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Complete enhancement workflow: detect bias, reframe, neutralize.
        
        Args:
            title:        Article title
            abstract:     Article abstract
            full_text:    Full article text (optional)
            user_context: Dict with user's reading history
        
        Returns:
            Dict with all enhancements
        """
        text = full_text or abstract
        
        # 1. Detect bias
        bias_analysis = self.detect_bias(text)
        
        # 2. Generate multiple perspectives
        perspectives = self.reframe_multi_perspective(title, abstract)
        
        # 3. Neutralize if biased
        neutral_version = None
        if bias_analysis['bias_score'] > 0.6:
            neutral_version = self.neutralize(abstract)
        
        # 4. Adaptive summary
        adaptive_summary = None
        if user_context and 'history_categories' in user_context:
            adaptive_summary = self.adaptive_summarize(
                text,
                user_context['history_categories'],
                target_length='medium',
            )
        
        # 5. Counterarguments
        counterarguments = self.generate_counterarguments(text)
        
        return {
            'original_title': title,
            'original_abstract': abstract,
            'bias_analysis': bias_analysis,
            'perspectives': perspectives,
            'neutral_version': neutral_version,
            'adaptive_summary': adaptive_summary,
            'counterarguments': counterarguments,
        }


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize reframer (requires API key)
    # You can set ANTHROPIC_API_KEY environment variable
    # or pass api_key='your-key-here'
    
    reframer = LLMContentReframer(provider='anthropic')
    
    # Example article
    title = "Senate Passes Landmark Climate Bill"
    abstract = ("Democrats celebrated a major victory as the Senate passed "
                "comprehensive climate legislation, though Republicans warned "
                "of economic consequences.")
    
    # Detect bias
    bias = reframer.detect_bias(abstract)
    print(f"\nBias Analysis:")
    print(f"  Score: {bias.get('bias_score', 'N/A')}")
    print(f"  Types: {bias.get('bias_types', [])}")
    
    # Multi-perspective reframing
    perspectives = reframer.reframe_multi_perspective(title, abstract)
    print(f"\nMultiple Perspectives:")
    for perspective, text in perspectives.items():
        print(f"  {perspective}: {text}")
    
    # Neutralize
    neutral = reframer.neutralize(abstract)
    print(f"\nNeutralized Version:")
    print(f"  {neutral}")
