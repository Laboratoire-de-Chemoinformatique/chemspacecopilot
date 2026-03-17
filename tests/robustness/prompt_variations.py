#!/usr/bin/env python
# coding: utf-8
"""
Prompt variation generation and management for robustness testing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class PromptVariationGenerator:
    """Generate and manage prompt variations for robustness testing."""

    def __init__(self, templates_path: Optional[Path] = None, validate_similarity: bool = True):
        """
        Initialize prompt variation generator.

        Args:
            templates_path: Path to YAML file with prompt templates
            validate_similarity: Whether to validate semantic similarity of variations
        """
        if templates_path is None:
            env_override = os.environ.get("CS_COPILOT_PROMPT_TEMPLATES")
            templates_path = (
                Path(env_override)
                if env_override
                else Path(__file__).parent / "fixtures" / "prompt_templates.yaml"
            )

        self.templates_path = Path(templates_path)
        self.templates = self._load_templates()
        self.validate_similarity = validate_similarity

        if validate_similarity:
            logger.info("Loading sentence transformer model for similarity validation")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.model = None

    def _load_templates(self) -> Dict:
        """Load prompt templates from YAML file."""
        try:
            with open(self.templates_path, "r") as f:
                data = yaml.safe_load(f)
            logger.info(f"Loaded prompt templates from {self.templates_path}")
            return data.get("prompts", {})
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            raise

    def get_variations(self, prompt_key: str, n: int = 10) -> List[str]:
        """
        Get N variations of a base prompt.

        Args:
            prompt_key: Key identifying the prompt category
            n: Number of variations to return

        Returns:
            List of prompt variations
        """
        if prompt_key not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Unknown prompt key: {prompt_key}. Available: {available}")

        template_data = self.templates[prompt_key]
        base_prompts = template_data.get("base_prompts")

        if base_prompts:
            all_prompts: List[str] = []
            for prompt_case in base_prompts:
                all_prompts.append(prompt_case.get("base", ""))
                all_prompts.extend(prompt_case.get("variations", [])[:n])
        else:
            base_prompt = template_data.get("base", "")
            variations = template_data.get("variations", [])
            all_prompts = [base_prompt] + variations

        if len(all_prompts) < n:
            logger.warning(
                f"Requested {n} variations but only {len(all_prompts)} available for {prompt_key}"
            )

        selected = all_prompts[:n]

        # Validate similarity if enabled
        if self.validate_similarity and len(selected) > 1:
            self._validate_variations(prompt_key, selected)

        return selected

    def get_base_prompt(self, prompt_key: str) -> str:
        """Get the base prompt for a given key."""
        if prompt_key not in self.templates:
            raise ValueError(f"Unknown prompt key: {prompt_key}")

        template = self.templates[prompt_key]
        base_prompts = template.get("base_prompts")
        if base_prompts:
            return base_prompts[0].get("base", "")

        return template.get("base", "")

    def get_prompt_cases(self, prompt_key: str, n_variations: int = 5) -> List[Dict]:
        """
        Return structured prompt cases with metadata.

        Each case includes the base prompt, a limited set of variations, and a
        flag indicating whether the agent should ask for clarification before
        proceeding.
        """

        if prompt_key not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Unknown prompt key: {prompt_key}. Available: {available}")

        template = self.templates[prompt_key]
        base_prompts = template.get("base_prompts")

        if not base_prompts:
            raise ValueError(
                f"Prompt key '{prompt_key}' does not define 'base_prompts'."
                " Use legacy helpers instead."
            )

        prompt_cases = []
        for prompt_case in base_prompts:
            variations = prompt_case.get("variations", [])[:n_variations]
            prompt_cases.append(
                {
                    "base": prompt_case.get("base", ""),
                    "variations": variations,
                    "requires_clarification": bool(
                        prompt_case.get("requires_clarification", False)
                    ),
                }
            )

        return prompt_cases

    def validate_variation(self, base: str, variation: str, min_similarity: float = 0.70) -> bool:
        """
        Ensure variation preserves intent via embedding similarity check.

        Args:
            base: Base prompt
            variation: Variation to validate
            min_similarity: Minimum cosine similarity threshold

        Returns:
            True if variation is valid
        """
        if not self.model:
            logger.warning("Similarity validation disabled - no model loaded")
            return True

        embeddings = self.model.encode([base, variation])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        is_valid = similarity >= min_similarity
        logger.debug(f"Similarity: {similarity:.3f} ({'valid' if is_valid else 'INVALID'})")

        return is_valid

    def _validate_variations(
        self, prompt_key: str, variations: List[str], min_similarity: float = 0.70
    ):
        """Validate all variations against the base prompt."""
        if not variations:
            return

        base = variations[0]  # First is the base
        logger.info(f"Validating {len(variations)-1} variations for '{prompt_key}'")

        for i, variation in enumerate(variations[1:], 1):
            is_valid = self.validate_variation(base, variation, min_similarity)
            if not is_valid:
                logger.warning(f"Variation {i} for '{prompt_key}' has low similarity to base")

    def list_available_prompts(self) -> List[str]:
        """List all available prompt keys."""
        return list(self.templates.keys())

    def add_variation(self, prompt_key: str, variation: str, validate: bool = True):
        """
        Add a new variation to an existing prompt category.

        Args:
            prompt_key: Key identifying the prompt category
            variation: New variation to add
            validate: Whether to validate semantic similarity
        """
        if prompt_key not in self.templates:
            raise ValueError(f"Unknown prompt key: {prompt_key}")

        if validate and self.model:
            base = self.templates[prompt_key]["base"]
            if not self.validate_variation(base, variation):
                logger.warning(f"New variation for '{prompt_key}' has low similarity to base")

        self.templates[prompt_key]["variations"].append(variation)
        logger.info(f"Added variation to '{prompt_key}'")

    def generate_llm_variations(
        self, base: str, n: int = 10, model_name: str = "gpt-4"
    ) -> List[str]:
        """
        Use LLM to generate paraphrases (future enhancement).

        Args:
            base: Base prompt to paraphrase
            n: Number of variations to generate
            model_name: LLM model to use for generation

        Returns:
            List of generated variations
        """
        # TODO: Implement LLM-based paraphrasing
        # This would use OpenAI/Anthropic API to generate semantic equivalents
        raise NotImplementedError(
            "LLM-based variation generation not yet implemented. "
            "Use template-based variations for now."
        )
