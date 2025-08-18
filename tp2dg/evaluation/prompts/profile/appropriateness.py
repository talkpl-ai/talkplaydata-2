"""
Profile Appropriateness Evaluation.

This module evaluates whether the generated listener profile is appropriate/realistic
given the profiling tracks, using LLM-as-a-judge with a 4-point rubric.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import get_profiling_tracks_content
from tp2dg.prompts.prompt_template import PromptTemplate

profile_appropriateness_prompt = PromptTemplate(
    name="profile_appropriateness",
    version="v1.0",
    description="Evaluate whether the generated listener profile is appropriate given the profiling tracks",
    required_params=[
        "listener_profile",
        "profiling_tracks_content",
    ],
    response_expected_fields=[
        "appropriateness_score",
    ],
    template="""
You are an expert evaluator of music recommendation systems. Assess whether the given listener profile is appropriate and realistic given the profiling tracks (tracks the listener has previously liked).

## LISTENER PROFILE:
{listener_profile}

## PROFILING TRACKS (Previously liked by listener):
{profiling_tracks_content}

## EVALUATION TASK:
Rate the appropriateness of this listener profile given the profiling tracks.

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Profile perfectly matches the profiling tracks and is highly realistic
- **3**: Good - Profile is appropriate and mostly consistent with the tracks
- **2**: Fair - Profile is somewhat appropriate but has some inconsistencies
- **1**: Poor - Profile is inappropriate or unrealistic given the profiling tracks

**Consider:**
- Profile-Track Alignment: Do the profile attributes (preferred_musical_culture, top_1_artist, top_1_genre) align with the profiling tracks?
- Realistic Profile: Is the profile realistic and coherent?
- Cultural Consistency: Does the preferred_musical_culture make sense given the track characteristics?

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
appropriateness_score: [1-4]
```
""",
)


class ProfileAppropriatenessEvaluator(BaseEvaluationTemplate):
    """Evaluates profile appropriateness using LLM-as-a-judge with proper Track entities."""

    def __init__(self):
        super().__init__()
        self.evaluation_name = "profile_appropriateness"
        self.prompt_template = profile_appropriateness_prompt

    def prepare_prompt_data(
        self,
        chat_json: dict[str, Any],
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Prepare data for the profile appropriateness prompt."""

        # Use empty dicts if no uploaded files provided (for backwards compatibility)
        if uploaded_audio_files is None:
            uploaded_audio_files = {}
        if uploaded_image_files is None:
            uploaded_image_files = {}

        # Extract listener profile
        listener_profile = chat_json["listener_profile"]
        profile_text = json.dumps(listener_profile)

        # Format tracks using their standard methods (with audio/image if available)
        profiling_tracks_content = get_profiling_tracks_content(chat_json, uploaded_audio_files, uploaded_image_files)

        return {
            "listener_profile": profile_text,
            "profiling_tracks_content": profiling_tracks_content,
        }

    async def evaluate_single(
        self,
        conversation_data: dict[str, Any],
        llm_call_func,
        client,
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Evaluate a single conversation using centralized uploaded files."""

        # Use empty dicts if no uploaded files provided (for backwards compatibility)
        if uploaded_audio_files is None:
            uploaded_audio_files = {}
        if uploaded_image_files is None:
            uploaded_image_files = {}

        # Get prompt data with uploaded files
        prompt_data = self.prepare_prompt_data(conversation_data, uploaded_audio_files, uploaded_image_files)

        content = self.prompt_template.format(**prompt_data)

        # Call LLM
        response = await llm_call_func(content, client)

        # Parse response
        try:
            parsed = robust_parse_yaml_response(
                response,
                self.prompt_template.response_expected_fields,
            )

            # Extract just the score
            score = parsed["appropriateness_score"]
            if isinstance(score, str):
                score = int(score)

            return {
                "appropriateness_score": score,
                "success": score > 0 and score <= 4,
                "raw_response": response,
                "parsed_response": parsed,
            }

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {
                "appropriateness_score": 0,
                "success": False,
                "raw_response": response,
                "error": str(e),
            }

    def aggregate_results(self, individual_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate individual results into summary statistics."""

        successful_results = [r for r in individual_results if r.get("success", False)]

        if not successful_results:
            return {
                "total_conversations": len(individual_results),
                "successful_evaluations": 0,
                "average_score": 0.0,
                "score_distribution": {},
                "success_rate": 0.0,
            }

        scores = [r["appropriateness_score"] for r in successful_results]

        # Calculate distribution
        score_distribution = {i: scores.count(i) for i in range(1, 5)}

        return {
            "total_conversations": len(individual_results),
            "successful_evaluations": len(successful_results),
            "average_score": sum(scores) / len(scores),
            "score_distribution": score_distribution,
            "success_rate": len(successful_results) / len(individual_results),
            "scores": scores,  # For further analysis
        }


# Create the evaluator instance for easy import
profile_appropriateness_evaluator = ProfileAppropriatenessEvaluator()
