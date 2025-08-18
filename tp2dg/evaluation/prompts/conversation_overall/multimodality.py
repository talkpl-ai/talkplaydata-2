"""
Multimodality Evaluation.

This module evaluates how multimodal aspects (audio and image) are correctly considered
by both Listener and Recsys in the conversation, rating as True/False/NotRelevant.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import extract_conversation_turns, get_recommended_tracks_content
from tp2dg.prompts.prompt_template import PromptTemplate

multimodality_prompt = PromptTemplate(
    name="multimodality",
    version="v1.0",
    description="Rate how multimodal aspects are correctly considered by Listener and Recsys",
    required_params=[
        "conversation_goal",
        "conversation_turns",
        "recommended_tracks_content",
    ],
    response_expected_fields=[
        "multimodal_consideration",
    ],
    template="""
You are an expert evaluator of music recommendation systems. Assess how multimodal aspects (audio and image) are correctly considered by both Listener and Recsys in the conversation.

## CONVERSATION GOAL:
{conversation_goal}

## CONVERSATION TURNS:
{conversation_turns}

## RECOMMENDED TRACKS:
{recommended_tracks_content}

## EVALUATION TASK:
Rate whether multimodal aspects (audio and image) are correctly considered by both Listener and Recsys when such information is not available through text modality alone.

**Consider:**
- Audio Utilization: Do the participants reference or consider audio characteristics when text descriptions are insufficient?
- Image Utilization: Do the participants reference or consider album cover images when text descriptions are insufficient?
- Contextual Relevance: Is multimodal information used appropriately in the conversation context?
- Information Gap: Is multimodal information used to fill gaps that text alone cannot address?

**Classification Criteria:**

**True (Correctly Considered)**: Multimodal aspects are appropriately considered
- Audio characteristics are referenced when text descriptions are insufficient
- Album cover images are considered when relevant to the conversation
- Multimodal information is used to enhance understanding beyond text
- Both Listener and Recsys demonstrate awareness of multimodal content

**False (Incorrectly Considered)**: Multimodal aspects are not properly considered
- Audio or image information is ignored when it should be relevant
- Multimodal content is mentioned but not meaningfully integrated
- Participants fail to utilize available multimodal information
- Multimodal aspects are considered inappropriately or incorrectly

**NotRelevant**: Multimodal aspects are not relevant to this conversation
- The conversation goal doesn't require audio or image consideration
- Text descriptions are sufficient for the conversation context
- No audio or image information is available or needed
- The conversation focuses purely on textual music information

**Examples of Multimodal Consideration:**
- **Audio**: Referencing tempo, mood, instrumentation, production quality
- **Image**: Considering album artwork style, visual themes, artist branding
- **Combined**: Using both audio and visual elements to make recommendations

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
multimodal_consideration: "True"/"False"/"NotRelevant"
```
""",
)


class MultimodalityEvaluator(BaseEvaluationTemplate):
    """Evaluates how multimodal aspects are correctly considered by Listener and Recsys."""

    def __init__(self):
        super().__init__()
        self.evaluation_name = "multimodality"
        self.prompt_template = multimodality_prompt

    def prepare_prompt_data(
        self,
        chat_json: dict[str, Any],
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Prepare data for the multimodality prompt."""

        # Use empty dicts if no uploaded files provided (for backwards compatibility)
        if uploaded_audio_files is None:
            uploaded_audio_files = {}
        if uploaded_image_files is None:
            uploaded_image_files = {}

        # Extract conversation goal
        conversation_goal = chat_json["conversation_goal"]
        goal_text = json.dumps(conversation_goal)

        # Extract conversation turns using utility function
        conversation_turns = extract_conversation_turns(chat_json)
        turns_text = json.dumps(conversation_turns)

        recommended_tracks_content = get_recommended_tracks_content(
            chat_json,
            uploaded_audio_files,
            uploaded_image_files,
        )

        return {
            "conversation_goal": goal_text,
            "conversation_turns": turns_text,
            "recommended_tracks_content": recommended_tracks_content,
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

            # Extract multimodal consideration
            multimodal_consideration = parsed["multimodal_consideration"]

            # Validate classification
            valid_classifications = ["True", "False", "NotRelevant"]
            if multimodal_consideration not in valid_classifications:
                raise ValueError(f"Invalid multimodal consideration: {multimodal_consideration}")

            return {
                "multimodal_consideration": multimodal_consideration,
                "success": True,
                "raw_response": response,
                "parsed_response": parsed,
            }

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {
                "multimodal_consideration": "NotRelevant",
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
                "success_rate": 0.0,
                "multimodal_success_rate": 0.0,
                "classification_distribution": {},
            }

        # Extract classifications
        classifications = [r.get("multimodal_consideration", "NotRelevant") for r in successful_results]

        # Calculate success rate (True / (True + False))
        true_count = classifications.count("True")
        false_count = classifications.count("False")
        not_relevant_count = classifications.count("NotRelevant")

        total_relevant = true_count + false_count
        multimodal_success_rate = true_count / total_relevant if total_relevant > 0 else 0.0

        # Calculate distribution
        classification_dist = {
            "True": {
                "count": true_count,
                "percentage": true_count / len(successful_results) * 100,
            },
            "False": {
                "count": false_count,
                "percentage": false_count / len(successful_results) * 100,
            },
            "NotRelevant": {
                "count": not_relevant_count,
                "percentage": not_relevant_count / len(successful_results) * 100,
            },
        }

        return {
            "total_conversations": len(individual_results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results),
            "multimodal_success_rate": multimodal_success_rate,
            "classification_distribution": classification_dist,
            "relevant_conversations": total_relevant,
            "individual_results": [
                {
                    "conversation_id": r.get("conversation_id", "unknown"),
                    "multimodal_consideration": r.get("multimodal_consideration"),
                }
                for r in successful_results
            ],
        }


# Create the evaluator instance for easy import
multimodality_evaluator = MultimodalityEvaluator()
