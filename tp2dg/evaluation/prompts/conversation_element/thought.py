"""
Thought Evaluation.

This module evaluates thought coherence and helpfulness for both Listener and Recsys,
assessing if thoughts explain the actual thought process and align with messages.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import extract_conversation_turns
from tp2dg.prompts.prompt_template import PromptTemplate

thought_evaluation_prompt = PromptTemplate(
    name="thought_evaluation",
    version="v1.0",
    description="Evaluate thought coherence and helpfulness for both Listener and Recsys in a single call",
    required_params=[
        "conversation_turns",
    ],
    response_expected_fields=[
        "listener_coherence_score",
        "recsys_coherence_score",
    ],
    template="""
You are an expert evaluator of music recommendation systems. Assess thought coherence and helpfulness for both Listener and Recsys.

## CONVERSATION TURNS:
{conversation_turns}

## EVALUATION TASKS:

### 1. Listener Thought Coherence and Helpfulness: `listener_coherence_score`
Rate if the Listener thoughts explain the actual thought process and align with their queries and responses.

**Consider:**
- Coherence: Do the thoughts follow logical reasoning? Are they internally consistent?
- Alignment: Do the thoughts align with the actual messages and responses? Do they explain the reasoning behind the listener's choices?
- Helpfulness: Do the thoughts provide useful insight into the decision-making process? Do they show understanding of the conversation context?
- Consistency: Are the thoughts consistent throughout the conversation? Do they maintain the same personality and preferences?

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Thoughts clearly explain the reasoning and perfectly align with messages
- **3**: Good - Thoughts are coherent and mostly align with messages
- **2**: Fair - Thoughts have some coherence issues or misalignment with messages
- **1**: Poor - Thoughts are incoherent or don't align with messages

### 2. Recsys Thought Coherence and Helpfulness: `recsys_coherence_score`
Rate if the Recsys thoughts explain the actual thought process and align with their queries and responses.

**Consider:**
- Coherence: Do the thoughts follow logical reasoning for recommendations? Are they internally consistent?
- Alignment: Do the thoughts align with the actual recommendation messages? Do they explain the reasoning behind track selections?
- Helpfulness: Do the thoughts provide useful insight into the recommendation process? Do they show understanding of the listener's preferences?
- Consistency: Are the thoughts consistent throughout the conversation? Do they maintain the same recommendation strategy?

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Thoughts clearly explain the reasoning and perfectly align with messages
- **3**: Good - Thoughts are coherent and mostly align with messages
- **2**: Fair - Thoughts have some coherence issues or misalignment with messages
- **1**: Poor - Thoughts are incoherent or don't align with messages

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
listener_coherence_score: [1-4]
recsys_coherence_score: [1-4]
```
""",
)


class ThoughtEvaluator(BaseEvaluationTemplate):
    """Evaluates thought coherence and helpfulness for both Listener and Recsys in a single call."""

    def __init__(self):
        super().__init__()
        self.evaluation_name = "thought"
        self.prompt_template = thought_evaluation_prompt

    def prepare_prompt_data(
        self,
        chat_json: dict[str, Any],
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Prepare data for the thought evaluation prompt."""

        # Extract conversation turns using utility function
        conversation_turns = extract_conversation_turns(chat_json)
        turns_text = json.dumps(conversation_turns)

        return {
            "conversation_turns": turns_text,
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

            # Extract both scores
            listener_coherence_score = parsed["listener_coherence_score"]
            recsys_coherence_score = parsed["recsys_coherence_score"]

            # Convert to int if needed
            if isinstance(listener_coherence_score, str):
                listener_coherence_score = int(listener_coherence_score)
            if isinstance(recsys_coherence_score, str):
                recsys_coherence_score = int(recsys_coherence_score)

            # Check if all scores are valid
            all_scores_valid = all(0 < score <= 4 for score in [listener_coherence_score, recsys_coherence_score])

            return {
                "listener_coherence_score": listener_coherence_score,
                "recsys_coherence_score": recsys_coherence_score,
                "success": all_scores_valid,
                "raw_response": response,
                "parsed_response": parsed,
            }

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {
                "listener_coherence_score": 0,
                "recsys_coherence_score": 0,
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
                "metrics": {
                    "listener_coherence": {"average_score": 0.0, "score_distribution": {}},
                    "recsys_coherence": {"average_score": 0.0, "score_distribution": {}},
                },
            }

        # Extract all scores
        listener_coherence_scores = [r["listener_coherence_score"] for r in successful_results]
        recsys_coherence_scores = [r["recsys_coherence_score"] for r in successful_results]

        # Calculate distributions
        listener_coherence_dist = {i: listener_coherence_scores.count(i) for i in range(1, 5)}
        recsys_coherence_dist = {i: recsys_coherence_scores.count(i) for i in range(1, 5)}

        return {
            "total_conversations": len(individual_results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results),
            "metrics": {
                "listener_coherence": {
                    "average_score": sum(listener_coherence_scores) / len(listener_coherence_scores),
                    "score_distribution": listener_coherence_dist,
                    "scores": listener_coherence_scores,
                },
                "recsys_coherence": {
                    "average_score": sum(recsys_coherence_scores) / len(recsys_coherence_scores),
                    "score_distribution": recsys_coherence_dist,
                    "scores": recsys_coherence_scores,
                },
            },
        }


# Create the evaluator instance for easy import
thought_evaluator = ThoughtEvaluator()
 