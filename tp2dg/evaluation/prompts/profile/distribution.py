"""
Profile Distribution Evaluation.

This module evaluates the distribution of profile attributes, particularly
preferred_musical_culture, to assess diversity and coverage.
"""

from collections import Counter
from typing import Any, Optional

from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate


class ProfileDistributionEvaluator(BaseEvaluationTemplate):
    """Evaluates profile distribution by computing statistics of profile attributes."""

    def __init__(self):
        super().__init__()
        self.evaluation_name = "profile_distribution"
        self.prompt_template = None  # No prompt template needed for computational evaluation

    def prepare_prompt_data(
        self,
        conversation_data: dict[str, Any],
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Prepare data for the profile distribution analysis.

        For computational evaluations, this just extracts the profile data.
        """
        listener_profile = conversation_data["listener_profile"]

        return {
            "listener_profile": listener_profile,
            "preferred_musical_culture": listener_profile.get("preferred_musical_culture", "Unknown"),
            "top_1_artist": listener_profile.get("top_1_artist", "Unknown"),
            "top_1_genre": listener_profile.get("top_1_genre", "Unknown"),
            "age_group": listener_profile.get("age_group", "Unknown"),
            "country": listener_profile.get("country", "Unknown"),
            "cultural_background": listener_profile.get("cultural_background", "Unknown"),
        }

    async def evaluate_single(
        self,
        conversation_data: dict[str, Any],
        llm_call_func,
        client,
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate a single conversation by extracting profile attributes.

        This is computational (no LLM call needed).
        """

        # Extract listener profile data
        listener_profile = conversation_data.get("listener_profile", {})

        preferred_musical_culture = listener_profile.get("preferred_musical_culture", "Unknown")
        top_1_artist = listener_profile.get("top_1_artist", "Unknown")
        top_1_genre = listener_profile.get("top_1_genre", "Unknown")
        age_group = listener_profile.get("age_group", "Unknown")
        country = listener_profile.get("country", "Unknown")
        cultural_background = listener_profile.get("cultural_background", "Unknown")

        return {
            "preferred_musical_culture": preferred_musical_culture,
            "top_1_artist": top_1_artist,
            "top_1_genre": top_1_genre,
            "age_group": age_group,
            "country": country,
            "cultural_background": cultural_background,
            "success": True,
        }

    def aggregate_results(self, individual_results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate individual results into distribution statistics and histograms.

        As specified in PLAN.md: "Show the histogram / show the top 10 preferred_musical_culture"
        """

        successful_results = [r for r in individual_results if r.get("success", False)]

        if not successful_results:
            return {
                "total_conversations": len(individual_results),
                "successful_evaluations": 0,
                "success_rate": 0.0,
                "preferred_musical_culture_distribution": {},
                "top_1_artist_distribution": {},
                "top_1_genre_distribution": {},
                "age_group_distribution": {},
                "country_distribution": {},
                "cultural_background_distribution": {},
            }

        # Extract data for analysis
        preferred_musical_cultures = [r["preferred_musical_culture"] for r in successful_results]
        top_1_artists = [r["top_1_artist"] for r in successful_results]
        top_1_genres = [r["top_1_genre"] for r in successful_results]
        age_groups = [r["age_group"] for r in successful_results]
        countries = [r["country"] for r in successful_results]
        cultural_backgrounds = [r["cultural_background"] for r in successful_results]

        # Compute distributions (counts and percentages)
        total_count = len(successful_results)

        def create_distribution(values):
            counts = Counter(values)
            return {
                value: {
                    "count": count,
                    "percentage": (count / total_count) * 100,
                }
                for value, count in counts.items()
            }

        # Create distributions for all profile attributes
        preferred_musical_culture_distribution = create_distribution(preferred_musical_cultures)
        top_1_artist_distribution = create_distribution(top_1_artists)
        top_1_genre_distribution = create_distribution(top_1_genres)
        age_group_distribution = create_distribution(age_groups)
        country_distribution = create_distribution(countries)
        cultural_background_distribution = create_distribution(cultural_backgrounds)

        # Compute diversity metrics (number of unique values)
        diversity_metrics = {
            "preferred_musical_culture_diversity": len(set(preferred_musical_cultures)),
            "top_1_artist_diversity": len(set(top_1_artists)),
            "top_1_genre_diversity": len(set(top_1_genres)),
            "age_group_diversity": len(set(age_groups)),
            "country_diversity": len(set(countries)),
            "cultural_background_diversity": len(set(cultural_backgrounds)),
        }

        # Get top 10 most common values for each attribute
        def get_top_10(values):
            counts = Counter(values)
            return counts.most_common(10)

        top_10_metrics = {
            "top_10_preferred_musical_cultures": get_top_10(preferred_musical_cultures),
            "top_10_top_1_artists": get_top_10(top_1_artists),
            "top_10_top_1_genres": get_top_10(top_1_genres),
            "top_10_age_groups": get_top_10(age_groups),
            "top_10_countries": get_top_10(countries),
            "top_10_cultural_backgrounds": get_top_10(cultural_backgrounds),
        }

        return {
            "total_conversations": len(individual_results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results),
            "preferred_musical_culture_distribution": preferred_musical_culture_distribution,
            "top_1_artist_distribution": top_1_artist_distribution,
            "top_1_genre_distribution": top_1_genre_distribution,
            "age_group_distribution": age_group_distribution,
            "country_distribution": country_distribution,
            "cultural_background_distribution": cultural_background_distribution,
            "diversity_metrics": diversity_metrics,
            "top_10_metrics": top_10_metrics,
            "raw_data": {
                "preferred_musical_cultures": preferred_musical_cultures,
                "top_1_artists": top_1_artists,
                "top_1_genres": top_1_genres,
                "age_groups": age_groups,
                "countries": countries,
                "cultural_backgrounds": cultural_backgrounds,
            },
        }


# Create the evaluator instance for easy import
profile_distribution_evaluator = ProfileDistributionEvaluator()
