from tp2dg.prompts.prompt_template import PromptTemplate

profile_query = PromptTemplate(
	name="profile_query",
	version="v1.0",
	description="Direct prompt for LLM to infer listener demographics.",
	required_params=["age_group", "country", "gender", "preferred_language"],
	response_expected_fields=[
		"preferred_musical_culture",
		"top_1_artist",
		"top_1_genre",
	],
	template="""
## SYSTEM INSTRUCTIONS:
You are an expert in music and demographic analysis. Given the demographic profile below and tracks,
Please analyze the tracks and infer the most representative preferred_musical_culture, artist and genre that define this listener's taste.

Demographic Profile:
- age_group: {age_group}
- country: {country}
- gender: {gender}
- preferred_language: {preferred_language}

Provide your analysis in this EXACT format:

```yaml
preferred_musical_culture: [the most representative preferred_musical_culture from these tracks that defines this listener's taste]
top_1_artist: [the most representative artist from these tracks that defines this listener's taste]
top_1_genre: [the most representative genre from these tracks that defines this listener's taste]
```

Consider these factors in your analysis:
- preferred_musical_culture is strongly associated with music track selection
- top_1_artist is strongly associated with music track selection
- top_1_genre is strongly associated with music track selection
""",
) 