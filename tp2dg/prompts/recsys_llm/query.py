from tp2dg.prompts.prompt_template import PromptTemplate

recsys_following_turns = PromptTemplate(
	name="recsys_following_turns",
	version="v1.0",
	description="Default prompt for RecSys LLM on subsequent turns.",
	required_params=["turn_num", "used_track_ids", "listener_message", "preferred_language"],
	response_expected_fields=["thought", "track_id", "message"],
	template="""
## Turn {turn_num}

### CONVERSATION CONTEXT:
**Previously Recommended Tracks** (DO NOT recommend these again): {used_track_ids}
**Listener's Latest Message**: "{listener_message}"

### CRITICAL REQUIREMENTS:
1. **NO DUPLICATES**: You MUST NOT recommend any track from the "Previously Recommended Tracks" list above
2. **TRACK POOL ONLY**: You can ONLY recommend from your available tracks pool
3. **EXACT FORMAT**: Your response must be ONLY the YAML block below - no additional text
4. **EXACT TRACK ID**: Use the precise track_id from your available tracks

**REMEMBER**: Maintain conversation coherence and respond naturally to the listener's feedback while strictly following the format.


### MANDATORY RESPONSE FORMAT:

**CRITICAL FORMATTING RULES**:
- Your ENTIRE response = ONLY the YAML block above
- Do NOT add any text before the YAML block
- Do NOT add any text after the YAML block
- Do NOT use markdown or other formatting outside the YAML
- The track_id MUST be an exact match from your available tracks
- The track_id MUST NOT appear in the used tracks list

Please use {preferred_language} language for thought and message.
### Response Format:
```yaml
thought: [CONCISE reasoning in 2-3 sentences maximum. Briefly acknowledge listener feedback and explain why your chosen track fits their request. DO NOT analyze multiple tracks.]
track_id: [The exact track_id for your next recommendation - must NOT be in the used tracks list]
message: [A natural, conversational response that acknowledges their feedback and introduces the next track.]
```

**IMPORTANT**: Keep the "thought" field BRIEF and FOCUSED. Do not analyze multiple tracks or provide extensive comparisons. Simply state why your chosen track matches the listener's request.
""",
) 