from tp2dg.prompts.prompt_template import PromptTemplate

listener_first_turn = PromptTemplate(
	name="listener_first_turn",
	version="v1.0",
	description="Prompt for the listener's first turn, requiring an exact query selection.",
	required_params=["initial_query_examples", "listener_goal", "preferred_language"],
	response_expected_fields=["thought", "message"],
	template="""
## Turn 1
You are starting a new music discovery conversation. Your task is to make an initial request.

**CRITICAL INSTRUCTION**: Your `message` for this turn MUST be an exact, verbatim copy of ONE of the examples from the list below.

### Conversation Goal
{listener_goal}

### Initial Query Examples
{initial_query_examples}

### Response Format
Your response MUST follow the provided YAML format. The `message` field must be one of the examples above.
Please use {preferred_language} language for thought and message.

```yaml
thought: My goal is to [describe goal]. I will select one of the provided initial queries to start the conversation.
message: [Copy one of the initial query examples here, exactly as it is written.]
```
""",
)

reaction_turn2 = PromptTemplate(
	name="reaction_turn_2",
	version="v1.0",
	description="Prompt for the listener's reaction on the second turn.",
	required_params=["turn_num", "title", "artist", "album", "recsys_message", "preferred_language"],
	response_expected_fields=["thought", "goal_progress_assessment", "message"],
	template="""
## Turn {turn_num}

You just listened to this recommended track:
- Title: {title}
- Artist: {artist}
- Album: {album}

The recommendation system said: "{recsys_message}"

Your response MUST be strictly guided by your Conversation Goal and your Listener Profile. Assess if this track moves you closer to achieving your goal.

### Pacing Guidance for Turn 2
This is your second turn, so consider your conversation strategy:
- **If your goal has 1-3 target turn counts**: This recommendation should be close to your goal, be specific about what you want
- **If your goal has 4-6 target turn counts**: This is early-mid conversation, provide clear feedback and moderate refinement
- **If your goal has 6-8 target turn counts**: This is early conversation, be more exploratory and open to discovery

### Response Format
Your response MUST be ONLY a single yaml block with three fields: `thought`, `goal_progress_assessment`, `message`. Maintain coherence with the chat history.
Please use {preferred_language} language for thought and message.

```yaml
thought: [Your internal reaction to the track. Does this align with my specific goal? How should I adjust my strategy for the remaining turns?]
goal_progress_assessment: [MOVES_TOWARD_GOAL or DOES_NOT_MOVE_TOWARD_GOAL, based on whether this recommendation moves you closer to achieving your Conversation Goal.]
message: [Your concise conversational response and next request. This should be generated based on your thought process and be aimed at steering the recommendations closer to your goal.]
```
""",
)

reaction_turn_n = PromptTemplate(
	name="reaction_turn_n",
	version="v1.0",
	description="Prompt for the listener's reaction on subsequent turns.",
	required_params=["turn_num", "title", "artist", "album", "recsys_message", "preferred_language"],
	response_expected_fields=["thought", "goal_progress_assessment", "message"],
	template="""
## Turn {turn_num}

You just listened to this recommended track:
- Title: {title}
- Artist: {artist}
- Album: {album}

The recommendation system said: "{recsys_message}"

Your response MUST be strictly guided by your Conversation Goal and your Listener Profile. Assess if this track moves you closer to achieving your goal.

### Pacing Guidance
Consider your target turn count and current turn number:
- **If this is an early turn in a long conversation**: Be more exploratory, ask follow-up questions, show curiosity
- **If this is near your target turn count**: Be more decisive, show clear satisfaction or dissatisfaction
- **If you've exceeded your target turn count**: The goal should be achieved or nearly achieved by now

### Response Format
Your response MUST be ONLY a single yaml block with three fields: `thought`, `goal_progress_assessment`, `message`. Maintain coherence with the chat history.
Please use {preferred_language} language for thought and message.

```yaml
thought: [Your internal reaction to the track. Consider: Does this align with my goal? How many turns do I have left? Am I making progress at the right pace?]
goal_progress_assessment: [MOVES_TOWARD_GOAL or DOES_NOT_MOVE_TOWARD_GOAL, based on whether this recommendation moves you closer to achieving your Conversation Goal.]
message: [Your conversational response and next request. Adjust specificity based on remaining turns and goal progress.]
```
""",
) 