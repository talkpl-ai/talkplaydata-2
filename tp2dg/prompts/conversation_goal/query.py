from tp2dg.prompts.prompt_template import PromptTemplate

conversation_goal_query_pt1 = PromptTemplate(
	name="conversation_goal_query_pt1",
	version="v1.0",
	description="Instruction for sampling conversation goals.",
	required_params=["number_of_conversation_goals"],
	response_expected_fields=[],
	template="""
# Conversation Goal Inference (Part 1)
You will be given a set of tracks with metadata and artifacts. You will propose {number_of_conversation_goals} potential conversation goals and later select one.
""",
)

conversation_goal_query_pt2 = PromptTemplate(
	name="conversation_goal_query_pt2",
	version="v1.0",
	description="Instruction to choose and output a single goal in YAML.",
	required_params=["conversation_goal_templates"],
	response_expected_fields=[
		"category_code",
		"category_description",
		"specificity_code",
		"specificity_description",
		"listener_goal",
		"listener_expertise",
		"initial_query_example_1",
		"initial_query_example_2",
		"iteration_query_example_1",
		"iteration_query_example_2",
		"achieved_query_example_1",
		"achieved_query_example_2",
		"target_turn_count",
	],
	template="""
# Conversation Goal Inference (Part 2)

Here are the allowed conversation goals:
{conversation_goal_templates}

Select one and output in YAML with keys:
- category_code
- category_description
- specificity_code
- specificity_description
- listener_goal
- listener_expertise
- initial_query_example_1
- initial_query_example_2
- iteration_query_example_1
- iteration_query_example_2
- achieved_query_example_1
- achieved_query_example_2
- target_turn_count
""",
) 