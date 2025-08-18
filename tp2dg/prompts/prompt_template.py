from dataclasses import dataclass


@dataclass
class PromptTemplate:
	"""A template for prompts with versioning and parameter substitution."""

	name: str
	version: str
	template: str
	description: str
	required_params: list[str]
	response_expected_fields: list[str]

	def __post_init__(self):
		if self.required_params is None:
			self.required_params = []

		if self.response_expected_fields is None:
			self.response_expected_fields = []

	def format(self, **kwargs) -> str:
		"""Format the template with provided parameters."""
		missing_params = [param for param in self.required_params if param not in kwargs]
		if missing_params:
			raise ValueError(f"Missing required parameters: {missing_params}")

		return self.template.format(**kwargs)

	def is_success(self, parsed_response: dict) -> bool:
		"""Check if the response is successful."""
		return all(key in parsed_response for key in self.response_expected_fields) 