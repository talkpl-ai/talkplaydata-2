from dataclasses import dataclass


@dataclass
class TokenUsage:
	input_text_tokens: int = 0
	input_image_tokens: int = 0
	input_audio_tokens: int = 0
	output_tokens: int = 0
	prompt_details: list | None = None
	response_details: list | None = None

	def to_dict(self) -> dict:
		return {
			"input_text_tokens": self.input_text_tokens,
			"input_image_tokens": self.input_image_tokens,
			"input_audio_tokens": self.input_audio_tokens,
			"output_tokens": self.output_tokens,
			"prompt_details": self.prompt_details or [],
			"response_details": self.response_details or [],
		} 