from dataclasses import dataclass


@dataclass
class ListenerProfile:
	age_group: str
	country: str
	gender: str
	preferred_musical_culture: str
	preferred_language: str
	top_1_artist: str
	top_1_genre: str
	success: bool = True
	code: str = "SUCCESS"

	def prompt_str(self, title: str = "## Listener Profile\n\n") -> str:
		return (
			f"{title}"
			f"- age_group: {self.age_group}\n"
			f"- country: {self.country}\n"
			f"- gender: {self.gender}\n"
			f"- preferred_musical_culture: {self.preferred_musical_culture}\n"
			f"- preferred_language: {self.preferred_language}\n"
			f"- top_1_artist: {self.top_1_artist}\n"
			f"- top_1_genre: {self.top_1_genre}\n"
		) 