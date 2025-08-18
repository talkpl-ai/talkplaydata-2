import time

from google import genai


class BaseComponent:
	def __init__(self, api_delay: float = 0.0):
		self.api_delay = api_delay

	def switch_client(self, client: genai.Client):
		self.client = client

	def get_client(self) -> genai.Client:
		return self.client

	def wait_for_rate_limit(self):
		if self.api_delay > 0:
			time.sleep(self.api_delay) 