import os
from typing import Any, Optional

from google import genai

from tp2dg.components.base import BaseComponent
from tp2dg.components.utils import call_with_timeout
from tp2dg.entities.track import Tracks


class BaseFileProcessor(BaseComponent):
	def __init__(self, client: genai.Client, base_path: str, modality: str):
		super().__init__()
		self.client = client
		self.base_path = base_path
		self.modality = modality
		self._conversation_cache = {}

	def get_cache_key(self, track_id: str) -> str:
		return f"{track_id}-{self.modality}"

	def get_uploaded_file(self, track_id: str) -> Optional[Any]:
		return self._conversation_cache.get(self.get_cache_key(track_id), None)

	def prepare_file(self, file_path: str) -> str:
		return file_path

	def upload_file(self, file_path: str, track_id: str) -> Any:
		cache_key = self.get_cache_key(track_id)
		if cache_key in self._conversation_cache:
			return self._conversation_cache[cache_key]
		uploaded_file = call_with_timeout(lambda: self.client.files.upload(file=file_path), timeout=60)
		self._conversation_cache[cache_key] = uploaded_file
		return uploaded_file

	def batch_upload_tracks(self, tracks: Tracks) -> dict[str, Any]:
		track_files = {}
		for track in tracks:
			path = track.get_artifact_path(self.modality, self.base_path)
			if os.path.exists(path):
				uploaded = self.upload_file(path, track.track_id)
				if uploaded:
					track_files[track.track_id] = uploaded
		return track_files


class AudioProcessor(BaseFileProcessor):
	def __init__(self, client: genai.Client, audio_base_path: str, snippet_duration: float = 3.0):
		super().__init__(client, audio_base_path, "audio")
		self.snippet_duration = snippet_duration


class ImageProcessor(BaseFileProcessor):
	def __init__(self, client: genai.Client, image_base_path: str):
		super().__init__(client, image_base_path, "image") 