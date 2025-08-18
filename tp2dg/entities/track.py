import os
from collections import UserList
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Track:
	track_id: str
	title: str
	artist: str
	album: Optional[str] = None
	lyrics: Optional[str] = None
	audio_path: Optional[str] = None
	image_path: Optional[str] = None
	tags: list[str] = None

	def get_artifact_path(self, modality: str, base_path: str) -> str:
		if modality == "audio":
			p = self.audio_path or ""
		elif modality == "image":
			p = self.image_path or ""
		else:
			raise NotImplementedError(f"Modality {modality} not implemented")
		return p if os.path.isabs(p) else os.path.join(base_path, p)

	def prompt_str(self, include_track_id: bool, title="### TRACK:\n", lyric_chars: int = 100, n_tags: int = 10) -> str:
		track_id_str = f" track_id: {self.track_id}\n" if include_track_id else "\n"
		lyrics = ""
		if lyric_chars and self.lyrics:
			dotdotdot = "..." if len(self.lyrics) > lyric_chars else ""
			lyrics = f"\nLyrics: --- Begin of Lyrics ---\n{self.lyrics[:lyric_chars]}{dotdotdot}--- End of Lyrics ---\n"
		tags = ", ".join((self.tags or [])[: n_tags if n_tags is not None else len(self.tags or [])])
		album = self.album or "Unknown"
		return f"""{title}- Title: {self.title} {track_id_str}- Artist: {self.artist}
- Album: {album}
- Tags: {tags}{lyrics}
"""

	def prompt_str_with_artifacts(
		self,
		include_track_id: bool,
		title="### TRACK:\n",
		lyric_chars=200,
		track_artifacts: dict[str, Any] | None = None,
		n_tags=None,
	) -> list[Any]:
		contents = [self.prompt_str(include_track_id=include_track_id, title=title, lyric_chars=lyric_chars, n_tags=n_tags)]
		track_artifacts = track_artifacts or {}
		for modality, file_object in track_artifacts.items():
			contents.append(f"- {modality}: ")
			contents.append(file_object)
		return contents

	@staticmethod
	def from_dict(data: dict) -> "Track":
		return Track(
			track_id=data.get("track_id", ""),
			title=data.get("title", ""),
			artist=data.get("artist", ""),
			album=data.get("album"),
			lyrics=data.get("lyrics"),
			audio_path=data.get("audio_path"),
			image_path=data.get("image_path"),
			tags=data.get("tags", []),
		)

	def to_dict(self) -> dict:
		return {
			"track_id": self.track_id,
			"title": self.title,
			"artist": self.artist,
			"album": self.album,
			"lyrics": self.lyrics,
			"audio_path": self.audio_path,
			"image_path": self.image_path,
			"tags": self.tags or [],
		}


class Tracks(UserList):
	def prompt_str(self, tracks_title="## TRACKS\n", **kwargs) -> str:
		return tracks_title + "\n".join([track.prompt_str(**kwargs) for track in self])

	def prompt_str_with_artifacts(
		self,
		tracks_title="## TRACKS\n",
		tracks_artifacts: dict[str, dict[str, Any]] | None = None,
		**kwargs,
	) -> list[Any]:
		if tracks_artifacts is None:
			tracks_artifacts = {}
		contents = [tracks_title]
		for track in self:
			track_artifacts = {}
			for modality, uploaded_files in tracks_artifacts.items():
				if track.track_id in uploaded_files:
					track_artifacts[modality] = uploaded_files[track.track_id]
			track_kwargs = dict(kwargs)
			track_kwargs["track_artifacts"] = track_artifacts
			contents.extend(track.prompt_str_with_artifacts(**track_kwargs))
		return contents 