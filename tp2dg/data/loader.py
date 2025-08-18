import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tp2dg.entities.track import Track, Tracks


DATA_DIR = os.path.join(os.path.dirname(__file__), "dummy")


@dataclass
class SessionData:
	user: Dict
	liked_tracks: Tracks
	pool_tracks: Tracks
	session_id: str


def _read_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def load_dummy() -> Tuple[Dict, Dict, List[Dict]]:
	users = _read_json(os.path.join(DATA_DIR, "users.json"))
	tracks = _read_json(os.path.join(DATA_DIR, "tracks.json"))
	sessions = _read_json(os.path.join(DATA_DIR, "playlists.json"))
	users_by_id = {u["user_id"]: u for u in users["users"]}
	tracks_by_id = {t["track_id"]: t for t in tracks["tracks"]}
	return users_by_id, tracks_by_id, sessions["sessions"]


def _to_track(d: Dict) -> Track:
	return Track(
		track_id=d["track_id"],
		title=d["title"],
		artist=d["artist"],
		album=d.get("album", "Unknown"),
		audio_path=d.get("audio_path"),
		image_path=d.get("image_path"),
		tags=d.get("tags", []),
	)


def get_first_session(profile_size: int = 3, pool_size: int = 8) -> SessionData:
	users_by_id, tracks_by_id, sessions = load_dummy()
	if not sessions:
		raise RuntimeError("No dummy sessions available")
	sess = sessions[0]
	user = users_by_id[sess["user_id"]]
	track_ids = [tid for tid in sess["track_ids"] if tid in tracks_by_id]
	liked = Tracks([_to_track(tracks_by_id[tid]) for tid in track_ids[:profile_size]])
	pool = Tracks([_to_track(tracks_by_id[tid]) for tid in track_ids[profile_size: profile_size + pool_size]])
	return SessionData(user=user, liked_tracks=liked, pool_tracks=pool, session_id=sess["session_id"]) 