"""
Utility functions for conversation evaluation.
"""

from typing import Any

from tp2dg.entities.track import Track, Tracks


def extract_conversation_turns(chat_json: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract conversation turns from chat.json data structure.

    Args:
        chat_json: The chat.json data containing conversation information

    Returns:
        List of conversation turns with thought, message, and track_id information
    """
    conversation = chat_json["conversation_turns"]
    turns = []

    for turn_data in conversation:
        turn_number = turn_data["turn_number"]

        # Extract listener turn data
        listener_turn = turn_data["listener_turn"]
        listener_thought = listener_turn["thought"]
        listener_message = listener_turn["message"]

        # Extract recsys turn data
        recsys_turn = turn_data["recsys_turn"]
        recsys_thought = recsys_turn["thought"]
        recsys_message = recsys_turn["message"]
        recsys_track = recsys_turn["track"]

        # Create turn object
        turn = {
            "turn_number": turn_number,
            "listener": {
                "thought": listener_thought,
                "message": listener_message,
            },
            "recsys": {
                "thought": recsys_thought,
                "message": recsys_message,
                "track": recsys_track,
            },
        }

        turns.append(turn)

    return turns


def extract_goal_progress_assessments(chat_json: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract goal progress assessments from listener turns.

    Args:
        chat_json: The chat.json data containing conversation information

    Returns:
        List of goal progress assessments from each listener turn
    """
    conversation = chat_json["conversation_turns"]
    assessments = []

    for turn_data in conversation:
        listener_turn = turn_data["listener_turn"]
        assessment = listener_turn["goal_progress_assessment"]

        assessments.append(
            {
                "turn_number": turn_data.get("turn_number", 0),
                "assessment": assessment,
            },
        )

    return assessments


def get_recommended_tracks_content(
    chat_json: dict[str, Any],
    uploaded_audio_files: dict[str, Any],
    uploaded_image_files: dict[str, Any],
) -> str:
    session_context = chat_json["session_context"]
    recommended_tracks_data = session_context["recommendation_pool_tracks"]
    recommended_tracks = Tracks([Track.from_dict(track) for track in recommended_tracks_data])

    recommended_tracks_content = recommended_tracks.prompt_str_with_artifacts(
        tracks_title="## RECOMMENDED TRACKS:\n",
        tracks_artifacts={
            "audio": uploaded_audio_files,
            "image": uploaded_image_files,
        },
        include_track_id=False,
        lyric_chars=100,
        n_tags=10,
    )

    return recommended_tracks_content


def get_profiling_tracks_content(
    chat_json: dict[str, Any],
    uploaded_audio_files: dict[str, Any],
    uploaded_image_files: dict[str, Any],
) -> str:
    session_context = chat_json["session_context"]
    profiling_tracks_data = session_context["listener_tracks"]
    profiling_tracks = Tracks([Track.from_dict(track) for track in profiling_tracks_data])

    profiling_tracks_content = profiling_tracks.prompt_str_with_artifacts(
        tracks_title="## PROFILING TRACKS (Previously liked by listener):\n",
        tracks_artifacts={
            "audio": uploaded_audio_files,
            "image": uploaded_image_files,
        },
        include_track_id=False,
        lyric_chars=100,
        n_tags=10,
    )

    return profiling_tracks_content
