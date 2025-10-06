# TalkPlayData 2: An Agentic Synthetic Data Pipeline for Trimodal and Conversational Music Recommendation


[![arXiv](https://img.shields.io/badge/arXiv-2509.09685-blue.svg)](https://arxiv.org/abs/2509.09685)
[![Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)

*talk to AI to play music!*

By Keunwoo Choi and Seungheon Doh: Team [talkpl.ai](https://talkpl.ai); and Juhan Nam

## Data Generation
This repo shows an executable pipeline to: load small dummy data, prompt a Gemini model to simulate a conversational music recommendation session, and save the outputs. 

This repo was created as a public version of the data generation pipeline of TalkPlayData 2. All the prompts and logics are identical to what we used 

Quickstart:

1. Install: `pip install -e .`
2. Set API key: `export GEMINI_API_KEY=...`
3. Run demo: `tp2dg-generate`
4. See results in `generated_conversations/` and summarize with `tp2dg-summary --input generated_conversations`.

## Flowchart


```text
┌───────────────────────────────────────────────────────────────────────────────┐
│ CLI Entry                                                                     │
│  pyproject.toml → [project.scripts]                                           │
│    tp2dg-generate = "tp2dg.call_gemini:main"                                  │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ tp2dg/call_gemini.py                                                          │
│  - parse_args()                                                               │
│  - main():                                                                    │
│      sess = data.loader.get_first_session()                                   │
│      orch = ConversationOrchestrator(model=...)                               │
│      outputs = orch.generate(user=sess.user, liked=sess.liked_tracks,         │
│                               pool=sess.pool_tracks, num_turns=...)           │
│      orch.save_outputs(outputs, out_dir)                                      │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ tp2dg/data/loader.py                                                          │
│  - load_dummy(): reads dummy JSONs                                            │
│      • data/dummy/users.json                                                  │
│      • data/dummy/tracks.json                                                 │
│      • data/dummy/playlists.json                                              │
│  - _to_track(...) → entities.track.Track                                      │
│  - get_first_session(...) → SessionData(user, liked Tracks, pool Tracks)      │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ tp2dg/conversation_orchestrator.py                                            │
│  class ConversationOrchestrator                                               │
│   - __init__(model, ...):                                                     │
│       • google.genai.Client(api_key=GEMINI_API_KEY)                           │
│       • components.processors.AudioProcessor/ImageProcessor                   │
│       • components.ChatSessionManager                                         │
│       • components.ProfileLLM / ConversationGoalLLM / RecsysLLM / ListenerLLM │
│                                                                               │
│   - generate(user, liked: Tracks, pool: Tracks, num_turns):                   │
│       1) Upload artifacts                                                     │
│          • AudioProcessor.batch_upload_tracks(liked + pool)                   │
│              ↳ BaseFileProcessor.upload_file (client.files.upload)            │
│          • ImageProcessor.batch_upload_tracks(liked + pool)                   │
│       2) Create profile & goal                                                │
│          • ProfileLLM.generate_from_tracks(liked, audio, image)               │
│          • ConversationGoalLLM.generate_from_recommendation_pool(pool, ...)   │
│       3) Initialize chats                                                     │
│          • ChatSessionManager.initialize_recsys_session(...)                   │
│              ↳ prompts.recsys_llm.system.*                                    │
│          • ChatSessionManager.initialize_listener_session(...)                 │
│              ↳ prompts.listener_llm.system.*                                  │
│          • RecsysLLM.set_chat_session(chat), ListenerLLM.set_chat_session(chat)│
│       4) Listener starts                                                      │
│          • ListenerLLM.get_initial_request(initial_query_examples, ...)       │
│              ↳ prompts.listener_llm.query.listener_first_turn                 │
│       5) Conversation loop for turn = 1..N                                    │
│          a. Recsys recommends                                                 │
│             • RecsysLLM.get_recommendation_with_thought(...) → RecsysTurn     │
│                 ↳ prompts.recsys_llm.query.recsys_following_turns             │
│          b. Append ConversationTurn(turn, listener_turn, recsys_turn)         │
│          c. Remove used track from available                                  │
│          d. If more turns and track exists                                    │
│             • ListenerLLM.get_reaction_with_thought(turn+1, track, ...)       │
│                 ↳ prompts.listener_llm.query.reaction_turn2 / reaction_turn_n │
│       6) Return outputs dict:                                                 │
│          { profiling, conversation_goal, chat (list of turns) }               │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Persist outputs                                                                │
│  tp2dg/conversation_orchestrator.py → save_outputs(outputs, out_dir)           │
│   - profiling.json                                                             │
│   - conversation_goal.json                                                     │
│   - chat.json                                                                  │
└───────────────────────────────────────────────────────────────────────────────┘

Notes
- Artifacts: Track audio/image paths come from dummy data and are uploaded via the
  shared google.genai client before prompting. These files may be referenced in
  prompts sent to the model.
- Entities: Conversation is recorded using `entities.turns.{ListenerTurn, RecsysTurn,
  ConversationTurn, ConversationTurns}`; tracks via `entities.track.{Track, Tracks}`.
```


## Dataset
TalkPlayData 2 is available on [HuggingFace](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)

## Paper
[arxiv:2509.09685](https://arxiv.org/abs/2509.09685)

```
@misc{choi2025talkplaydata2agenticsynthetic,
      title={TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation}, 
      author={Keunwoo Choi and Seungheon Doh and Juhan Nam},
      year={2025},
      eprint={2509.09685},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2509.09685}, 
}
```
