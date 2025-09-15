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
