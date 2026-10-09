# Dodona

Ask questions about speech, get calibrated probabilities.

```python
from dodona import Dodona

model = Dodona()
result = model.ask("call.wav", {
    "angry":  {},                                   # a catalogue question
    "refund": {"type": "noul", "word": "refund"},   # was this word said?
    "intent": {"type": "choice", "instructions": "What does the caller want?",
               "criteria": {"refund": "Money back", "help": "Technical help", "other": "Something else"}},
})
result["answers"]["angry"]
# {"type": "noul", "noul": 0.82, "where": [20.0, 30.0], "status": "calibrated"}
```

Dodona answers yes/no, multiple-choice and scale questions about a recording — *is the speaker
angry? was "refund" said? how many people are talking?* — and its probabilities mean what they
say: when it answers 80%, it is right about 80% of the time. It reads the audio once and answers
every question on top of it, so many questions cost little more than one.

It is two LoRA adapters on [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
and a calibration layer. The questions use [Jev](https://flaviocopes.com/jev/)'s format
(`noul`, `choice`, `score`), with audio in place of text.

- **Model, results and limitations:** the model card on Hugging Face,
  [bahadobay/dodona-audio](https://huggingface.co/bahadobay/dodona-audio)
- **Inputs and outputs, step by step:** [EXAMPLES.md](EXAMPLES.md)

## Install

```bash
pip install dodona-audio
```

Needs a CUDA GPU. The first `Dodona()` downloads the adapters (~135 MB) and Qwen2-Audio-7B (~16 GB).

## Asking

Each question has a name you choose and one of three types:

| type | you give | you get |
|---|---|---|
| `noul` (yes/no) | `instructions`, or `word` | `noul`: probability of yes |
| `choice` | `instructions`, `criteria` {name: description} | `choice` and `probabilities` |
| `score` | `instructions`, `criteria` [lowest, ..., highest] | `score` (expected level), `probabilities`, `legend` |

`model.questions()` lists the catalogue: the 46 questions Dodona was trained and calibrated on,
asked by name (`"angry": {}`). Any word can be asked with `{"type": "noul", "word": ...}`.
Both are **calibrated**. Questions in your own words are answered too, marked **off catalogue**:
useful, but their probabilities are not verified.

Long audio is read in 10-second windows. A yes/no answer is the highest window, with `where` it
is; choice and score answers are the average. `result["windows"]` has every window's answers.

## Use it from Claude (MCP)

```bash
pip install "dodona-audio[mcp]"
```

Add to your MCP client's configuration (for Claude Desktop, `claude_desktop_config.json`):

```json
{"mcpServers": {"dodona": {"command": "dodona-mcp"}}}
```

The assistant gets two tools: `list_questions` and `ask` (an audio file path and questions in the
format above).

## Licence

Apache-2.0 (code, adapters, calibration). The base model is Apache-2.0. Training data
attributions are in [NOTICE](NOTICE).
