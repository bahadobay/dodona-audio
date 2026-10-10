# Dodona

**A voice-native decision model.**

Ask your audio about what was said, how it was said, and who said it. Every answer comes with
a probability you can trust.

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-bahadobay%2Fdodona--audio-yellow)](https://huggingface.co/bahadobay/dodona-audio)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

> At Dodona, the oracle listened to the rustling of an oak and answered the questions brought
> to it.

## What you send, what you get

Two minutes of a real meeting (AMI corpus, ES2002a, 50 to 170 s), six questions:

```python
from dodona import Dodona

model = Dodona()
result = model.ask("meeting.wav", {
    "laughs":     {"type": "noul", "instructions": "Someone laughs in this recording."},
    # a built-in choice question brings its own options: one, two, three_or_more
    "n_speakers": {"type": "choice", "instructions": "How many distinct speakers are in this recording?"},
    "remote":     {"type": "noul", "word": "remote"},  # was this word said? (noul = yes/no)
    "disagree":   {"type": "noul", "instructions": "Does someone disagree with another person?"},
    "topic":      {"type": "choice", "instructions": "What is this meeting about?",
                   "criteria": {"design": "Designing a product", "money": "A budget or prices",
                                "other": "Something else"}},
    "energy":     {"type": "score", "instructions": "How lively is the conversation?",
                   "criteria": ["low", "mid", "high"]},   # score: options from lowest to highest
})
```

The real output (`result["answers"]`; one NVIDIA L40S, 12 windows):

```json
{
  "laughs":     {"type": "noul", "noul": 0.93, "where": [40.0, 50.0], "status": "calibrated"},
  "n_speakers": {"type": "choice", "choice": "one",
                 "probabilities": {"one": 0.43, "two": 0.34, "three_or_more": 0.23}, "status": "calibrated"},
  "remote":     {"type": "noul", "noul": 0.99, "where": [60.0, 70.0], "status": "calibrated"},
  "disagree":   {"type": "noul", "noul": 0.631, "where": [110.0, 120.0], "status": "off catalogue"},
  "topic":      {"type": "choice", "choice": "design",
                 "probabilities": {"design": 0.5, "money": 0.166, "other": 0.334}, "status": "off catalogue"},
  "energy":     {"type": "score", "score": 0.828, "probabilities": {"0": 0.426, "1": 0.32, "2": 0.254},
                 "legend": {"0": "low", "1": "mid", "2": "high"}, "status": "off catalogue"}
}
```

How to read it:
- **`laughs` 0.93, `where` [40, 50]:** Dodona hears laughter, most clearly 40 to 50 s into the piece.
  A yes/no answer is about its strongest 10 seconds, not the whole recording.
- **`remote` 0.99:** the word "remote" is said (the meeting designs a remote control).
- **`n_speakers` "one" (0.43):** choice and score answers are averaged over the 10-second
  windows, so this is about 10-second stretches, not the meeting: four people take part in it.
- **`score` 0.828:** the expected level, between 0 (low) and 2 (high); `legend` names them.
  Dodona reads the probability of each option (0.426 low, 0.32 mid, 0.254 high) and the score
  is their weighted average.
- **`status`:** the built-in questions (`laughs`, `n_speakers`) and words are `calibrated`, so
  0.93 means right about 93% of the time. Your own questions say `off catalogue`: calibrated
  by a general map, not one fitted to your question (measured below).
- **`result["windows"]`** has the same answers for every 10 seconds, plus how much Dodona's two
  adapters `disagree` (high means it is guessing).

## Why Dodona

- **It hears how, not only what.** Tone, emotion, who is speaking, laughter, hesitation,
  and whether a word was said, all from the audio itself, with no transcript in between.
- **Your questions, typed answers.** Write the question in plain language and pick its type
  (yes/no, a choice, a scale). The answer always has that shape, with a probability, and
  nothing to parse: no prompt engineering, no JSON that sometimes breaks. In our test, a plain
  audio LLM asked the usual way left 1.3% of its answers without a usable letter.
- **Many questions, one listen.** The recording is read once and every question is answered
  on top of it, by reading the probability of each answer (no text is generated). Forty
  questions take about four times as long as one, not forty times.
- **Probabilities you can act on.** Catalogue answers are calibrated question by question, so a
  threshold means what it says; your own questions use a general map. Each answer says which,
  and how much Dodona's two halves disagree (high disagreement: it is guessing).

Decision models that answer typed questions about text and images already exist. Dodona does it
for speech.

## Install

```bash
pip install git+https://github.com/bahadobay/dodona-audio
```

Needs a CUDA GPU. The first `Dodona()` downloads the adapters (~135 MB) and
[Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) (~16 GB).

## Asking

Name each question and give it one of three types:

| type | you give | you get |
|---|---|---|
| `noul` (yes/no) | `instructions` (or a `word`); optional `criteria` {"false": ..., "true": ...} to word the two answers | `noul`: probability of yes |
| `choice` | `instructions`, `criteria` {name: description} | `choice`, `probabilities` |
| `score` | `instructions`, `criteria` [lowest, ..., highest] | `score` (expected level), `probabilities`, `legend` |

A `score` is an ordered `choice`: Dodona reads the probability of each option, and `score` is
the expected level (0 is the first option). `choice` and `score` take 2 to 8 options. A
question that does not fit this table is refused with a message saying what is wrong.

- **Your own questions:** write them like the table says. They are marked `"off catalogue"`:
  calibrated by a general map (fitted on rewordings of trained questions), not one fitted to
  your question. Measured on yes/no questions: for concepts Dodona was trained on, your own
  wording works as well as the built-in question ("Is the speaker angry?": AUROC 0.982 against
  0.974); for new topics the ranking holds and a high yes is a little overconfident (shown 97%,
  right 89 to 94%). Trust it less when the two adapters disagree. Choice and score questions
  of your own use the same map, not measured.
- **Any word:** `{"type": "noul", "word": "refund"}`, calibrated for words never seen in training too.
- **Built-in questions:** `model.questions()` lists the 46 questions Dodona was trained and
  calibrated on. Ask one with its exact wording (`laughs` and `n_speakers` above are two; a
  choice then brings its own options), or by name alone as a shortcut: `"angry": {}`. Their
  answers are marked `calibrated`.

**Long audio** is read in 10-second windows. A yes/no answer is the most likely window, with
`where` it is; choice and score answers are averaged. `result["windows"]` holds every window's
answers, for a timeline. More examples: [EXAMPLES.md](EXAMPLES.md).

## How well it works

Same test windows, same questions; skill is mean AUROC (balanced accuracy for multiple choice).
This gives Qwen2-Audio its best chance: its answers are read the way Dodona reads its own, from
the probability of each answer letter, with its own calibration:

| | Qwen2-Audio alone | Dodona |
|---|---|---|
| emotion (acted) | 0.848 | **0.967** |
| tone: speaking rate, hesitation, speakers, overlap | 0.456 | **0.821** |
| content: topics, questions, numbers | 0.779 | **0.941** |
| sounds, dialogue acts, voices | 0.693 | **0.869** |
| was a word said | 0.891 | **0.961** |

Asked the usual way instead (each question its own prompt, the answer generated and its letter
parsed; balanced accuracy on a sample of 100 windows per corpus), the backbone falls further
behind: emotion 0.656 against Dodona's 0.946, tone 0.433 against 0.764, content 0.594 against
0.838, and 1.3% of its answers had no usable letter.

Against a transcript (Whisper large-v3-turbo): on par for whether a word was said (0.907
against 0.904); a transcript is more accurate on topics defined by keyword lists (the test's labels, which it
matches by construction). Dodona adds how and who, with
calibrated probabilities, in the same pass. Trained on English; on Turkish speech, voices and
words work as well as in English. Full numbers, calibration and limits: the
[model card](https://huggingface.co/bahadobay/dodona-audio).

## Use it from Claude (MCP)

```bash
pip install "dodona-audio[mcp] @ git+https://github.com/bahadobay/dodona-audio"
```

```json
{"mcpServers": {"dodona": {"command": "dodona-mcp"}}}
```

The assistant gets two tools: `list_questions` and `ask` (an audio file and questions as above).
The server runs Dodona on your own machine, so it needs a CUDA GPU (about 24 GB, see the
model card). A hosted version that needs no GPU is planned.

## Limits

Built by one person on one cloud GPU, from the openly licensed speech that exists: emotion is
learned from actors, few voices of children or older people, scales such as speaking rate are
weak. Do not use it to make decisions about people. Details in the
[model card](https://huggingface.co/bahadobay/dodona-audio).

## Licence

Apache-2.0: code, adapters and calibration. Training data attributions: [NOTICE](NOTICE).
