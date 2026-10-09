# Dodona — examples

> **The numbers below are illustrative**, written to show the shape of the output. They will be
> replaced by real outputs of the released model before publishing.

Dodona speaks [Jev](https://flaviocopes.com/jev/)'s question format, with audio instead of text:
you name each question, say its type (`noul` = yes/no, `choice`, `score`) and get one answer per
name. A question can also be a catalogue name (`"angry": {}`) or a word (`{"type": "noul", "word": "refund"}`):
those answers are **calibrated** — when Dodona says 80%, it is right about 80% of the time.
Questions in your own words are answered too, marked `"off catalogue"`.

```python
from dodona import Dodona
model = Dodona()
```

## 1. A customer call (45 seconds)

```python
result = model.ask("call.wav", {
    "angry":   {},                                           # catalogue question
    "refund":  {"type": "noul", "word": "refund"},           # was the word said?
    "intent":  {"type": "choice", "instructions": "What does the caller want?",
                "criteria": {"refund": "Money back", "help": "Technical help", "cancel": "To cancel",
                             "other": "Something else"}},
    "upset":   {"type": "score", "instructions": "How upset is the caller?",
                "criteria": ["calm", "annoyed", "very upset"]},
})
```

```json
{
  "model": "dodona-1.0",
  "duration": 45.0,
  "answers": {
    "angry":  {"type": "noul", "noul": 0.82, "where": [20.0, 30.0], "status": "calibrated"},
    "refund": {"type": "noul", "noul": 0.91, "where": [30.0, 40.0], "status": "calibrated"},
    "intent": {"type": "choice", "choice": "refund",
               "probabilities": {"refund": 0.64, "help": 0.12, "cancel": 0.17, "other": 0.07},
               "status": "off catalogue"},
    "upset":  {"type": "score", "score": 1.1,
               "probabilities": {"0": 0.31, "1": 0.27, "2": 0.42},
               "legend": {"0": "calm", "1": "annoyed", "2": "very upset"},
               "status": "off catalogue"}
  },
  "windows": [
    {"start": 0.0, "end": 10.0, "answers": {"angry": {"type": "noul", "noul": 0.04, "status": "calibrated",
                                                       "disagreement": 0.01}, "...": "..."}},
    "... one entry per 10 seconds ..."
  ]
}
```

How to read it:
- **`angry` 0.82, `where` [20, 30]**: the angriest 10 seconds of the call are 20–30 s, and there
  the speaker is angry with probability 0.82. It is a statement about those 10 seconds, not "82%
  of the call is angry".
- **`refund` 0.91**: the word "refund" (or "refunds", "refunded") is said, most clearly at 30–40 s.
- **`intent`, `upset`**: averaged over the whole call. They are your own questions, so they
  are marked `off catalogue`: useful, but their probabilities are not verified.
- **`windows`**: the same answers for every 10 seconds, to draw a timeline.

## 2. A short voice message (6 seconds)

Under 10 seconds there is one window, and the answers look exactly like Jev's.

```python
model.ask("message.wav", {
    "emotion":  {},
    "question": {"type": "noul", "instructions": "Is the person asking a question?"},
})
```

```json
{
  "model": "dodona-1.0",
  "duration": 6.0,
  "answers": {
    "emotion":  {"type": "choice", "choice": "happy",
                 "probabilities": {"angry": 0.02, "happy": 0.81, "sad": 0.03, "neutral": 0.11,
                                   "fear": 0.01, "disgust": 0.02},
                 "status": "calibrated"},
    "question": {"type": "noul", "noul": 0.12, "where": [0.0, 6.0], "status": "off catalogue"}
  },
  "windows": ["... one window ..."]
}
```

## 3. A meeting (2 minutes)

```python
model.ask("meeting.wav", {
    "laughs":     {},                                        # catalogue: someone laughs
    "n_speakers": {},                                        # catalogue: how many speakers (one / two / three or more)
    "budget":     {"type": "noul", "word": "budget"},
    "disagree":   {"type": "noul", "instructions": "Does someone disagree with another person?"},
})
```

```json
{
  "model": "dodona-1.0",
  "duration": 120.0,
  "answers": {
    "laughs":   {"type": "noul", "noul": 0.93, "where": [70.0, 80.0], "status": "calibrated"},
    "n_speakers": {"type": "choice", "choice": "three_or_more",
                 "probabilities": {"one": 0.08, "two": 0.21, "three_or_more": 0.71},
                 "status": "calibrated"},
    "budget":   {"type": "noul", "noul": 0.88, "where": [40.0, 50.0], "status": "calibrated"},
    "disagree": {"type": "noul", "noul": 0.57, "where": [90.0, 100.0], "status": "off catalogue"}
  },
  "windows": ["... twelve windows ..."]
}
```

## What can be asked by name

```python
model.questions()
# [{"name": "angry", "type": "noul", "asks": "The speaker sounds angry in this recording.",
#   "criteria": None, "status": "calibrated"}, ...]
```
