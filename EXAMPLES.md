# Dodona examples

Real outputs of the released model (one NVIDIA L40S), on test recordings Dodona never trained
on. The audio is not included here; each example names its file in the corpus it comes from.

```python
from dodona import Dodona
model = Dodona()
```

Each question has a name you choose and one of three types:
- `noul`: a yes/no question, answered with the probability of yes;
- `choice`: pick one of several options, answered with a probability for each;
- `score`: a choice whose options are ordered from lowest to highest. Dodona reads the
  probability of each option and `score` is the expected level (0 is the first option).

Write your own questions with `instructions` (and `criteria` for choice and score); they are
marked `"off catalogue"`: calibrated by a general map, not one fitted to your question (how
well it holds: the README's "Asking"). For the most common questions there are built-in
shortcuts: ask one by its name (`"angry": {}`) or ask for any word
(`{"type": "noul", "word": "refund"}`), and the answer is **calibrated** for that question:
when Dodona says 0.8, it is right about 80% of the time.

## 1. Two minutes of a meeting

AMI Meeting Corpus, `ES2002a.Mix-Headset.wav`, 50 to 170 s. Four people design a remote control.
A test meeting: none of it, and none of its speakers, was in training.

```python
result = model.ask("meeting.wav", {
    "laughs":     {"type": "noul", "instructions": "Someone laughs in this recording."},
    # a built-in choice question brings its own options: one, two, three_or_more
    "n_speakers": {"type": "choice", "instructions": "How many distinct speakers are in this recording?"},
    "remote":     {"type": "noul", "word": "remote"},
    "disagree":   {"type": "noul", "instructions": "Does someone disagree with another person?"},
    "topic":      {"type": "choice", "instructions": "What is this meeting about?",
                   "criteria": {"design": "Designing a product", "money": "A budget or prices",
                                "other": "Something else"}},
    "energy":     {"type": "score", "instructions": "How lively is the conversation?",
                   "criteria": ["low", "mid", "high"]},
})
```

```json
{
  "model": "dodona-1.0",
  "duration": 120.0,
  "answers": {
    "laughs":     {"type": "noul", "noul": 0.93, "where": [40.0, 50.0], "status": "calibrated"},
    "n_speakers": {"type": "choice", "choice": "one",
                   "probabilities": {"one": 0.43, "two": 0.34, "three_or_more": 0.23}, "status": "calibrated"},
    "remote":     {"type": "noul", "noul": 0.99, "where": [60.0, 70.0], "status": "calibrated"},
    "disagree":   {"type": "noul", "noul": 0.631, "where": [110.0, 120.0], "status": "off catalogue"},
    "topic":      {"type": "choice", "choice": "design",
                   "probabilities": {"design": 0.5, "money": 0.166, "other": 0.334}, "status": "off catalogue"},
    "energy":     {"type": "score", "score": 0.828, "probabilities": {"0": 0.426, "1": 0.32, "2": 0.254},
                   "legend": {"0": "low", "1": "mid", "2": "high"}, "status": "off catalogue"}
  },
  "windows": [
    {"start": 0.0, "end": 10.0, "answers": {
      "laughs":     {"type": "noul", "noul": 0.033, "status": "calibrated", "disagreement": 0.014},
      "n_speakers": {"type": "choice", "choice": "one",
                     "probabilities": {"one": 0.67, "two": 0.255, "three_or_more": 0.075},
                     "status": "calibrated", "disagreement": 0.036},
      "remote":     {"type": "noul", "noul": 0.01, "status": "calibrated", "disagreement": 0.014},
      "...": "..."}},
    "... 11 more windows ..."
  ]
}
```

How to read it:
- **A yes/no answer is about its strongest 10 seconds.** `laughs` 0.93 with `where` [40, 50]:
  laughter, most clearly 40 to 50 s into the piece. `remote` 0.99: the word is said, clearest
  at 60 to 70 s. In the first 10 seconds alone, neither happens (0.033, 0.01).
- **Choice and score answers are averages over the windows.** `n_speakers` says "one" (0.43)
  although four people take part: it describes 10-second stretches, not the meeting.
- **`score` is the expected level**, 0.828 on a scale from 0 (low) to 2 (high): the
  probabilities of the three options (0.426, 0.32, 0.254) weighted by their level.
- **`disagreement`**, in each window, is how far Dodona's two adapters are apart. Near 0: they
  agree. High: Dodona is guessing.

## 2. A short angry sentence

CREMA-D, `1001_ITH_ANG_XX.wav` (2.4 s), an actor reading a sentence angrily. A test actor,
never heard in training; CREMA-D's 12 sentences are read by every actor, so the sentence itself
was. Under 10 seconds there is one window, and its answers are the answers.

```python
model.ask("angry.wav", {
    # a built-in choice question brings its own options: angry, happy, sad, neutral, fear, disgust
    "emotion": {"type": "choice", "instructions": "Which emotion is the speaker expressing?"},
    "angry":   {"type": "noul", "instructions": "The speaker sounds angry in this recording."},
    "upset":   {"type": "score", "instructions": "How upset is the speaker?",
                "criteria": ["calm", "annoyed", "very upset"]},
})
```

```json
{
  "emotion": {"type": "choice", "choice": "angry",
              "probabilities": {"angry": 0.972, "happy": 0.009, "sad": 0.005, "neutral": 0.004,
                                "fear": 0.004, "disgust": 0.006}, "status": "calibrated"},
  "angry":   {"type": "noul", "noul": 0.99, "where": [0.0, 2.369], "status": "calibrated"},
  "upset":   {"type": "score", "score": 1.247, "probabilities": {"0": 0.143, "1": 0.466, "2": 0.39},
              "legend": {"0": "calm", "1": "annoyed", "2": "very upset"}, "status": "off catalogue"}
}
```

`emotion` is a `choice` (one of six emotions) and `angry` a `noul` (yes/no); both are built-in
questions and clear (`angry` 0.99, the two adapters' disagreement 0.0). `upset` is a `score`
question of our own, and it lands between "annoyed" and "very upset" (1.247). Its window shows
a disagreement of 0.528: the two adapters answer it differently, so read it as a guess.

## 3. A short happy sentence

CREMA-D, `1001_WSI_HAP_XX.wav` (2.3 s), the same actor, happily, same questions.

```json
{
  "emotion": {"type": "choice", "choice": "happy",
              "probabilities": {"angry": 0.002, "happy": 0.974, "sad": 0.005, "neutral": 0.007,
                                "fear": 0.005, "disgust": 0.007}, "status": "calibrated"},
  "angry":   {"type": "noul", "noul": 0.01, "where": [0.0, 2.276], "status": "calibrated"},
  "upset":   {"type": "score", "score": 0.55, "probabilities": {"0": 0.601, "1": 0.247, "2": 0.151},
              "legend": {"0": "calm", "1": "annoyed", "2": "very upset"}, "status": "off catalogue"}
}
```

Emotion was learned from actors reading 12 sentences; on natural speech it is not verified.

## The built-in questions

```python
model.questions()
# [{"name": "angry", "type": "noul", "asks": "The speaker sounds angry in this recording.",
#   "criteria": None, "status": "calibrated"}, ...]
```

46 calibrated questions, plus six that are answered but marked `"not verified"`. The full list,
with what each was trained on: [TRAINING_DATA.md](https://huggingface.co/bahadobay/dodona-audio/blob/main/TRAINING_DATA.md).
