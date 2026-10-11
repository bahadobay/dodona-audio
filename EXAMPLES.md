# Dodona examples

Real outputs of the released model on test recordings Dodona never trained on, run on the demo
Space's GPU (the same answers as on an NVIDIA L40S to within 0.012). The audio is not included
here; each example names its file in the corpus it comes from.

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
    "several":    {"type": "noul", "instructions": "More than one person speaks in this excerpt."},
    "remote":     {"type": "noul", "word": "remote"},
    "introduce":  {"type": "noul", "instructions": "Does someone introduce themselves?"},
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
    "several":    {"type": "noul", "noul": 0.958, "where": [80.0, 90.0], "status": "calibrated"},
    "remote":     {"type": "noul", "noul": 0.99, "where": [60.0, 70.0], "status": "calibrated"},
    "introduce":  {"type": "noul", "noul": 0.91, "where": [30.0, 40.0], "status": "off catalogue"},
    "topic":      {"type": "choice", "choice": "design",
                   "probabilities": {"design": 0.504, "money": 0.166, "other": 0.329}, "status": "off catalogue"},
    "energy":     {"type": "score", "score": 0.827, "probabilities": {"0": 0.426, "1": 0.321, "2": 0.253},
                   "legend": {"0": "low", "1": "mid", "2": "high"}, "status": "off catalogue"}
  },
  "windows": [
    {"start": 0.0, "end": 10.0, "answers": {
      "laughs":     {"type": "noul", "noul": 0.037, "status": "calibrated", "disagreement": 0.015},
      "several":    {"type": "noul", "noul": 0.205, "status": "calibrated", "disagreement": 0.034},
      "remote":     {"type": "noul", "noul": 0.01, "status": "calibrated", "disagreement": 0.015},
      "...": "..."}},
    "... 11 more windows ..."
  ]
}
```

How to read it:
- **A yes/no answer is about its strongest 10 seconds.** `laughs` 0.93 with `where` [40, 50]:
  laughter, most clearly 40 to 50 s into the piece. `remote` 0.99: the word is said, clearest
  at 60 to 70 s. In the first 10 seconds alone, neither happens (0.037, 0.01). `introduce` 0.91
  at [30, 40]: the four people introduce themselves around 20 to 40 s. AMI's own annotation
  puts the laughter, the word and the introductions in the same places.
- **`several` follows who is talking.** It is 0.958 at its strongest. Window by window, it is
  0.92 to 0.96 in every 10 seconds where AMI's annotation has two or more people speaking and
  0.30 or less where one person speaks (0.205 in the first window, a monologue). In one window
  the second person says two words: there it is 0.473, with a high `disagreement` (0.698).
- **Choice and score answers are averages over the windows.** `topic` "design" (0.504)
  describes the recording as a whole.
- **`score` is the expected level**, 0.827 on a scale from 0 (low) to 2 (high): the
  probabilities of the three options (0.426, 0.321, 0.253) weighted by their level.
- **`disagreement`**, in each window, is how far Dodona's two adapters are apart. Near 0: they
  agree. High: Dodona is guessing.

## 2. A short angry sentence

CREMA-D, `1006_WSI_ANG_XX.wav` (2.7 s), an actor reading a sentence angrily; 10 of 10 people
who rated the voice alone heard anger (CREMA-D's own ratings). A test actor, never heard in
training; CREMA-D's 12 sentences are read by every actor, so the sentence itself was. Under 10
seconds there is one window, and its answers are the answers.

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
              "probabilities": {"angry": 0.969, "happy": 0.01, "sad": 0.004, "neutral": 0.005,
                                "fear": 0.004, "disgust": 0.008}, "status": "calibrated"},
  "angry":   {"type": "noul", "noul": 0.978, "where": [0.0, 2.736], "status": "calibrated"},
  "upset":   {"type": "score", "score": 1.27, "probabilities": {"0": 0.07, "1": 0.589, "2": 0.341},
              "legend": {"0": "calm", "1": "annoyed", "2": "very upset"}, "status": "off catalogue"}
}
```

`emotion` is a `choice` (one of six emotions) and `angry` a `noul` (yes/no); both are built-in
questions and clear (`angry` 0.978, the two adapters' disagreement 0.0). `upset` is a `score`
question of our own, and it lands just above "annoyed" (1.27). Its window shows a disagreement
of 0.248, higher than the built-in questions': the two adapters answer it less alike.

## 3. A short happy sentence

CREMA-D, `1006_DFA_HAP_XX.wav` (2.3 s), the same actor, happily, same questions; 8 of 9 people
who rated the voice alone heard happiness.

```json
{
  "emotion": {"type": "choice", "choice": "happy",
              "probabilities": {"angry": 0.002, "happy": 0.983, "sad": 0.003, "neutral": 0.004,
                                "fear": 0.003, "disgust": 0.005}, "status": "calibrated"},
  "angry":   {"type": "noul", "noul": 0.01, "where": [0.0, 2.302], "status": "calibrated"},
  "upset":   {"type": "score", "score": 0.407, "probabilities": {"0": 0.72, "1": 0.152, "2": 0.127},
              "legend": {"0": "calm", "1": "annoyed", "2": "very upset"}, "status": "off catalogue"}
}
```

Emotion was learned from actors reading 12 sentences, and it predicts the emotion the actor was
asked to portray. Listeners do not always hear that emotion in the voice: for an earlier pair of
clips from another actor, most of the people rating the voice alone heard anger in the "happy"
one. On natural speech it is not verified.

## The built-in questions

```python
model.questions()
# [{"name": "angry", "type": "noul", "asks": "The speaker sounds angry in this recording.",
#   "criteria": None, "status": "calibrated"}, ...]
```

46 calibrated questions, plus six that are answered but marked `"not verified"`. The full list,
with what each was trained on: [TRAINING_DATA.md](https://huggingface.co/bahadobay/dodona-audio/blob/main/TRAINING_DATA.md).
