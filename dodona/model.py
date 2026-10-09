"""Dodona: a voice-native decision model: ask typed questions about speech, get calibrated probabilities.

    from dodona import Dodona
    model = Dodona()                       # the adapters (~140 MB) + Qwen2-Audio-7B (~16 GB)
    result = model.ask("call.wav", {
        "angry":  {"type": "noul", "instructions": "Is the speaker angry?"},
        "refund": {"type": "noul", "word": "refund"},
        "intent": {"type": "choice", "instructions": "What does the caller want?",
                   "criteria": {"refund": "Money back", "help": "Technical help", "other": "Something else"}},
        "anger":  {"type": "score", "instructions": "How angry is the speaker?", "criteria": ["low", "mid", "high"]},
    })
    result["answers"]["angry"]["noul"]     # 0.82: the angriest 10 seconds, at result["answers"]["angry"]["where"]

The audio is cut into 10-second windows (the length Dodona was trained on). Each window is read
once and every question is answered on top of it, by reading the probability of each answer
letter (no text is generated). Two adapters answer one after the other; their mean is
calibrated, and how much they disagree is reported (high = the model is guessing).
For the whole recording, a noul question reports its highest window ("did it happen, and
where"); choice and score questions report the windows' average ("overall"). Each window's own
answers are in result["windows"].
"""

import copy
import pathlib

import numpy as np
import torch

from .catalog import Catalog

BASE = "Qwen/Qwen2-Audio-7B-Instruct"
REPO = "bahadobay/dodona-audio"
RATE, WINDOW_SECONDS, SHORTEST_SECONDS = 16000, 10.0, 1.0
LETTERS = "ABCDEFGH"
ADAPTERS = ("seed0", "seed1")
ROWS = 32  # questions read in one batch; lower it on a GPU under 40 GB
MODEL_NAME = "dodona-1.0"


def load_audio(audio):
    """A mono float32 waveform at 16 kHz, from a path or from an array already at 16 kHz."""
    if not isinstance(audio, (str, pathlib.Path)):
        return np.asarray(audio, dtype=np.float32)
    import soundfile
    from scipy.signal import resample_poly

    waveform, rate = soundfile.read(audio, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if rate != RATE:
        waveform = resample_poly(waveform, RATE, rate).astype(np.float32)
    return waveform


def windows_of(waveform):
    """(start sample, piece) per 10-second window; a sliver under a second at the end is dropped."""
    step = int(WINDOW_SECONDS * RATE)
    pieces = [(s, waveform[s: s + step]) for s in range(0, max(len(waveform), 1), step)]
    return [pieces[0]] + [(s, p) for s, p in pieces[1:] if len(p) >= SHORTEST_SECONDS * RATE]


def prompt_text(question):
    lettered = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(question.option_texts))
    return f"Question: {question.instruction}\nOptions:\n{lettered}\nAnswer with the letter only."


def answer(question, p, disagreement=None):
    """One question's answer (noul / choice / score), from its calibrated probabilities `p`."""
    out = {"type": question.type}
    if question.type == "noul":
        out["noul"] = round(float(p[1]), 3)
    else:
        out["probabilities"] = {o: round(float(v), 3) for o, v in zip(question.options, p)}
        if question.type == "choice":
            out["choice"] = question.options[int(np.argmax(p))]
        else:
            out["score"] = round(float(np.dot(np.arange(len(p)), p)), 3)
            out["legend"] = dict(zip(question.options, question.option_texts))
    out["status"] = question.status
    if disagreement is not None:
        out["disagreement"] = round(disagreement, 3)
    return out


def whole(questions, windows):
    """The recording's answers: a noul question's highest window (and where), the others'
    average over the windows, weighted by each window's length."""
    lengths = np.array([w["end"] - w["start"] for w in windows])
    out = {}
    for q in questions:
        p = np.array([w["p"][q.name] for w in windows])
        if q.type == "noul":
            best = int(np.argmax(p[:, 1]))
            out[q.name] = answer(q, p[best]) | {"where": [windows[best]["start"], windows[best]["end"]]}
        else:
            out[q.name] = answer(q, (p * lengths[:, None]).sum(0) / lengths.sum())
    return out


class Dodona:
    def __init__(self, folder=None, device=None):
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        if folder is None:
            from huggingface_hub import snapshot_download
            folder = snapshot_download(REPO)
        folder = pathlib.Path(folder)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.catalog = Catalog(folder)
        self.processor = AutoProcessor.from_pretrained(BASE)
        base = Qwen2AudioForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16)
        self.net = PeftModel.from_pretrained(base, folder / ADAPTERS[0], adapter_name=ADAPTERS[0])
        self.net.load_adapter(folder / ADAPTERS[1], adapter_name=ADAPTERS[1])
        self.net.to(self.device).eval()
        tokenizer = self.processor.tokenizer
        self.letters = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]

    def questions(self):
        """What can be asked, with each question's status."""
        return self.catalog.questions()

    def prompt(self, question):
        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": "window.wav"},
                                                      {"type": "text", "text": prompt_text(question)}]}]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def probabilities(self, waveform, questions, adapters=ADAPTERS):
        """Raw answer probabilities, [adapter][question] -> array, one adapter after the other.
        The audio and the prompt up to the question are run once per adapter and shared by every
        question. Questions are read ROWS at a time, each batch on its own copy of the shared
        part: every row holds a copy of the audio's cache (~0.4 GB for a window), so all at once
        would not fit. (Both adapters in one mixed batch is 8% faster, but on a GPU in bfloat16 it
        moved a few answers by up to 0.67; one after the other, by at most 0.05 -- experiment 076.)"""
        texts = [self.prompt(q) for q in questions]
        split = texts[0].index("Question:")
        shared = self.processor(text=texts[0][:split], audio=[waveform], sampling_rate=RATE,
                                return_tensors="pt").to(self.device)
        out = []
        for adapter in adapters:
            prefix = self.net(**shared, use_cache=True, adapter_names=[adapter]).past_key_values
            answers = []
            for first in range(0, len(questions), ROWS):
                part = slice(first, first + ROWS)
                past = prefix if first + ROWS >= len(questions) else copy.deepcopy(prefix)
                answers += self.read(past, shared.input_ids.shape[1], texts[part], split, questions[part], adapter)
            out.append(answers)
        return out

    def read(self, past, n_shared, texts, split, questions, adapter):
        """One batch: each question's ending on top of the shared cache."""
        n = len(questions)
        endings = [self.processor.tokenizer(t[split:], add_special_tokens=False).input_ids for t in texts]
        longest = max(map(len, endings))
        ids = torch.full((n, longest), self.processor.tokenizer.pad_token_id)
        mask = torch.ones(n, n_shared + longest, dtype=torch.long)
        positions = torch.zeros(n, longest, dtype=torch.long)
        for i, e in enumerate(endings):  # left-padded: every row ends at the last column
            ids[i, longest - len(e):] = torch.tensor(e)
            mask[i, n_shared: n_shared + longest - len(e)] = 0
            positions[i, longest - len(e):] = torch.arange(n_shared, n_shared + len(e))
        past.batch_repeat_interleave(n)
        last = self.net(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                        position_ids=positions.to(self.device), past_key_values=past,
                        adapter_names=[adapter] * n, logits_to_keep=1).logits[:, -1].float()
        return [last[i, self.letters[:len(q.options)]].softmax(0).cpu().numpy() for i, q in enumerate(questions)]

    def ask(self, audio, questions):
        """{"model", "duration", "answers", "windows"} for a path (or a 16 kHz array) and a
        dict of typed questions {name: spec} (see dodona.catalog)."""
        waveform = load_audio(audio)
        asked = [self.catalog.resolve(name, spec) for name, spec in questions.items()]
        windows = []
        for start, piece in windows_of(waveform):
            first, second = self.probabilities(piece, asked)
            window = {"start": start / RATE, "end": (start + len(piece)) / RATE, "answers": {}, "p": {}}
            for q, a, b in zip(asked, first, second):
                d = float(0.5 * np.abs(a - b).sum())
                window["p"][q.name] = self.catalog.calibrate(q, (a + b) / 2, d)
                window["answers"][q.name] = answer(q, window["p"][q.name], d)
            windows.append(window)
        result = {"model": MODEL_NAME, "duration": len(waveform) / RATE, "answers": whole(asked, windows)}
        result["windows"] = [{k: v for k, v in w.items() if k != "p"} for w in windows]
        return result
