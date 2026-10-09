"""Turning a Jev-style question into Dodona's prompt, and its raw answer into a calibrated one.

Questions come as Jev asks them, by a name you choose:
  {"type": "noul",   "instructions": "Is the speaker angry?"}                    yes / no
  {"type": "noul",   "instructions": "...", "criteria": {"true": ..., "false": ...}}
  {"type": "choice", "instructions": "...", "criteria": {"refund": "Money back", ...}}
  {"type": "score",  "instructions": "...", "criteria": ["low", "mid", "high"]}
Dodona adds two shortcuts:
  {"type": "noul", "word": "refund"}      was this word said (any word; calibrated)
  "angry": {}                             a catalogue question by its name (calibrated)
A question whose instructions are one of the trained wordings is a catalogue question too.
Every other question is answered but marked "off catalogue": its calibration is not verified.
"""

import json
import pathlib
import re
from dataclasses import dataclass

import numpy as np

WORD = re.compile(r'^\s*the word\s+["“]?(.+?)["”]?\s+is said in this recording\.?\s*$', re.IGNORECASE)
WORD_TEMPLATE = 'The word "{word}" is said in this recording.'
TYPES = {"yes/no": "noul", "choice": "choice", "score": "score"}


def normal(text):
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


@dataclass
class Question:
    name: str           # the user's name for it
    type: str           # "noul", "choice" or "score"
    instruction: str    # the question line of the prompt
    options: list       # answer names: ["no", "yes"], the choice keys, or "0".."n-1"
    option_texts: list  # the answers as the prompt lists them
    map: str            # "yes/no", "words", "choice", "score" or "off catalogue"
    status: str         # "calibrated", "not verified" or "off catalogue"
    id: str = None      # catalogue id or says_<word>


def noul(name, statement, map, status, id=None, texts=("no", "yes")):
    """A yes/no question; a statement is asked as the model was trained: "Is this true ...?"."""
    line = statement if statement.rstrip().endswith("?") else f"Is this true of the excerpt? {statement}"
    return Question(name, "noul", line, ["no", "yes"], list(texts), map, status, id)


class Catalog:
    def __init__(self, folder):
        folder = pathlib.Path(folder)
        catalog = json.loads((folder / "catalog.json").read_text())
        self.maps = json.loads((folder / "calibration.json").read_text())["maps"]
        self.words_status = "calibrated" if catalog["word_template"] else "not verified"
        self.listed, self.rows = [], {}
        for row, status in [(r, "calibrated") for r in catalog["questions"]] + \
                           [(r, "not verified") for r in catalog["not_listed"]]:
            row = dict(row, status=status)
            self.listed.append(row)
            self.rows[row["id"]] = row
            for wording in row["wordings"]:
                self.rows[normal(wording)] = row

    def questions(self):
        """The catalogue: what can be asked by name, with its status."""
        out = [{"name": r["id"], "type": TYPES[r["kind"]], "asks": r["wordings"][0],
                "criteria": r["option_texts"], "status": r["status"]} for r in self.listed]
        out.append({"name": "<any>", "type": "noul", "asks": '{"type": "noul", "word": "<any word>"}',
                    "criteria": None, "status": self.words_status})
        return out

    def from_row(self, name, row):
        if row["kind"] == "yes/no":
            return noul(name, row["wordings"][0], row["map"], row["status"], row["id"])
        options = row["options"] if row["kind"] == "choice" else [str(i) for i in range(len(row["option_texts"]))]
        return Question(name, TYPES[row["kind"]], row["wordings"][0], options, row["option_texts"],
                        row["map"], row["status"], row["id"])

    def resolve(self, name, spec):
        """The Question to ask for one entry of the request: `name` and its spec."""
        if spec.get("word"):
            word = spec["word"].strip().lower()
            return noul(name, WORD_TEMPLATE.format(word=word), "words", self.words_status, f"says_{word}")
        text = spec.get("instructions")
        if not text:
            if name not in self.rows:
                raise ValueError(f"{name!r}: no instructions, and not a catalogue name (see Dodona.questions())")
            return self.from_row(name, self.rows[name])
        found = WORD.match(text)
        if found:
            return self.resolve(name, {"word": found.group(1)})
        if normal(text) in self.rows and "criteria" not in spec:
            return self.from_row(name, self.rows[normal(text)])
        kind, criteria = spec.get("type", "noul"), spec.get("criteria")
        if kind == "noul":
            texts = (criteria["false"], criteria["true"]) if criteria else ("no", "yes")
            return noul(name, text, "off catalogue", "off catalogue", texts=texts)
        if kind == "choice":
            criteria = criteria if isinstance(criteria, dict) else {c: c for c in criteria}
            return Question(name, "choice", text, list(criteria), list(criteria.values()), "off catalogue", "off catalogue")
        if kind == "score":
            return Question(name, "score", text, [str(i) for i in range(len(criteria))], list(criteria),
                            "off catalogue", "off catalogue")
        raise ValueError(f"{name!r}: type must be noul, choice or score, not {kind!r}")

    def calibrate(self, question, p, disagreement):
        """Calibrated probabilities from the two adapters' mean `p` and their disagreement."""
        if question.map in ("yes/no", "words"):
            curve = self.maps[question.map]
            yes = float(np.interp(p[1], curve["x"], curve["y"]))
            return np.array([1 - yes, yes])
        if question.map == "off catalogue":  # fitted on yes/no rewordings; applied to any width
            m = self.maps["off catalogue"]["yes/no"]
            t = m["t0"] + m["a"] * disagreement
        else:
            t = self.maps[question.map]["temperature"]
        z = np.log(np.clip(p, 1e-8, 1)) / t
        z = np.exp(z - z.max())
        return z / z.sum()
