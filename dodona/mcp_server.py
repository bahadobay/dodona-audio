"""Dodona as an MCP server, so an assistant (Claude, ...) can ask questions about audio files.

    pip install "dodona-audio[mcp]"
    dodona-mcp                         # stdio; add it to your MCP client's configuration

The model (~16 GB) is loaded on the first question, not at start-up, and kept for the next.
"""

from mcp.server.mcpserver import MCPServer

from .model import Dodona

server = MCPServer(
    "dodona",
    instructions=(
        "Dodona listens to an audio file and answers yes/no, multiple-choice and scale questions "
        "about it with calibrated probabilities. Call list_questions first: when the user's "
        "question means the same as a catalogue question, ask it by its catalogue name with an "
        "empty spec ({\"angry\": {}}): catalogue questions and the word shortcut are calibrated "
        "question by question, other questions only by a general map."
    ),
)
model = None


def loaded():
    global model
    if model is None:
        model = Dodona()
    return model


@server.tool()
def list_questions() -> list[dict]:
    """The catalogue: questions Dodona was trained on, with their type, wording, options and
    status ("calibrated" or "not verified")."""
    return loaded().questions()


@server.tool()
def ask(audio_path: str, questions: dict) -> dict:
    """Ask questions about an audio file (wav, flac, mp3, ...; any length, read in 10 s windows).

    `questions` maps a name you choose to a question:
      {}                                            a catalogue question (the name is its catalogue name)
      {"type": "noul", "word": "refund"}            was this word said?
      {"type": "noul", "instructions": "Is the speaker in a hurry?"}
      {"type": "choice", "instructions": "What does the caller want?",
       "criteria": {"refund": "Money back", "help": "Technical help"}}
      {"type": "score", "instructions": "How upset is the caller?",
       "criteria": ["calm", "annoyed", "very upset"]}
    A choice or score question has 2 to 8 options. A malformed question is refused with a message.

    Returns {"model", "duration", "answers", "windows"}. A noul answer is the probability of
    "yes" in the recording's highest 10 s window, with "where" that window is; choice and score
    answers are averaged over the windows. Each answer has a "status": "calibrated", "not
    verified" or "off catalogue" (calibrated by a general map, not one fitted to the question)."""
    return loaded().ask(audio_path, questions)


def main():
    server.run()


if __name__ == "__main__":
    main()
