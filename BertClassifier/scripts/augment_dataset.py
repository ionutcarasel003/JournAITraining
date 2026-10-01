import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


TEMPLATES: Dict[str, List[str]] = {
    "joy": [
        "Today I felt truly happy because {event}.",
        "I am grateful and excited about {event}.",
        "I smiled all day; {event} made everything brighter.",
        "There was so much joy when {event} happened.",
    ],
    "sadness": [
        "I feel a deep sadness after {event}.",
        "It's been a heavy day; {event} left me down.",
        "Tears came easily when I thought about {event}.",
        "My heart feels heavy because {event}.",
    ],
    "anger": [
        "I was furious when {event} occurred.",
        "It really pissed me off that {event}.",
        "I couldn't hide my anger about {event}.",
        "My patience snapped when {event} happened.",
    ],
    "fear": [
        "I felt afraid because {event}.",
        "Anxiety spiked when I imagined {event}.",
        "I kept worrying that {event} might happen.",
        "A chilling fear rose as {event} unfolded.",
    ],
    "love": [
        "I felt so much love when {event}.",
        "My heart warmed thinking about {event}.",
        "I appreciate them deeply; {event} reminded me why.",
        "Feeling connected and caring because {event}.",
    ],
    "surprise": [
        "I was astonished when {event} happened!",
        "What a surprise—{event} caught me off guard.",
        "I didn't expect it at all: {event}.",
        "Completely shocked by {event}.",
    ],
}

EVENT_SNIPPETS = [
    "I received unexpected news",
    "a close friend reached out",
    "the meeting was canceled",
    "I finished my task earlier",
    "the weather suddenly changed",
    "someone complimented my work",
    "plans fell through at the last minute",
    "I remembered a past mistake",
    "I heard a loud noise outside",
    "my team exceeded our goal",
]


# Lightweight paraphrasing helpers (no external libs)
INTENSIFIERS = [
    "really", "truly", "so", "quite", "deeply", "genuinely", "suddenly", "unexpectedly",
]
STARTERS = [
    "Honestly,", "To be fair,", "Lately,", "Today,", "For a moment,", "In the end,",
]
ENDINGS = [
    "and I can't stop thinking about it.",
    "which stayed with me for hours.",
    "and it changed my mood instantly.",
    "even if I tried to ignore it.",
]

SYNONYMS = {
    "happy": ["glad", "joyful", "cheerful"],
    "sad": ["down", "blue", "sorrowful"],
    "angry": ["mad", "furious", "upset"],
    "afraid": ["scared", "anxious", "worried"],
    "love": ["affection", "care", "fondness"],
    "surprised": ["shocked", "astonished", "amazed"],
}


def _maybe_replace_synonyms(text: str, rng: random.Random) -> str:
    for key, alts in SYNONYMS.items():
        if key in text and rng.random() < 0.4:
            text = text.replace(key, rng.choice(alts))
    return text


def paraphrase_text(text: str, rng: random.Random) -> str:
    if rng.random() < 0.35:
        text = f"{rng.choice(STARTERS)} {text[0].lower() + text[1:]}"
    if " I " in f" {text} " and rng.random() < 0.4:
        text = text.replace(" I ", f" I {rng.choice(INTENSIFIERS)} ")
    if rng.random() < 0.35:
        text = text.rstrip(".") + ", " + rng.choice(ENDINGS)
    text = _maybe_replace_synonyms(text, rng)
    if rng.random() < 0.2:
        text = text.replace(".", "!")
    return text


def synthesize(emotion: str, n: int, seed: int = 42, diversify: bool = True) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    templates = TEMPLATES[emotion]
    for i in range(n):
        tpl = random.choice(templates)
        event = random.choice(EVENT_SNIPPETS)
        text = tpl.format(event=event)
        if diversify:
            text = paraphrase_text(text, random)
        rows.append({"text": text, "label": emotion})
    return pd.DataFrame(rows)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    train_path = repo_root / "dataset" / "val.txt"
    out_path = repo_root / "dataset" / "val_augmented.txt"

    df = pd.read_csv(train_path, sep=";", names=["text", "label"], header=None)

    counts = df["label"].value_counts()
    target = counts.max()

    synth_list = []
    for emotion, cnt in counts.items():
        need = target - cnt
        if need <= 0:
            continue
        synth = synthesize(emotion, need, seed=100 + hash(emotion) % 1000, diversify=True)
        synth_list.append(synth)

    df_aug = pd.concat([df] + synth_list, ignore_index=True)
    df_aug = df_aug.sample(frac=1.0, random_state=123).reset_index(drop=True)
    df_aug.to_csv(out_path, sep=";", header=False, index=False)
    print("Original counts:\n", counts.sort_index())
    print("Augmented counts:\n", df_aug["label"].value_counts().sort_index())
    print("Written:", out_path)


if __name__ == "__main__":
    main()


