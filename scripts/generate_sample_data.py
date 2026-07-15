"""
Generates a small synthetic multi-class news dataset used as the default
demo dataset for the pipeline (data/sample_news.csv). Not meant to be a
research-grade dataset -- just enough topical diversity across 4 classes
(World, Sports, Business, Sci/Tech) to exercise the full annotation +
training pipeline end-to-end. Swap in AG News, Reuters-21578, or any other
CSV by editing config/config.yaml -> data.dataset_path.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(7)

TEMPLATES = {
    "World": [
        "The {country} government announced new {policy} measures amid rising tensions.",
        "{leader} met with foreign ministers to discuss the ongoing crisis in {region}.",
        "United Nations officials called for an immediate ceasefire in {region}.",
        "Elections in {country} drew record turnout despite security concerns.",
        "The prime minister of {country} resigned following weeks of protests.",
        "Diplomats from {country} and {country2} resumed peace talks this week.",
        "A humanitarian convoy reached {region} after being delayed at the border.",
        "The president signed a landmark treaty with {country2} on {policy}.",
    ],
    "Sports": [
        "The {team} clinched the championship after a dramatic overtime win.",
        "{player} scored a hat-trick to lead {team} to victory in the final.",
        "Coach {player} announced retirement after two decades in {sport}.",
        "{team} secured a playoff spot with a last-minute goal against their rivals.",
        "The {sport} tournament final drew a record television audience.",
        "Injuries forced {player} out of the upcoming {sport} season opener.",
        "{team} broke the league record for consecutive wins in {sport}.",
        "Fans celebrated as {team} lifted the trophy for the third year running.",
    ],
    "Business": [
        "The central bank raised interest rates by {number}% to curb inflation.",
        "{company} reported quarterly revenue growth of {number}% driven by strong demand.",
        "Shares of {company} tumbled after disappointing earnings guidance.",
        "{company} announced a merger with a rival firm worth billions.",
        "The stock market rallied after positive {sector} sector earnings.",
        "{company} plans to cut {number} jobs as part of a restructuring effort.",
        "Investors reacted cautiously to the new trade tariffs on {sector} goods.",
        "The unemployment rate fell to its lowest level in a decade, boosting markets.",
    ],
    "Sci/Tech": [
        "Researchers unveiled a new {tech} that could transform {industry}.",
        "{company} launched an updated {tech} with improved battery life.",
        "Scientists discovered a breakthrough in {field} research this week.",
        "A new study shows {tech} adoption is accelerating across {industry}.",
        "{company} announced a partnership to develop next-generation {tech}.",
        "Engineers successfully tested a prototype {tech} in a controlled environment.",
        "The space agency confirmed a successful launch of its new {tech} mission.",
        "Cybersecurity experts warned of a new vulnerability affecting {tech} systems.",
    ],
}

FILLERS = {
    "country": ["France", "Brazil", "Japan", "Kenya", "Germany", "India", "Canada", "Egypt"],
    "country2": ["Spain", "Mexico", "South Korea", "Nigeria", "Italy", "Vietnam"],
    "leader": ["The foreign minister", "The chancellor", "The special envoy", "The ambassador"],
    "region": ["the eastern border region", "the capital", "the disputed territory", "the coastal province"],
    "policy": ["trade", "immigration", "climate", "security", "energy"],
    "team": ["the Falcons", "the Riverside Union", "the Northside Titans", "the Harbor City FC", "the Summit Wolves"],
    "player": ["Alvarez", "Kowalski", "Chen", "Okafor", "Petrov", "Nakamura"],
    "sport": ["football", "basketball", "cricket", "tennis", "rugby"],
    "company": ["Nova Dynamics", "BrightPeak Corp", "Solara Industries", "Vertex Financial", "Northwind Retail"],
    "number": ["3", "5", "12", "0.5", "7", "20"],
    "sector": ["technology", "energy", "retail", "healthcare", "manufacturing"],
    "tech": ["battery technology", "AI chip", "quantum processor", "wearable device", "satellite system"],
    "industry": ["healthcare", "manufacturing", "transportation", "agriculture", "logistics"],
    "field": ["genomics", "materials science", "renewable energy", "neuroscience"],
}


def fill(template: str, rng: random.Random) -> str:
    out = template
    for key, options in FILLERS.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(options))
    return out


def generate(n_per_class: int = 60, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    idx = 0
    for label, templates in TEMPLATES.items():
        for _ in range(n_per_class):
            template = rng.choice(templates)
            text = fill(template, rng)
            rows.append({"id": f"row_{idx}", "text": text, "label": label})
            idx += 1
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"row_{i}"
    return rows


def main() -> None:
    out_path = Path(__file__).resolve().parent.parent / "data" / "sample_news.csv"
    rows = generate(n_per_class=60)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
