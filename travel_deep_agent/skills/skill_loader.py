"""
skills/skill_loader.py
------------------------
A "skill" here is just a folder with a SKILL.md file describing HOW to do
one job well (persona, house rules, output format). This is deliberately
plain text, not Python, so a non-programmer on a travel team could tweak an
agent's behaviour by editing a markdown file -- no code changes needed.

Each subagent loads exactly one skill and uses it as its system instruction.
"""
import os

SKILLS_DIR = os.path.dirname(__file__)


def load_skill(skill_name: str) -> str:
    path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No SKILL.md found for skill '{skill_name}' at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
