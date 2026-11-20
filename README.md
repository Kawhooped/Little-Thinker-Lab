# LittleThinker Lab

A collection of tiny experimental language models with distinct inner worlds.

## First prototype: Mosslight — Laughing Overflow v0.2

Mosslight is a small fictional world inhabited by Pip (a moss-spark), Mumo (a cloudling), Tikka (a bell-bug), and Oru (a glow-seed). The prototype explores simple language, reusable concepts, uncertainty, observation, playful expression, and laughter as a response to surprise or absurdity.

### Training snapshot
- 1,310 synthetic training examples
- 21,830 next-token training positions
- 405 vocabulary tokens
- 111,861 parameters
- Small GRU language model; 5 training epochs

## Repository layout
- `worlds/mosslight/world.json` — world premise, inhabitants, places, and rules
- `models/laughing-overflow/vocabulary.json` — tokenizer vocabulary
- `models/laughing-overflow/training_report.json` — measurements and limitations
- `models/laughing-overflow/generate.py` — minimal inference script

## Package transfer status
The original v0.2 ZIP also contains the full `training_data.jsonl` and trained `littlethinker_overflow.pt` weights. Those two files are not yet committed here; retain the original ZIP as the complete source package until they are added.

## Run
Requires Python and PyTorch. Put `littlethinker_overflow.pt` beside `generate.py`, then run:
```bash
python generate.py --prompt "<BOS> <LAUGH_TRIGGER> <TURN>"
```

## Principles
- Explore non-human ways of noticing and thinking.
- Distinguish facts, observations, guesses, and uncertainty.
- Treat questions as beginnings, not failures.
- Laughter is expressive behavior, not a replacement for error handling.
- Test generalization on held-out prompts.

## Limitations
This is a synthetic-corpus GRU baseline, not a general-purpose LLM. Falling training loss alone does not establish useful generalization.
