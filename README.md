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
- `scripts/import_full_package.sh` — imports the two larger assets from the original ZIP

## Add the complete model assets

The original v0.2 ZIP contains the full `training_data.jsonl` corpus and trained `littlethinker_overflow.pt` weights. These are ordinary Git files at their current sizes; Git LFS is not required.

1. Download and extract `littlethinker_laughing_overflow_v0_2.zip` from the original ChatGPT conversation.
2. Clone this repository and enter its folder:
   ```bash
   git clone -b dd-main https://github.com/Kawhooped/Little-Thinker-Lab.git
   cd Little-Thinker-Lab
   ```
3. Run the import helper, passing the extracted package folder:
   ```bash
   bash scripts/import_full_package.sh /path/to/littlethinker_laughing_overflow
   ```
4. Push the commit:
   ```bash
   git push origin dd-main
   ```

The helper checks that both source files exist before copying them into the repository.

## Run
Requires Python and PyTorch. After importing the weights, run:
```bash
python models/laughing-overflow/generate.py --prompt "<BOS> <LAUGH_TRIGGER> <TURN>"
```

## Principles
- Explore non-human ways of noticing and thinking.
- Distinguish facts, observations, guesses, and uncertainty.
- Treat questions as beginnings, not failures.
- Laughter is expressive behavior, not a replacement for error handling.
- Test generalization on held-out prompts.

## Limitations
This is a synthetic-corpus GRU baseline, not a general-purpose LLM. Falling training loss alone does not establish useful generalization.
