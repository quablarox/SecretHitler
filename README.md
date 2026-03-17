# Secret Hitler

A Python implementation of the board game **Secret Hitler**, featuring:

- A complete, rules-accurate game engine
- Pluggable AI opponents (random, rule-based, and trainable Q-learning RL agent)
- A terminal / CLI interface for desktop play
- A [Kivy](https://kivy.org)-based GUI that runs on **Android** (and desktop)

---

## Project structure

```
secret_hitler/   ← core game engine (roles, board, deck, game state machine)
ai/              ← AI agents + training script
  random_agent.py      baseline random agent
  rule_based_agent.py  heuristic agent
  rl_agent.py          tabular Q-learning agent
  train.py             training loop / CLI
ui/
  cli.py               terminal interface (single-player vs AI)
  kivy_app.py          Kivy GUI (Android-ready)
main.py          ← top-level entry point
tests/           ← pytest unit tests
buildozer.spec   ← Android build configuration
requirements.txt
```

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt   # installs Kivy
```

### Play in the terminal

```bash
python main.py                    # 6-player game, you sit at seat 0
python main.py --players 8 --seat 3 --ai rule
python main.py --players 5 --ai rl --qtable qtable.json
```

### Launch the Kivy GUI (desktop preview / Android)

```bash
python main.py --gui
# or directly:
python ui/kivy_app.py
```

### Train the RL agent

```bash
python main.py --train --episodes 50000 --players 6 --save qtable.json
# or directly:
python -m ai.train --episodes 50000 --players 6 --save qtable.json

# Train against different opponent types
python main.py --train --episodes 10000 --opponent random   # vs random agents
python main.py --train --episodes 10000 --opponent mixed    # vs random + rule-based mix
```

After training, a benchmark automatically evaluates the agent against random,
rule-based, and mixed opponents, reporting overall and per-role win rates.

### Evaluate a pre-trained agent

```bash
python main.py --evaluate --qtable qtable.json --episodes 1000 --players 6
```

### Run all-agent games (no human)

```bash
python main.py --agents-only 500 --opponent rule --players 6
python main.py --agents-only 500 --opponent random
python main.py --agents-only 500 --opponent mixed
```

### Benchmark from Python

```python
from ai.train import train, benchmark, evaluate

agent = train(num_episodes=10000, num_players=6, save_path="qtable.json")
results = benchmark(agent, num_episodes=1000, num_players=6)
# results is a list of dicts with keys: opponent_type, win_rate,
# liberal_win_rate, fascist_win_rate, etc.
```

---

## Building for Android

1. Install [Buildozer](https://buildozer.readthedocs.io/):
   ```bash
   pip install buildozer
   ```
2. Run the Android debug build from the repo root:
   ```bash
   buildozer android debug
   ```
   The resulting `.apk` will be placed in `bin/`.

> **Note**: Buildozer requires a Linux environment with the Android NDK/SDK.
> On macOS/Windows, use a Docker image such as `kivy/buildozer`.

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Game rules summary

| Win condition | Party |
|---|---|
| 5 Liberal policies enacted | 🟦 **Liberals** |
| Hitler executed | 🟦 **Liberals** |
| 6 Fascist policies enacted | 🟥 **Fascists** |
| Hitler elected Chancellor (after 3 Fascist policies) | 🟥 **Fascists** |

For more details, check out [Issue #1](https://github.com/quablarox/SecretHitler/issues/1).

---

## AI agent design

### Agent types

| Agent | Strategy | Use case |
|---|---|---|
| `RandomAgent` | Uniformly random valid actions | Baseline / control |
| `RuleBasedAgent` | Hand-crafted heuristics, suspicion tracking | Strong opponent for training |
| `RLAgent` | Tabular Q-learning with trajectory updates | Trainable agent |

### Training approach

The RL agent can be trained **with different agents as opponents or all the same**.
Use `--opponent` to select: `random` (all RandomAgent), `rule` (all RuleBasedAgent),
or `mixed` (randomly alternating).

The training loop uses **curriculum learning**: when training against rule-based
opponents, the first 50% of episodes use random opponents (easier) so the agent
learns fundamentals before facing stronger play.

Key RL improvements over a basic Q-learning setup:
- **Full trajectory updates**: Q-values are propagated backwards through every
  decision in the game (not just the last one)
- **Intermediate reward shaping**: Small rewards (+0.1 / −0.1) for each policy
  enacted that benefits or hurts the agent's side
- **Semantic actions**: Discard actions encode tile type (`pres_discard_Liberal`)
  rather than positional index, so the agent learns which *type* of policy to
  discard regardless of draw order
- **Three backward passes** per episode for faster bootstrapping
