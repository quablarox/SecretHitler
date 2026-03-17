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
