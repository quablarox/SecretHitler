"""Q-learning reinforcement learning agent for Secret Hitler."""
from __future__ import annotations

import json
import math
import os
import random
from typing import List, Optional, Tuple, TYPE_CHECKING

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from secret_hitler.game import Game
    from secret_hitler.roles import Party

from secret_hitler.roles import Party as PartyEnum, Role


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _encode_state(game: "Game", player_id: int) -> Tuple:
    """
    Encode a compact, hashable state representation for Q-table lookup.
    We intentionally keep the state space small so tabular Q-learning is feasible.
    """
    board = game.board
    obs = game.get_observation(player_id)
    role = game.players[player_id].role

    # Discretise suspicion broadly — we don't track it in base state
    fascist_policies = min(board.fascist_policies, 6)
    liberal_policies = min(board.liberal_policies, 5)
    election_tracker = board.election_tracker
    alive_count = len(game.alive_players)
    my_party = role.party.value  # "Liberal" | "Fascist"
    phase = game.phase.name

    return (
        my_party,
        fascist_policies,
        liberal_policies,
        election_tracker,
        alive_count,
        phase,
    )


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class RLAgent(BaseAgent):
    """
    Tabular Q-learning agent.

    The agent maintains a Q-table mapping (state, action_key) → value.
    Actions are encoded as strings to be hashable.

    Training is done externally via ``train.py``.
    """

    DEFAULT_ALPHA = 0.1      # learning rate
    DEFAULT_GAMMA = 0.95     # discount factor
    DEFAULT_EPSILON = 0.2    # exploration rate (epsilon-greedy)

    def __init__(
        self,
        player_id: int,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
        seed: int = None,
        qtable_path: Optional[str] = None,
    ):
        super().__init__(player_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self._qtable: dict[Tuple, dict[str, float]] = {}

        # Experience buffer: (state, action_key, reward, next_state)
        self._last_state: Optional[Tuple] = None
        self._last_action: Optional[str] = None

        if qtable_path and os.path.exists(qtable_path):
            self.load(qtable_path)

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------

    def _q(self, state: Tuple, action: str) -> float:
        return self._qtable.get(state, {}).get(action, 0.0)

    def _best_action(self, state: Tuple, actions: List[str]) -> str:
        return max(actions, key=lambda a: self._q(state, a))

    def _choose(self, state: Tuple, actions: List[str]) -> str:
        """Epsilon-greedy action selection."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(actions)
        return self._best_action(state, actions)

    def update(self, reward: float, next_state: Optional[Tuple]) -> None:
        """Receive a reward and update Q-values (called by training loop)."""
        if self._last_state is None or self._last_action is None:
            return
        s, a = self._last_state, self._last_action
        max_future = 0.0
        if next_state:
            max_future = max(
                self._q(next_state, act)
                for act in self._qtable.get(next_state, {"": 0.0})
            ) if self._qtable.get(next_state) else 0.0
        old = self._q(s, a)
        new = old + self.alpha * (reward + self.gamma * max_future - old)
        self._qtable.setdefault(s, {})[a] = new

    def _record(self, state: Tuple, action: str) -> None:
        self._last_state = state
        self._last_action = action

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def nominate_chancellor(self, game: "Game") -> int:
        state = _encode_state(game, self.player_id)
        eligible = [p.player_id for p in game.eligible_chancellors()]
        actions = [f"nominate_{pid}" for pid in eligible]
        chosen = self._choose(state, actions)
        self._record(state, chosen)
        return int(chosen.split("_")[1])

    def vote(self, game: "Game") -> bool:
        state = _encode_state(game, self.player_id)
        actions = ["vote_ja", "vote_nein"]
        chosen = self._choose(state, actions)
        self._record(state, chosen)
        return chosen == "vote_ja"

    def president_discard(self, game: "Game", tiles: List["Party"]) -> int:
        state = _encode_state(game, self.player_id)
        actions = [f"pres_discard_{i}" for i in range(len(tiles))]
        chosen = self._choose(state, actions)
        self._record(state, chosen)
        return int(chosen.split("_")[-1])

    def chancellor_discard(self, game: "Game", tiles: List["Party"]) -> int:
        state = _encode_state(game, self.player_id)
        actions = [f"chan_discard_{i}" for i in range(len(tiles))]
        chosen = self._choose(state, actions)
        self._record(state, chosen)
        return int(chosen.split("_")[-1])

    def choose_investigate_target(self, game: "Game") -> int:
        state = _encode_state(game, self.player_id)
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        actions = [f"investigate_{pid}" for pid in candidates]
        chosen = self._choose(state, actions)
        self._record(state, chosen)
        return int(chosen.split("_")[1])

    def choose_special_election_target(self, game: "Game") -> int:
        state = _encode_state(game, self.player_id)
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        actions = [f"special_election_{pid}" for pid in candidates]
        chosen = self._choose(state, actions)
        self._record(state, chosen)
        return int(chosen.split("_")[-1])

    def choose_execution_target(self, game: "Game") -> int:
        state = _encode_state(game, self.player_id)
        candidates = [pid for pid in game.alive_ids if pid != self.player_id]
        actions = [f"execute_{pid}" for pid in candidates]
        chosen = self._choose(state, actions)
        self._record(state, chosen)
        return int(chosen.split("_")[1])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save Q-table to a JSON file."""
        serialisable = {
            str(k): v for k, v in self._qtable.items()
        }
        with open(path, "w") as f:
            json.dump(serialisable, f)

    def load(self, path: str) -> None:
        """Load Q-table from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        import ast
        self._qtable = {ast.literal_eval(k): v for k, v in data.items()}
