"""Tests for AI agents and training."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secret_hitler.game import Game, Phase
from secret_hitler.roles import Party, Role
from ai.random_agent import RandomAgent
from ai.rule_based_agent import RuleBasedAgent
from ai.rl_agent import RLAgent, _encode_state
from ai.train import run_game


# ---------------------------------------------------------------------------
# RandomAgent tests
# ---------------------------------------------------------------------------

class TestRandomAgent:
    def test_nominate_chancellor_returns_eligible(self):
        game = Game(num_players=5, seed=0)
        agent = RandomAgent(game.president_idx, seed=1)
        choice = agent.nominate_chancellor(game)
        eligible_ids = [p.player_id for p in game.eligible_chancellors()]
        assert choice in eligible_ids

    def test_vote_returns_bool(self):
        game = Game(num_players=5, seed=0)
        game.nominate_chancellor(game.eligible_chancellors()[0].player_id)
        agent = RandomAgent(0, seed=2)
        vote = agent.vote(game)
        assert isinstance(vote, bool)

    def test_president_discard_returns_valid_index(self):
        game = Game(num_players=5, seed=0)
        agent = RandomAgent(0, seed=3)
        tiles = [Party.LIBERAL, Party.FASCIST, Party.LIBERAL]
        idx = agent.president_discard(game, tiles)
        assert idx in range(len(tiles))

    def test_chancellor_discard_returns_valid_index(self):
        game = Game(num_players=5, seed=0)
        agent = RandomAgent(0, seed=4)
        tiles = [Party.LIBERAL, Party.FASCIST]
        idx = agent.chancellor_discard(game, tiles)
        assert idx in range(len(tiles))


# ---------------------------------------------------------------------------
# RuleBasedAgent tests
# ---------------------------------------------------------------------------

class TestRuleBasedAgent:
    def test_liberal_discards_fascist_policy(self):
        game = Game(num_players=5, seed=0)
        lib_id = next(p.player_id for p in game.players if p.role == Role.LIBERAL)
        agent = RuleBasedAgent(lib_id, seed=10)
        tiles = [Party.FASCIST, Party.LIBERAL, Party.LIBERAL]
        idx = agent.president_discard(game, tiles)
        assert tiles[idx] == Party.FASCIST

    def test_fascist_discards_liberal_policy(self):
        game = Game(num_players=5, seed=0)
        fas_id = next(p.player_id for p in game.players
                      if p.role in (Role.FASCIST, Role.HITLER))
        agent = RuleBasedAgent(fas_id, seed=10)
        tiles = [Party.LIBERAL, Party.FASCIST, Party.FASCIST]
        idx = agent.president_discard(game, tiles)
        assert tiles[idx] == Party.LIBERAL

    def test_investigate_result_updates_known_party(self):
        game = Game(num_players=5, seed=0)
        agent = RuleBasedAgent(0, seed=11)
        agent.on_investigate_result(1, Party.FASCIST)
        assert agent._known_party[1] == Party.FASCIST
        assert agent._suspicion[1] == 1.0


# ---------------------------------------------------------------------------
# RLAgent tests
# ---------------------------------------------------------------------------

class TestRLAgent:
    def test_encode_state_is_hashable(self):
        game = Game(num_players=5, seed=0)
        state = _encode_state(game, 0)
        d = {state: 42}
        assert d[state] == 42

    def test_vote_returns_bool(self):
        game = Game(num_players=5, seed=0)
        game.nominate_chancellor(game.eligible_chancellors()[0].player_id)
        agent = RLAgent(0, seed=5)
        vote = agent.vote(game)
        assert isinstance(vote, bool)

    def test_update_does_not_crash(self):
        game = Game(num_players=5, seed=0)
        agent = RLAgent(0, seed=5)
        agent.nominate_chancellor(game)  # records last state/action
        agent.update(1.0, None)  # should not raise

    def test_save_and_load(self, tmp_path):
        game = Game(num_players=5, seed=0)
        agent = RLAgent(0, seed=5)
        # Generate some Q-table entries
        agent.nominate_chancellor(game)
        agent.update(1.0, None)
        path = str(tmp_path / "qtable.json")
        agent.save(path)
        agent2 = RLAgent(0)
        agent2.load(path)
        assert agent2._qtable == agent._qtable


# ---------------------------------------------------------------------------
# Full game simulation tests
# ---------------------------------------------------------------------------

class TestFullGameSimulation:
    def _make_agents(self, game: Game, agent_type="random"):
        agents = []
        for pid in range(game.num_players):
            if agent_type == "random":
                agents.append(RandomAgent(pid, seed=pid * 7))
            else:
                agents.append(RuleBasedAgent(pid, seed=pid * 7))
        return agents

    def test_random_agents_complete_game(self):
        game = Game(num_players=5, seed=777)
        agents = self._make_agents(game, "random")
        completed = run_game(agents, game)
        assert completed.is_over()
        assert completed.result is not None

    def test_rule_agents_complete_game(self):
        game = Game(num_players=6, seed=888)
        agents = self._make_agents(game, "rule")
        completed = run_game(agents, game)
        assert completed.is_over()
        assert completed.result is not None

    def test_multiple_games_random_seeds(self):
        for seed in range(10):
            game = Game(num_players=5 + (seed % 6), seed=seed)
            agents = [RandomAgent(pid, seed=seed + pid) for pid in range(game.num_players)]
            completed = run_game(agents, game)
            assert completed.is_over(), f"Game with seed={seed} did not complete"

    def test_winner_is_always_set(self):
        for seed in range(20):
            game = Game(num_players=6, seed=seed)
            agents = [RuleBasedAgent(pid, seed=seed + pid) for pid in range(6)]
            run_game(agents, game)
            assert game.result is not None
            assert game.result.winner in (Party.LIBERAL, Party.FASCIST)

    def test_rl_agent_in_full_game(self):
        game = Game(num_players=5, seed=999)
        agents = [RLAgent(pid, seed=pid) for pid in range(5)]
        completed = run_game(agents, game)
        assert completed.is_over()


# ---------------------------------------------------------------------------
# Short training smoke test
# ---------------------------------------------------------------------------

class TestTraining:
    def test_short_training_run(self, tmp_path):
        from ai.train import train
        save_path = str(tmp_path / "qtable_test.json")
        agent = train(
            num_episodes=50,
            num_players=5,
            rl_player_id=0,
            save_path=save_path,
            seed=42,
        )
        assert os.path.exists(save_path)
        assert isinstance(agent._qtable, dict)
