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

    def test_train_with_random_opponents(self, tmp_path):
        from ai.train import train
        save_path = str(tmp_path / "qtable_random.json")
        agent = train(
            num_episodes=50,
            num_players=5,
            rl_player_id=0,
            save_path=save_path,
            seed=42,
            opponent_type="random",
        )
        assert os.path.exists(save_path)
        assert len(agent._qtable) > 0

    def test_train_with_mixed_opponents(self, tmp_path):
        from ai.train import train
        save_path = str(tmp_path / "qtable_mixed.json")
        agent = train(
            num_episodes=50,
            num_players=5,
            rl_player_id=0,
            save_path=save_path,
            seed=42,
            opponent_type="mixed",
        )
        assert os.path.exists(save_path)
        assert len(agent._qtable) > 0


# ---------------------------------------------------------------------------
# Trajectory and episode lifecycle tests
# ---------------------------------------------------------------------------

class TestRLAgentTrajectory:
    def test_reset_clears_trajectory(self):
        agent = RLAgent(0, seed=0)
        game = Game(num_players=5, seed=0)
        agent.nominate_chancellor(game)
        assert len(agent._trajectory) > 0
        agent.reset_episode()
        assert len(agent._trajectory) == 0
        assert agent._last_state is None
        assert agent._last_action is None

    def test_end_episode_updates_qtable(self):
        agent = RLAgent(0, seed=0)
        game = Game(num_players=5, seed=0)
        agent.nominate_chancellor(game)
        assert len(agent._trajectory) == 1
        agent.end_episode(1.0)
        assert len(agent._qtable) > 0
        # trajectory cleared after end_episode
        assert len(agent._trajectory) == 0

    def test_record_step_reward(self):
        agent = RLAgent(0, seed=0)
        game = Game(num_players=5, seed=0)
        agent.nominate_chancellor(game)
        agent.record_step_reward(0.1)
        s, a, r = agent._trajectory[-1]
        assert r == 0.1

    def test_record_step_reward_accumulates(self):
        agent = RLAgent(0, seed=0)
        game = Game(num_players=5, seed=0)
        agent.nominate_chancellor(game)
        agent.record_step_reward(0.1)
        agent.record_step_reward(0.05)
        s, a, r = agent._trajectory[-1]
        assert r == pytest.approx(0.15)

    def test_end_episode_with_empty_trajectory(self):
        agent = RLAgent(0, seed=0)
        # Should not raise
        agent.end_episode(1.0)
        assert len(agent._qtable) == 0


# ---------------------------------------------------------------------------
# Evaluation and benchmark tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_returns_correct_structure(self):
        from ai.train import evaluate
        agent = RLAgent(0, seed=42)
        result = evaluate(agent, num_episodes=20, num_players=5,
                          opponent_type="random", seed=0)
        assert "opponent_type" in result
        assert "total_games" in result
        assert "wins" in result
        assert "losses" in result
        assert "win_rate" in result
        assert "liberal_wins" in result
        assert "liberal_total" in result
        assert "liberal_win_rate" in result
        assert "fascist_wins" in result
        assert "fascist_total" in result
        assert "fascist_win_rate" in result
        assert result["total_games"] == 20
        assert result["wins"] + result["losses"] == 20

    def test_evaluate_per_role_totals(self):
        from ai.train import evaluate
        agent = RLAgent(0, seed=42)
        result = evaluate(agent, num_episodes=50, num_players=5,
                          opponent_type="random", seed=1)
        assert result["liberal_total"] + result["fascist_total"] == 50
        assert result["liberal_wins"] <= result["liberal_total"]
        assert result["fascist_wins"] <= result["fascist_total"]

    def test_benchmark_returns_three_results(self):
        from ai.train import benchmark
        agent = RLAgent(0, seed=42)
        results = benchmark(agent, num_episodes=20, num_players=5, seed=0)
        assert len(results) == 3
        types = [r["opponent_type"] for r in results]
        assert "random" in types
        assert "rule" in types
        assert "mixed" in types

    def test_evaluate_restores_epsilon(self):
        from ai.train import evaluate
        agent = RLAgent(0, seed=42, epsilon=0.15)
        evaluate(agent, num_episodes=10, num_players=5,
                 opponent_type="random", seed=0)
        assert agent.epsilon == 0.15


# ---------------------------------------------------------------------------
# State encoding tests
# ---------------------------------------------------------------------------

class TestStateEncoding:
    def test_state_includes_president_chancellor(self):
        game = Game(num_players=5, seed=0)
        pres_state = _encode_state(game, game.president_idx)
        # is_president should be True for the president
        assert pres_state[5] is True   # is_president
        assert pres_state[6] is False  # is_chancellor (no chancellor yet)

    def test_state_different_for_president_vs_non_president(self):
        game = Game(num_players=5, seed=0)
        non_pres = [pid for pid in range(5) if pid != game.president_idx][0]
        pres_state = _encode_state(game, game.president_idx)
        other_state = _encode_state(game, non_pres)
        # States differ at minimum in is_president
        assert pres_state != other_state
