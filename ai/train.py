"""Training loop for the RL agent."""
from __future__ import annotations

import argparse
import os
import random
from typing import List

from secret_hitler.game import Game, Phase
from secret_hitler.roles import Party, Role
from ai.rl_agent import RLAgent, _encode_state
from ai.rule_based_agent import RuleBasedAgent
from ai.random_agent import RandomAgent


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def _compute_reward(game: Game, player_id: int) -> float:
    """
    Return a scalar reward for *player_id* given the current game state.
    Called at the END of each game.
    """
    if not game.is_over():
        return 0.0
    player = game.players[player_id]
    winner = game.result.winner
    if player.role == Role.LIBERAL:
        return 1.0 if winner == Party.LIBERAL else -1.0
    else:  # Fascist or Hitler
        return 1.0 if winner == Party.FASCIST else -1.0


def _step_reward_from_delta(
    prev_liberal: int, prev_fascist: int,
    game: Game, player_id: int,
) -> float:
    """Return a small intermediate reward when a policy was just enacted."""
    board = game.board
    role = game.players[player_id].role
    is_liberal = role.party == Party.LIBERAL

    lib_delta = board.liberal_policies - prev_liberal
    fas_delta = board.fascist_policies - prev_fascist

    if lib_delta > 0:
        return 0.1 if is_liberal else -0.1
    if fas_delta > 0:
        return 0.1 if not is_liberal else -0.1
    return 0.0


# ---------------------------------------------------------------------------
# Game runner — resolves all decisions via agents
# ---------------------------------------------------------------------------

def run_game(agents: List, game: Game) -> Game:
    """Play through a full game, returning the completed Game object."""
    # Identify RL agents for intermediate reward tracking
    rl_agents = [a for a in agents if isinstance(a, RLAgent)]

    while not game.is_over():
        phase = game.phase
        pres_id = game.president_idx

        # Snapshot policy counts before any action (for intermediate reward)
        prev_lib = game.board.liberal_policies
        prev_fas = game.board.fascist_policies

        if phase == Phase.NOMINATION:
            agent = agents[pres_id]
            choice = agent.nominate_chancellor(game)
            game.nominate_chancellor(choice)

        elif phase == Phase.VOTE:
            for pid in game.alive_ids:
                vote = agents[pid].vote(game)
                result = game.cast_vote(pid, vote)
                if result is not None:
                    break  # vote resolved

        elif phase == Phase.LEGISLATIVE_PRESIDENT:
            agent = agents[pres_id]
            idx = agent.president_discard(game, game._drawn_tiles)
            game.president_discard(idx)

        elif phase == Phase.LEGISLATIVE_CHANCELLOR:
            chan_id = game.chancellor_idx
            agent = agents[chan_id]
            if (game.board.veto_unlocked
                    and not game._veto_rejected
                    and agent.request_veto(game, game._chancellor_tiles)):
                game.chancellor_request_veto()
            else:
                idx = agent.chancellor_discard(game, game._chancellor_tiles)
                game.chancellor_discard(idx)

        elif phase == Phase.VETO_REQUESTED:
            agent = agents[pres_id]
            accept = agent.respond_veto(game)
            game.president_respond_veto(accept)

        elif phase == Phase.PRESIDENTIAL_POWER:
            agent = agents[pres_id]
            power = game.pending_power
            if power == "policy_peek":
                tiles = game.use_policy_peek()
                agent.on_policy_peek(tiles)
            elif power == "investigate":
                target = agent.choose_investigate_target(game)
                party = game.use_investigate(target)
                agent.on_investigate_result(target, party)
            elif power == "special_election":
                target = agent.choose_special_election_target(game)
                game.use_special_election(target)
            elif power == "execution":
                target = agent.choose_execution_target(game)
                game.use_execution(target)

        # Check for intermediate reward (policy enacted this step)
        if rl_agents:
            new_lib = game.board.liberal_policies
            new_fas = game.board.fascist_policies
            if new_lib != prev_lib or new_fas != prev_fas:
                for rla in rl_agents:
                    sr = _step_reward_from_delta(
                        prev_lib, prev_fas, game, rla.player_id,
                    )
                    if sr != 0.0:
                        rla.record_step_reward(sr)

    return game


# ---------------------------------------------------------------------------
# Opponent factory
# ---------------------------------------------------------------------------

def _make_opponents(
    num_players: int,
    rl_player_id: int,
    opponent_type: str,
    game_seed: int,
    rng: random.Random,
) -> List:
    """Create opponent agents for seats other than *rl_player_id*."""
    agents: List = [None] * num_players  # type: ignore[list-item]
    for pid in range(num_players):
        if pid == rl_player_id:
            continue
        if opponent_type == "random":
            agents[pid] = RandomAgent(pid, seed=game_seed + pid)
        elif opponent_type == "rule":
            agents[pid] = RuleBasedAgent(pid, seed=game_seed + pid)
        else:  # "mixed"
            if rng.random() < 0.5:
                agents[pid] = RandomAgent(pid, seed=game_seed + pid)
            else:
                agents[pid] = RuleBasedAgent(pid, seed=game_seed + pid)
    return agents


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    num_episodes: int = 10_000,
    num_players: int = 6,
    rl_player_id: int = 0,
    save_path: str = "qtable.json",
    seed: int = None,
    opponent_type: str = "rule",
) -> RLAgent:
    """
    Train a single RLAgent against opponents.

    Args:
        num_episodes: Number of complete games to play.
        num_players: Table size (5-10).
        rl_player_id: Seat of the RL agent (0-indexed).
        save_path: Where to write the final Q-table.
        seed: Random seed for reproducibility.
        opponent_type: ``"rule"``, ``"random"``, or ``"mixed"``.

    Returns:
        The trained RLAgent.

    Note:
        Training uses alpha=0.25 and epsilon=0.25 (higher than the
        RLAgent class defaults of 0.1/0.2) for faster convergence
        in limited episodes.  Epsilon decays to 0.05 over training.
    """
    rng = random.Random(seed)

    rl_agent = RLAgent(rl_player_id, alpha=0.25, epsilon=0.25, seed=seed)
    wins = 0
    losses = 0

    print(f"Training RLAgent (player {rl_player_id}) for {num_episodes} episodes…")

    for episode in range(1, num_episodes + 1):
        game_seed = rng.randint(0, 2**31)
        game = Game(num_players=num_players, seed=game_seed)

        rl_agent.reset_episode()
        rl_agent.player_id = rl_player_id

        # Curriculum: use easier opponents early when training against
        # rule-based, so the agent learns fundamentals first.
        curriculum_cutoff = int(num_episodes * 0.5)
        if opponent_type == "rule" and episode <= curriculum_cutoff:
            effective_opp = "random"
        else:
            effective_opp = opponent_type

        # Bump exploration when transitioning to harder opponents
        if opponent_type == "rule" and episode == curriculum_cutoff + 1:
            rl_agent.epsilon = max(rl_agent.epsilon, 0.15)

        # Assign agents
        agents = _make_opponents(num_players, rl_player_id, effective_opp, game_seed, rng)
        agents[rl_player_id] = rl_agent

        run_game(agents, game)

        reward = _compute_reward(game, rl_player_id)
        if reward > 0:
            wins += 1
        else:
            losses += 1

        rl_agent.end_episode(reward)

        # Decay epsilon
        rl_agent.epsilon = max(0.05, rl_agent.epsilon * 0.997)

        if episode % 1000 == 0:
            total = wins + losses
            win_rate = wins / total if total else 0
            print(
                f"  Episode {episode:>7}/{num_episodes} | "
                f"win rate: {win_rate:.2%} | "
                f"epsilon: {rl_agent.epsilon:.4f} | "
                f"Q-table size: {sum(len(v) for v in rl_agent._qtable.values())}"
            )

    rl_agent.save(save_path)
    print(f"\nTraining complete. Q-table saved to '{save_path}'.")
    print(f"Final win rate: {wins / (wins + losses):.2%} ({wins}/{wins + losses})")
    return rl_agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    rl_agent: RLAgent,
    num_episodes: int = 1000,
    num_players: int = 6,
    opponent_type: str = "rule",
    seed: int = None,
) -> dict:
    """Evaluate a trained RL agent's performance.

    Args:
        rl_agent: Pre-trained agent (epsilon will be temporarily set to 0).
        num_episodes: Games to play.
        num_players: Table size.
        opponent_type: ``"random"``, ``"rule"``, or ``"mixed"``.
        seed: Random seed.

    Returns:
        Dictionary with win/loss stats and per-role breakdown.
    """
    rng = random.Random(seed)
    saved_epsilon = rl_agent.epsilon
    rl_agent.epsilon = 0.0  # pure exploitation

    wins = 0
    losses = 0
    liberal_wins = 0
    liberal_total = 0
    fascist_wins = 0
    fascist_total = 0

    rl_pid = 0  # always seat 0 for evaluation

    for _ in range(num_episodes):
        game_seed = rng.randint(0, 2**31)
        game = Game(num_players=num_players, seed=game_seed)

        rl_agent.reset_episode()
        rl_agent.player_id = rl_pid

        agents = _make_opponents(num_players, rl_pid, opponent_type, game_seed, rng)
        agents[rl_pid] = rl_agent

        run_game(agents, game)

        reward = _compute_reward(game, rl_pid)
        is_liberal = game.players[rl_pid].role == Role.LIBERAL

        if reward > 0:
            wins += 1
            if is_liberal:
                liberal_wins += 1
            else:
                fascist_wins += 1
        else:
            losses += 1

        if is_liberal:
            liberal_total += 1
        else:
            fascist_total += 1

    rl_agent.epsilon = saved_epsilon  # restore

    total = wins + losses
    result = {
        "opponent_type": opponent_type,
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total else 0.0,
        "liberal_wins": liberal_wins,
        "liberal_total": liberal_total,
        "liberal_win_rate": liberal_wins / liberal_total if liberal_total else 0.0,
        "fascist_wins": fascist_wins,
        "fascist_total": fascist_total,
        "fascist_win_rate": fascist_wins / fascist_total if fascist_total else 0.0,
    }

    print(
        f"  vs {opponent_type:>6}: "
        f"win rate {result['win_rate']:.2%} ({wins}/{total}) | "
        f"as Liberal {result['liberal_win_rate']:.2%} ({liberal_wins}/{liberal_total}) | "
        f"as Fascist {result['fascist_win_rate']:.2%} ({fascist_wins}/{fascist_total})"
    )
    return result


def benchmark(
    rl_agent: RLAgent,
    num_episodes: int = 1000,
    num_players: int = 6,
    seed: int = None,
) -> List[dict]:
    """Run evaluation against all three opponent types and print a comparison.

    Returns:
        List of result dicts (one per opponent type).
    """
    print(f"\n{'=' * 72}")
    print(f"  Benchmark: {num_episodes} episodes per opponent type, {num_players} players")
    print(f"{'=' * 72}")

    results = []
    for opp in ("random", "rule", "mixed"):
        r = evaluate(rl_agent, num_episodes=num_episodes, num_players=num_players,
                     opponent_type=opp, seed=seed)
        results.append(r)

    print(f"{'=' * 72}\n")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Secret Hitler")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--seat", type=int, default=0, help="RL agent seat (0-indexed)")
    parser.add_argument("--save", default="qtable.json", help="Output Q-table path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--opponent", choices=["random", "rule", "mixed"], default="rule",
                        help="Opponent type (default: rule)")
    args = parser.parse_args()

    agent = train(
        num_episodes=args.episodes,
        num_players=args.players,
        rl_player_id=args.seat,
        save_path=args.save,
        seed=args.seed,
        opponent_type=args.opponent,
    )
    benchmark(agent, num_episodes=500, num_players=args.players, seed=args.seed)


if __name__ == "__main__":
    main()
