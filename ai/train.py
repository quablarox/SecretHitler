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


# ---------------------------------------------------------------------------
# Game runner — resolves all decisions via agents
# ---------------------------------------------------------------------------

def run_game(agents: List, game: Game) -> Game:
    """Play through a full game, returning the completed Game object."""
    while not game.is_over():
        phase = game.phase
        pres_id = game.president_idx

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
    return game


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    num_episodes: int = 10_000,
    num_players: int = 6,
    rl_player_id: int = 0,
    save_path: str = "qtable.json",
    seed: int = None,
) -> RLAgent:
    """
    Train a single RLAgent against RuleBasedAgent opponents.

    Args:
        num_episodes: Number of complete games to play.
        num_players: Table size (5-10).
        rl_player_id: Seat of the RL agent (0-indexed).
        save_path: Where to write the final Q-table.
        seed: Random seed for reproducibility.

    Returns:
        The trained RLAgent.
    """
    rng = random.Random(seed)

    rl_agent = RLAgent(rl_player_id, epsilon=0.3, seed=seed)
    wins = 0
    losses = 0

    print(f"Training RLAgent (player {rl_player_id}) for {num_episodes} episodes…")

    for episode in range(1, num_episodes + 1):
        game_seed = rng.randint(0, 2**31)
        game = Game(num_players=num_players, seed=game_seed)

        # Assign agents — RL at fixed seat, rule-based for the rest
        agents = []
        for pid in range(num_players):
            if pid == rl_player_id:
                rl_agent.player_id = pid
                agents.append(rl_agent)
            else:
                agents.append(RuleBasedAgent(pid, seed=game_seed + pid))

        run_game(agents, game)

        reward = _compute_reward(game, rl_player_id)
        if reward > 0:
            wins += 1
        else:
            losses += 1

        rl_agent.update(reward, None)

        # Decay epsilon slowly
        rl_agent.epsilon = max(0.05, rl_agent.epsilon * 0.9999)

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
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Secret Hitler")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--seat", type=int, default=0, help="RL agent seat (0-indexed)")
    parser.add_argument("--save", default="qtable.json", help="Output Q-table path")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        num_players=args.players,
        rl_player_id=args.seat,
        save_path=args.save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
