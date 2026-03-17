#!/usr/bin/env python3
"""
Secret Hitler — main entry point.

Usage:
    python main.py              # start CLI game
    python main.py --gui        # start Kivy GUI (desktop / Android)
    python main.py --train      # train the RL agent
    python main.py --evaluate   # benchmark a pre-trained Q-table
    python main.py --agents-only 500  # run 500 all-agent games
    python main.py --help       # show all options
"""
import argparse
import sys


def _cli_play(args):
    from ui.cli import main as cli_main
    cli_main(
        [
            "--players", str(args.players),
            "--seat", str(args.seat),
            "--ai", args.ai,
        ]
        + (["--qtable", args.qtable] if args.qtable else [])
        + (["--seed", str(args.seed)] if args.seed is not None else [])
    )


def _gui_play(args):
    from ui.kivy_app import SecretHitlerApp
    SecretHitlerApp().run()


def _train(args):
    from ai.train import train, benchmark
    agent = train(
        num_episodes=args.episodes,
        num_players=args.players,
        rl_player_id=args.seat,
        save_path=args.save,
        seed=args.seed,
        opponent_type=args.opponent,
    )
    benchmark(agent, num_episodes=500, num_players=args.players, seed=args.seed)


def _evaluate(args):
    from ai.rl_agent import RLAgent
    from ai.train import benchmark

    if not args.qtable:
        print("Error: --qtable is required for --evaluate", file=sys.stderr)
        sys.exit(1)
    agent = RLAgent(0, qtable_path=args.qtable)
    benchmark(agent, num_episodes=args.episodes or 1000,
              num_players=args.players, seed=args.seed)


def _agents_only(args):
    from ai.train import run_game, _compute_reward
    from ai.random_agent import RandomAgent
    from ai.rule_based_agent import RuleBasedAgent
    from secret_hitler.game import Game
    from secret_hitler.roles import Party
    import random

    rng = random.Random(args.seed)
    num_games = args.agents_only
    wins = {Party.LIBERAL: 0, Party.FASCIST: 0}

    for i in range(num_games):
        game_seed = rng.randint(0, 2**31)
        game = Game(num_players=args.players, seed=game_seed)
        agents = []
        for pid in range(args.players):
            if args.opponent == "random":
                agents.append(RandomAgent(pid, seed=game_seed + pid))
            elif args.opponent == "rule":
                agents.append(RuleBasedAgent(pid, seed=game_seed + pid))
            else:
                if rng.random() < 0.5:
                    agents.append(RandomAgent(pid, seed=game_seed + pid))
                else:
                    agents.append(RuleBasedAgent(pid, seed=game_seed + pid))
        run_game(agents, game)
        wins[game.result.winner] += 1

    total = sum(wins.values())
    print(f"\nAll-agent games ({args.opponent}): {total} games, {args.players} players")
    for party, count in wins.items():
        print(f"  {party.value}: {count} wins ({count / total:.2%})")


def main():
    parser = argparse.ArgumentParser(
        description="Secret Hitler — Python implementation with AI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", help="Launch the Kivy GUI")
    parser.add_argument("--train", action="store_true", help="Train the RL agent")
    parser.add_argument("--evaluate", action="store_true",
                        help="Benchmark a pre-trained Q-table")
    parser.add_argument("--agents-only", type=int, default=None, metavar="N",
                        help="Run N all-agent games and show results")
    parser.add_argument("--players", type=int, default=6, choices=range(5, 11),
                        metavar="{5..10}", help="Number of players (default: 6)")
    parser.add_argument("--seat", type=int, default=0,
                        help="Human / RL-agent seat index (default: 0)")
    parser.add_argument("--ai", choices=["rule", "rl"], default="rule",
                        help="AI opponent type (default: rule)")
    parser.add_argument("--opponent", choices=["random", "rule", "mixed"],
                        default="rule",
                        help="Opponent type for training / evaluation (default: rule)")
    parser.add_argument("--qtable", default=None,
                        help="Path to Q-table JSON for the RL agent")
    parser.add_argument("--episodes", type=int, default=10_000,
                        help="Training episodes (default: 10 000)")
    parser.add_argument("--save", default="qtable.json",
                        help="Where to save the trained Q-table (default: qtable.json)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.train:
        _train(args)
    elif args.evaluate:
        _evaluate(args)
    elif args.agents_only is not None:
        _agents_only(args)
    elif args.gui:
        _gui_play(args)
    else:
        _cli_play(args)


if __name__ == "__main__":
    main()
