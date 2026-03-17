#!/usr/bin/env python3
"""
Secret Hitler — main entry point.

Usage:
    python main.py              # start CLI game
    python main.py --gui        # start Kivy GUI (desktop / Android)
    python main.py --train      # train the RL agent
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
    from ai.train import train
    train(
        num_episodes=args.episodes,
        num_players=args.players,
        rl_player_id=args.seat,
        save_path=args.save,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Secret Hitler — Python implementation with AI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gui", action="store_true", help="Launch the Kivy GUI")
    parser.add_argument("--train", action="store_true", help="Train the RL agent")
    parser.add_argument("--players", type=int, default=6, choices=range(5, 11),
                        metavar="{5..10}", help="Number of players (default: 6)")
    parser.add_argument("--seat", type=int, default=0,
                        help="Human / RL-agent seat index (default: 0)")
    parser.add_argument("--ai", choices=["rule", "rl"], default="rule",
                        help="AI opponent type (default: rule)")
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
    elif args.gui:
        _gui_play(args)
    else:
        _cli_play(args)


if __name__ == "__main__":
    main()
