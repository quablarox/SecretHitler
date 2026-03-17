"""CLI interface for Secret Hitler (single-player vs AI)."""
from __future__ import annotations

import sys
from typing import List, Optional

from secret_hitler.game import Game, Phase
from secret_hitler.roles import Party, Role
from ai.rule_based_agent import RuleBasedAgent
from ai.rl_agent import RLAgent


# ANSI colours
_RED = "\033[91m"
_BLUE = "\033[94m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

FASCIST_CLR = _RED
LIBERAL_CLR = _BLUE


def _party_str(party: Party) -> str:
    if party == Party.FASCIST:
        return f"{FASCIST_CLR}Fascist{_RESET}"
    return f"{LIBERAL_CLR}Liberal{_RESET}"


def _policy_str(tile: Party) -> str:
    if tile == Party.FASCIST:
        return f"{FASCIST_CLR}F{_RESET}"
    return f"{LIBERAL_CLR}L{_RESET}"


def _tiles_str(tiles: List[Party]) -> str:
    return "[" + ", ".join(_policy_str(t) for t in tiles) + "]"


def _print_board(game: Game) -> None:
    b = game.board
    print(
        f"\n{_BOLD}Board{_RESET}  "
        f"{LIBERAL_CLR}Liberal{_RESET}: {b.liberal_policies}/5  "
        f"{FASCIST_CLR}Fascist{_RESET}: {b.fascist_policies}/6  "
        f"Election tracker: {b.election_tracker}/3"
    )
    alive = ", ".join(game.players[pid].name for pid in game.alive_ids)
    print(f"Alive: {alive}")


def _input_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            val = int(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  Please enter a number between {lo} and {hi}.")
        except (ValueError, EOFError):
            print("  Invalid input. Try again.")


def _input_bool(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("j", "ja", "y", "yes", "1"):
            return True
        if ans in ("n", "nein", "no", "0"):
            return False
        print("  Enter 'ja' or 'nein'.")


class CLIGame:
    """Run a single-player Secret Hitler game in the terminal."""

    def __init__(
        self,
        num_players: int = 6,
        human_seat: int = 0,
        ai_type: str = "rule",
        qtable_path: Optional[str] = None,
        seed: int = None,
    ):
        self.game = Game(num_players=num_players, seed=seed)
        self.human_id = human_seat
        self.agents: List = []

        for pid in range(num_players):
            if pid == human_seat:
                self.agents.append(None)  # human
            elif ai_type == "rl" and qtable_path:
                self.agents.append(RLAgent(pid, qtable_path=qtable_path))
            else:
                self.agents.append(RuleBasedAgent(pid, seed=(seed or 0) + pid))

    def _agent(self, pid: int):
        return self.agents[pid]

    def _is_human(self, pid: int) -> bool:
        return pid == self.human_id

    # ------------------------------------------------------------------
    # Intro / reveal
    # ------------------------------------------------------------------

    def _show_role(self) -> None:
        g = self.game
        player = g.players[self.human_id]
        role_clr = LIBERAL_CLR if player.is_liberal else FASCIST_CLR
        print(f"\n{_BOLD}Your role:{_RESET} {role_clr}{player.role.value}{_RESET}")
        if player.is_fascist or (player.is_hitler and g.num_players <= 6):
            fascists = [
                p.name for p in g.players
                if p.role == Role.FASCIST and p.player_id != self.human_id
            ]
            hitler = next(
                (p.name for p in g.players if p.role == Role.HITLER), None
            )
            if player.is_fascist and fascists:
                print(f"  Fellow Fascists: {', '.join(fascists)}")
            if player.is_fascist:
                print(f"  Hitler: {hitler}")
            if player.is_hitler:
                print(f"  Fascist teammates: {', '.join(fascists)}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        g = self.game
        print(f"\n{'='*50}")
        print(f"  {_BOLD}SECRET HITLER{_RESET}  —  {g.num_players} players")
        print(f"{'='*50}")
        for pid, p in enumerate(g.players):
            marker = " ← YOU" if pid == self.human_id else ""
            print(f"  Seat {pid}: {p.name}{marker}")
        self._show_role()
        input("\nPress Enter to begin…")

        while not g.is_over():
            _print_board(g)
            self._step()

        # Game over
        _print_board(g)
        res = g.result
        winner_clr = LIBERAL_CLR if res.winner == Party.LIBERAL else FASCIST_CLR
        print(
            f"\n{_BOLD}GAME OVER{_RESET}  "
            f"{winner_clr}{res.winner.value}s win!{_RESET}  "
            f"({res.reason.value})"
        )
        print("\nPlayer roles:")
        for p in g.players:
            alive_str = "" if p.is_alive else " [DEAD]"
            role_clr = LIBERAL_CLR if p.is_liberal else FASCIST_CLR
            print(f"  {p.name}: {role_clr}{p.role.value}{_RESET}{alive_str}")

    def _step(self) -> None:
        g = self.game
        phase = g.phase
        pres_id = g.president_idx

        print(f"\n{_BOLD}Phase: {phase.name}{_RESET}  |  President: {g.president.name}")

        if phase == Phase.NOMINATION:
            self._do_nomination(pres_id)

        elif phase == Phase.VOTE:
            self._do_vote()

        elif phase == Phase.LEGISLATIVE_PRESIDENT:
            self._do_pres_legislative(pres_id)

        elif phase == Phase.LEGISLATIVE_CHANCELLOR:
            self._do_chan_legislative(g.chancellor_idx)

        elif phase == Phase.VETO_REQUESTED:
            self._do_veto_response(pres_id)

        elif phase == Phase.PRESIDENTIAL_POWER:
            self._do_presidential_power(pres_id)

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _do_nomination(self, pres_id: int) -> None:
        g = self.game
        eligible = g.eligible_chancellors()
        if self._is_human(pres_id):
            print("  Eligible chancellors:")
            for i, p in enumerate(eligible):
                print(f"    {i}: {p.name} (seat {p.player_id})")
            idx = _input_int("  Your choice (index): ", 0, len(eligible) - 1)
            g.nominate_chancellor(eligible[idx].player_id)
        else:
            choice = self._agent(pres_id).nominate_chancellor(g)
            print(f"  {g.president.name} nominates {g.players[choice].name} as Chancellor.")
            g.nominate_chancellor(choice)

    def _do_vote(self) -> None:
        g = self.game
        print(f"  Proposed government: {g.president.name} (President) + {g.chancellor.name} (Chancellor)")
        for pid in g.alive_ids:
            if self._is_human(pid):
                vote = _input_bool("  Vote Ja!/Nein! [ja/nein]: ")
            else:
                vote = self._agent(pid).vote(g)
            result = g.cast_vote(pid, vote)
            if result is not None:
                # All votes tallied
                ja = sum(1 for v in g.votes if v)
                nein = len(g.alive_ids) - ja
                print(f"  Result: {ja} Ja / {nein} Nein — {'ELECTED' if ja > nein else 'REJECTED'}")
                break

    def _do_pres_legislative(self, pres_id: int) -> None:
        g = self.game
        tiles = g._drawn_tiles
        if self._is_human(pres_id):
            print(f"  You drew: {_tiles_str(tiles)}")
            idx = _input_int("  Discard which tile? (0/1/2): ", 0, len(tiles) - 1)
        else:
            idx = self._agent(pres_id).president_discard(g, tiles)
            print(f"  {g.president.name} (President) discards a tile.")
        g.president_discard(idx)

    def _do_chan_legislative(self, chan_id: int) -> None:
        g = self.game
        tiles = g._chancellor_tiles
        if self._is_human(chan_id):
            print(f"  You received: {_tiles_str(tiles)}")
            # Offer veto if unlocked and not already rejected this round
            if g.board.veto_unlocked and not g._veto_rejected:
                if _input_bool("  Request veto? [ja/nein]: "):
                    g.chancellor_request_veto()
                    return
            idx = _input_int("  Discard which tile? (0/1): ", 0, len(tiles) - 1)
        else:
            agent = self._agent(chan_id)
            if g.board.veto_unlocked and not g._veto_rejected and agent.request_veto(g, tiles):
                print(f"  {g.chancellor.name} (Chancellor) requests a veto!")
                g.chancellor_request_veto()
                return
            idx = agent.chancellor_discard(g, tiles)
            enacted = tiles[1 - idx]
            print(f"  {g.chancellor.name} (Chancellor) enacts a {_party_str(enacted)} policy.")
        g.chancellor_discard(idx)

    def _do_veto_response(self, pres_id: int) -> None:
        g = self.game
        print(f"  {g.chancellor.name} has requested a veto!")
        if self._is_human(pres_id):
            accept = _input_bool("  Accept veto? [ja/nein]: ")
        else:
            accept = self._agent(pres_id).respond_veto(g)
            print(f"  {g.president.name} {'accepts' if accept else 'rejects'} the veto.")
        g.president_respond_veto(accept)

    def _do_presidential_power(self, pres_id: int) -> None:
        g = self.game
        power = g.pending_power
        print(f"  Presidential power: {power}")

        if power == "policy_peek":
            if self._is_human(pres_id):
                tiles = g.deck.peek(3)
                print(f"  Top 3 policies: {_tiles_str(tiles)}")
                input("  (Press Enter to continue…)")
            g.use_policy_peek()

        elif power == "investigate":
            if self._is_human(pres_id):
                candidates = [pid for pid in g.alive_ids if pid != pres_id]
                print("  Investigate which player?")
                for i, pid in enumerate(candidates):
                    print(f"    {i}: {g.players[pid].name}")
                idx = _input_int("  Choice: ", 0, len(candidates) - 1)
                target = candidates[idx]
            else:
                target = self._agent(pres_id).choose_investigate_target(g)
            party = g.use_investigate(target)
            if self._is_human(pres_id):
                print(f"  {g.players[target].name}'s party: {_party_str(party)}")

        elif power == "special_election":
            if self._is_human(pres_id):
                candidates = [pid for pid in g.alive_ids if pid != pres_id]
                print("  Choose next Presidential candidate:")
                for i, pid in enumerate(candidates):
                    print(f"    {i}: {g.players[pid].name}")
                idx = _input_int("  Choice: ", 0, len(candidates) - 1)
                target = candidates[idx]
            else:
                target = self._agent(pres_id).choose_special_election_target(g)
                print(f"  {g.president.name} picks {g.players[target].name} for special election.")
            g.use_special_election(target)

        elif power == "execution":
            if self._is_human(pres_id):
                candidates = [pid for pid in g.alive_ids if pid != pres_id]
                print("  Execute which player?")
                for i, pid in enumerate(candidates):
                    print(f"    {i}: {g.players[pid].name}")
                idx = _input_int("  Choice: ", 0, len(candidates) - 1)
                target = candidates[idx]
            else:
                target = self._agent(pres_id).choose_execution_target(g)
                print(f"  {g.president.name} executes {g.players[target].name}!")
            g.use_execution(target)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Play Secret Hitler (CLI)")
    parser.add_argument("--players", type=int, default=6, choices=range(5, 11))
    parser.add_argument("--seat", type=int, default=0, help="Your player seat (0-indexed)")
    parser.add_argument("--ai", choices=["rule", "rl"], default="rule")
    parser.add_argument("--qtable", default=None, help="Q-table path for RL agent")
    parser.add_argument("--seed", type=int, default=None)
    parsed = parser.parse_args(args)

    cli = CLIGame(
        num_players=parsed.players,
        human_seat=parsed.seat,
        ai_type=parsed.ai,
        qtable_path=parsed.qtable,
        seed=parsed.seed,
    )
    cli.run()


if __name__ == "__main__":
    main()
