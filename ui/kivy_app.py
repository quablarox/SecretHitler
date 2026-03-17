"""
Kivy-based Secret Hitler UI for Android (and desktop).

Architecture:
  - ScreenManager with multiple screens:
      MenuScreen       -> main menu (new game / settings / train AI)
      SetupScreen      -> player count / AI type selection
      GameScreen       -> main game board
      RoleRevealScreen -> show the human player their secret role
      VoteScreen       -> vote on the proposed government
      LegislativeScreen -> legislative session (president / chancellor)
      PowerScreen      -> presidential power actions
      ResultScreen     -> game over screen

Run with:
    python ui/kivy_app.py
"""
from __future__ import annotations

import os
import sys

# Ensure repo root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.utils import get_color_from_hex

# Game engine
from secret_hitler.game import Game, Phase
from secret_hitler.roles import Party, Role
from ai.rule_based_agent import RuleBasedAgent
from ai.rl_agent import RLAgent

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
LIBERAL_COLOR = get_color_from_hex("#4A90D9FF")
FASCIST_COLOR = get_color_from_hex("#C0392BFF")
NEUTRAL_COLOR = get_color_from_hex("#2C3E50FF")
ACCENT_COLOR  = get_color_from_hex("#F39C12FF")
BG_COLOR      = get_color_from_hex("#1A252FFF")
TEXT_COLOR    = get_color_from_hex("#ECF0F1FF")

# ---------------------------------------------------------------------------
# Reusable widget helpers
# ---------------------------------------------------------------------------

def _btn(text: str, callback, size_hint_y=None, height=dp(50), **kw) -> Button:
    b = Button(
        text=text,
        size_hint_y=size_hint_y,
        height=height,
        background_color=ACCENT_COLOR,
        color=TEXT_COLOR,
        font_size=dp(16),
        **kw,
    )
    b.bind(on_press=callback)
    return b


def _lbl(text: str, **kw) -> Label:
    return Label(
        text=text,
        color=TEXT_COLOR,
        font_size=dp(14),
        markup=True,
        **kw,
    )


def _policy_symbol(tile: Party) -> str:
    if tile == Party.FASCIST:
        return "[color=ff4444]F[/color]"
    return "[color=4499ff]L[/color]"


# ---------------------------------------------------------------------------
# Shared game state — passed between screens via the App
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.game: Game | None = None
        self.agents: list = []
        self.human_id: int = 0
        self.num_players: int = 6
        self.ai_type: str = "rule"
        self.qtable_path: str | None = None
        self.seed: int | None = None


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

class MenuScreen(Screen):
    def __init__(self, state: AppState, **kw):
        super().__init__(name="menu", **kw)
        self.state = state
        layout = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(10))
        layout.add_widget(Label(size_hint_y=0.1))
        layout.add_widget(_lbl(
            "[b][size=28]SECRET HITLER[/size][/b]",
            size_hint_y=0.15, halign="center", valign="middle",
        ))
        layout.add_widget(_lbl(
            "A game of political deception",
            size_hint_y=0.08, halign="center",
        ))
        layout.add_widget(Label(size_hint_y=0.05))
        layout.add_widget(_btn("New Game", self._new_game))
        layout.add_widget(_btn("How to Play", self._how_to_play))
        layout.add_widget(Label(size_hint_y=0.3))
        self.add_widget(layout)

    def _new_game(self, *_):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "setup"

    def _how_to_play(self, *_):
        text = (
            "[b]SECRET HITLER — Rules Summary[/b]\n\n"
            "• Players are secretly Liberals, Fascists, or Hitler.\n"
            "• Each round, a President nominates a Chancellor.\n"
            "• All players vote. If elected, they draw & enact a policy.\n"
            "• [color=4499ff]Liberals[/color] win by enacting 5 Liberal policies "
            "or shooting Hitler.\n"
            "• [color=ff4444]Fascists[/color] win by enacting 6 Fascist policies "
            "or electing Hitler as Chancellor (after 3 Fascist policies).\n"
            "• If 3 elections in a row fail, the top policy is enacted automatically."
        )
        popup = Popup(
            title="How to Play",
            content=_lbl(text, halign="left", valign="top"),
            size_hint=(0.9, 0.7),
        )
        popup.open()


class SetupScreen(Screen):
    def __init__(self, state: AppState, **kw):
        super().__init__(name="setup", **kw)
        self.state = state
        layout = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(12))
        layout.add_widget(_lbl("[b]Game Setup[/b]", size_hint_y=None, height=dp(40), halign="center"))

        layout.add_widget(_lbl("Number of Players (5-10):", size_hint_y=None, height=dp(30)))
        self.player_spinner = Spinner(
            text="6",
            values=[str(n) for n in range(5, 11)],
            size_hint_y=None, height=dp(44),
        )
        layout.add_widget(self.player_spinner)

        layout.add_widget(_lbl("Your seat (0-indexed):", size_hint_y=None, height=dp(30)))
        self.seat_spinner = Spinner(
            text="0",
            values=[str(n) for n in range(10)],
            size_hint_y=None, height=dp(44),
        )
        layout.add_widget(self.seat_spinner)

        layout.add_widget(_lbl("AI Opponent Type:", size_hint_y=None, height=dp(30)))
        self.ai_spinner = Spinner(
            text="Rule-based",
            values=["Rule-based", "RL (trained)"],
            size_hint_y=None, height=dp(44),
        )
        layout.add_widget(self.ai_spinner)

        layout.add_widget(Label())
        layout.add_widget(_btn("Start Game", self._start))
        layout.add_widget(_btn("Back", self._back, height=dp(40)))
        self.add_widget(layout)

    def _start(self, *_):
        num_players = int(self.player_spinner.text)
        seat = min(int(self.seat_spinner.text), num_players - 1)
        ai_type = "rl" if "RL" in self.ai_spinner.text else "rule"

        st = self.state
        st.num_players = num_players
        st.human_id = seat
        st.ai_type = ai_type

        # Build game
        st.game = Game(num_players=num_players, seed=st.seed)
        st.agents = []
        for pid in range(num_players):
            if pid == seat:
                st.agents.append(None)
            elif ai_type == "rl" and st.qtable_path:
                st.agents.append(RLAgent(pid, qtable_path=st.qtable_path))
            else:
                st.agents.append(RuleBasedAgent(pid, seed=(st.seed or 0) + pid))

        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "role_reveal"

    def _back(self, *_):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "menu"


class RoleRevealScreen(Screen):
    def __init__(self, state: AppState, **kw):
        super().__init__(name="role_reveal", **kw)
        self.state = state
        self.layout = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(10))
        self.add_widget(self.layout)

    def on_enter(self, *_):
        self.layout.clear_widgets()
        g = self.state.game
        player = g.players[self.state.human_id]

        if player.is_liberal:
            color_hex = "4499ff"
        else:
            color_hex = "ff4444"

        self.layout.add_widget(_lbl(
            f"[b]Your Secret Role[/b]",
            size_hint_y=0.2, halign="center",
        ))
        self.layout.add_widget(_lbl(
            f"[b][size=32][color={color_hex}]{player.role.value}[/color][/size][/b]",
            size_hint_y=0.3, halign="center",
        ))

        info_lines = []
        if player.is_fascist:
            fascists = [p.name for p in g.players if p.role == Role.FASCIST and p.player_id != self.state.human_id]
            hitler = next((p.name for p in g.players if p.role == Role.HITLER), "?")
            if fascists:
                info_lines.append(f"Fellow Fascists: {', '.join(fascists)}")
            info_lines.append(f"Hitler: {hitler}")
        elif player.is_hitler and g.num_players <= 6:
            fascists = [p.name for p in g.players if p.role == Role.FASCIST]
            info_lines.append(f"Fascist teammates: {', '.join(fascists)}")
        else:
            info_lines.append("You don't know who the Fascists are.")
            info_lines.append("Enact 5 Liberal policies or shoot Hitler to win!")

        self.layout.add_widget(_lbl("\n".join(info_lines), size_hint_y=0.25, halign="center"))
        self.layout.add_widget(_btn("Begin Game", self._start))

    def _start(self, *_):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "game"
        game_screen = self.manager.get_screen("game")
        game_screen.advance()


class GameScreen(Screen):
    """Main game screen — shows the board and drives the AI phases automatically."""

    def __init__(self, state: AppState, **kw):
        super().__init__(name="game", **kw)
        self.state = state
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))

        # --- Board header ---
        self.board_lbl = _lbl("", size_hint_y=None, height=dp(80), halign="center")
        root.add_widget(self.board_lbl)

        # --- Log scroll ---
        self.log_layout = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(2))
        self.log_layout.bind(minimum_height=self.log_layout.setter("height"))
        scroll = ScrollView()
        scroll.add_widget(self.log_layout)
        root.add_widget(scroll)

        # --- Action area ---
        self.action_area = BoxLayout(
            orientation="vertical", size_hint_y=None, height=dp(180), spacing=dp(6),
        )
        root.add_widget(self.action_area)

        self.add_widget(root)

    def _log(self, text: str) -> None:
        lbl = _lbl(text, size_hint_y=None, height=dp(24), halign="left")
        self.log_layout.add_widget(lbl)

    def _update_board(self) -> None:
        g = self.state.game
        b = g.board
        lib = "[color=4499ff]" + "■" * b.liberal_policies + "□" * (5 - b.liberal_policies) + "[/color]"
        fas = "[color=ff4444]" + "■" * b.fascist_policies + "□" * (6 - b.fascist_policies) + "[/color]"
        alive = len(g.alive_players)
        self.board_lbl.text = (
            f"[b]Liberal:[/b] {lib}  [b]Fascist:[/b] {fas}\n"
            f"Alive: {alive}  Election: {b.election_tracker}/3  "
            f"President: {g.president.name}"
        )

    def _clear_actions(self) -> None:
        self.action_area.clear_widgets()

    def advance(self) -> None:
        """Drive AI decisions automatically; pause for human decisions."""
        g = self.state.game
        self._update_board()

        if g.is_over():
            self._update_board()
            self._log(f"[b]GAME OVER![/b] {g.result.winner.value}s win — {g.result.reason.value}")
            self._show_result()
            return

        phase = g.phase
        pres_id = g.president_idx
        human = self.state.human_id

        self._log(f"[b]{phase.name}[/b] — President: {g.president.name}")

        # -------- NOMINATION --------
        if phase == Phase.NOMINATION:
            if pres_id == human:
                self._ask_nomination()
            else:
                agent = self.state.agents[pres_id]
                choice = agent.nominate_chancellor(g)
                g.nominate_chancellor(choice)
                self._log(f"  {g.president.name} nominates {g.players[choice].name}")
                self.advance()

        # -------- VOTE --------
        elif phase == Phase.VOTE:
            self._collect_votes()

        # -------- LEGISLATIVE PRESIDENT --------
        elif phase == Phase.LEGISLATIVE_PRESIDENT:
            if pres_id == human:
                self._ask_pres_discard()
            else:
                agent = self.state.agents[pres_id]
                idx = agent.president_discard(g, g._drawn_tiles)
                g.president_discard(idx)
                self._log(f"  {g.president.name} discards a tile.")
                self.advance()

        # -------- LEGISLATIVE CHANCELLOR --------
        elif phase == Phase.LEGISLATIVE_CHANCELLOR:
            chan_id = g.chancellor_idx
            if chan_id == human:
                self._ask_chan_discard()
            else:
                agent = self.state.agents[chan_id]
                if (g.board.veto_unlocked
                        and not g._veto_rejected
                        and agent.request_veto(g, g._chancellor_tiles)):
                    g.chancellor_request_veto()
                    self._log(f"  {g.chancellor.name} requests a veto!")
                    self.advance()
                else:
                    idx = agent.chancellor_discard(g, g._chancellor_tiles)
                    enacted = g._chancellor_tiles[1 - idx]
                    self._log(f"  {g.chancellor.name} enacts a {enacted.value} policy.")
                    g.chancellor_discard(idx)
                    self.advance()

        # -------- VETO REQUESTED --------
        elif phase == Phase.VETO_REQUESTED:
            if pres_id == human:
                self._ask_veto_response()
            else:
                accept = self.state.agents[pres_id].respond_veto(g)
                g.president_respond_veto(accept)
                self._log(f"  {g.president.name} {'accepts' if accept else 'rejects'} veto.")
                self.advance()

        # -------- PRESIDENTIAL POWER --------
        elif phase == Phase.PRESIDENTIAL_POWER:
            if pres_id == human:
                self._ask_presidential_power()
            else:
                self._do_ai_power(pres_id)

    # ------------------------------------------------------------------
    # Human input builders
    # ------------------------------------------------------------------

    def _ask_nomination(self) -> None:
        g = self.state.game
        self._clear_actions()
        self.action_area.add_widget(_lbl("[b]Nominate a Chancellor:[/b]",
                                        size_hint_y=None, height=dp(30)))
        for p in g.eligible_chancellors():
            def cb(btn, pid=p.player_id):
                g.nominate_chancellor(pid)
                self._log(f"  You nominate {g.players[pid].name}")
                self._clear_actions()
                self.advance()
            self.action_area.add_widget(_btn(p.name, cb))

    def _collect_votes(self) -> None:
        g = self.state.game
        # Non-human players vote immediately
        for pid in g.alive_ids:
            if pid != self.state.human_id and g.votes[pid] is None:
                vote = self.state.agents[pid].vote(g)
                result = g.cast_vote(pid, vote)
                if result is not None:
                    self._log(f"  Vote resolved: {result.name}")
                    self._clear_actions()
                    self.advance()
                    return

        # Human hasn't voted yet
        self._clear_actions()
        self.action_area.add_widget(_lbl(
            f"[b]Vote on:[/b] {g.president.name} + {g.chancellor.name}",
            size_hint_y=None, height=dp(30),
        ))
        row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        def ja_cb(*_):
            result = g.cast_vote(self.state.human_id, True)
            self._log("  You voted [color=4499ff]Ja![/color]")
            self._clear_actions()
            if result is not None:
                self.advance()
            else:
                self._collect_votes()
        def nein_cb(*_):
            result = g.cast_vote(self.state.human_id, False)
            self._log("  You voted [color=ff4444]Nein![/color]")
            self._clear_actions()
            if result is not None:
                self.advance()
            else:
                self._collect_votes()
        row.add_widget(_btn("Ja!", ja_cb, background_color=LIBERAL_COLOR))
        row.add_widget(_btn("Nein!", nein_cb, background_color=FASCIST_COLOR))
        self.action_area.add_widget(row)

    def _ask_pres_discard(self) -> None:
        g = self.state.game
        tiles = g._drawn_tiles
        self._clear_actions()
        self.action_area.add_widget(_lbl(
            "[b]Discard a policy tile (as President):[/b]",
            size_hint_y=None, height=dp(30),
        ))
        row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(8))
        for i, tile in enumerate(tiles):
            def cb(btn, idx=i):
                g.president_discard(idx)
                self._log(f"  You discarded tile {idx}.")
                self._clear_actions()
                self.advance()
            sym = "Fascist" if tile == Party.FASCIST else "Liberal"
            clr = FASCIST_COLOR if tile == Party.FASCIST else LIBERAL_COLOR
            row.add_widget(_btn(sym, cb, background_color=clr))
        self.action_area.add_widget(row)

    def _ask_chan_discard(self) -> None:
        g = self.state.game
        tiles = g._chancellor_tiles
        self._clear_actions()
        if g.board.veto_unlocked and not g._veto_rejected:
            self.action_area.add_widget(_btn("Request Veto", self._do_veto_request,
                                             background_color=NEUTRAL_COLOR))
        self.action_area.add_widget(_lbl(
            "[b]Discard a policy tile (as Chancellor):[/b]",
            size_hint_y=None, height=dp(30),
        ))
        row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(8))
        for i, tile in enumerate(tiles):
            def cb(btn, idx=i):
                g.chancellor_discard(idx)
                self._log(f"  You (Chancellor) discarded tile {idx}.")
                self._clear_actions()
                self.advance()
            sym = "Fascist" if tile == Party.FASCIST else "Liberal"
            clr = FASCIST_COLOR if tile == Party.FASCIST else LIBERAL_COLOR
            row.add_widget(_btn(sym, cb, background_color=clr))
        self.action_area.add_widget(row)

    def _do_veto_request(self, *_) -> None:
        self.state.game.chancellor_request_veto()
        self._log("  You request a veto!")
        self._clear_actions()
        self.advance()

    def _ask_veto_response(self) -> None:
        g = self.state.game
        self._clear_actions()
        self.action_area.add_widget(_lbl(
            f"[b]{g.chancellor.name} requests a veto! Accept?[/b]",
            size_hint_y=None, height=dp(30),
        ))
        row = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        def accept(*_):
            g.president_respond_veto(True)
            self._log("  You accept the veto.")
            self._clear_actions()
            self.advance()
        def reject(*_):
            g.president_respond_veto(False)
            self._log("  You reject the veto.")
            self._clear_actions()
            self.advance()
        row.add_widget(_btn("Accept", accept, background_color=LIBERAL_COLOR))
        row.add_widget(_btn("Reject", reject, background_color=FASCIST_COLOR))
        self.action_area.add_widget(row)

    def _ask_presidential_power(self) -> None:
        g = self.state.game
        power = g.pending_power
        self._clear_actions()
        self.action_area.add_widget(_lbl(
            f"[b]Presidential Power: {power}[/b]",
            size_hint_y=None, height=dp(30),
        ))

        if power == "policy_peek":
            tiles = g.deck.peek(3)
            symbols = "  ".join(
                ("[color=ff4444]F[/color]" if t == Party.FASCIST else "[color=4499ff]L[/color]")
                for t in tiles
            )
            self.action_area.add_widget(_lbl(
                f"Top 3 policies: {symbols}", size_hint_y=None, height=dp(40),
            ))
            def done(*_):
                g.use_policy_peek()
                self._clear_actions()
                self.advance()
            self.action_area.add_widget(_btn("OK", done))

        elif power in ("investigate", "special_election", "execution"):
            candidates = [pid for pid in g.alive_ids if pid != g.president_idx]
            for pid in candidates:
                def cb(btn, target=pid):
                    self._apply_power(power, target)
                self.action_area.add_widget(_btn(g.players[pid].name, cb))

    def _apply_power(self, power: str, target: int) -> None:
        g = self.state.game
        self._clear_actions()
        if power == "investigate":
            party = g.use_investigate(target)
            self._log(f"  {g.players[target].name} is a {party.value}!")
        elif power == "special_election":
            g.use_special_election(target)
            self._log(f"  {g.players[target].name} will be the next President.")
        elif power == "execution":
            g.use_execution(target)
            self._log(f"  {g.players[target].name} has been executed!")
        self.advance()

    def _do_ai_power(self, pres_id: int) -> None:
        g = self.state.game
        agent = self.state.agents[pres_id]
        power = g.pending_power
        if power == "policy_peek":
            tiles = g.use_policy_peek()
            agent.on_policy_peek(tiles)
            self._log(f"  {g.players[pres_id].name} peeks at top 3 policies.")
        elif power == "investigate":
            target = agent.choose_investigate_target(g)
            party = g.use_investigate(target)
            agent.on_investigate_result(target, party)
            self._log(f"  {g.players[pres_id].name} investigates {g.players[target].name}.")
        elif power == "special_election":
            target = agent.choose_special_election_target(g)
            g.use_special_election(target)
            self._log(f"  {g.players[pres_id].name} picks {g.players[target].name} for special election.")
        elif power == "execution":
            target = agent.choose_execution_target(g)
            g.use_execution(target)
            self._log(f"  {g.players[pres_id].name} executes {g.players[target].name}!")
        self.advance()

    def _show_result(self) -> None:
        result_screen = self.manager.get_screen("result")
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "result"
        result_screen.show_result()


class ResultScreen(Screen):
    def __init__(self, state: AppState, **kw):
        super().__init__(name="result", **kw)
        self.state = state
        self.layout = BoxLayout(orientation="vertical", padding=dp(20), spacing=dp(10))
        self.add_widget(self.layout)

    def show_result(self) -> None:
        self.layout.clear_widgets()
        g = self.state.game
        res = g.result
        if res.winner == Party.LIBERAL:
            clr = "4499ff"
        else:
            clr = "ff4444"
        self.layout.add_widget(_lbl(
            f"[b][size=26][color={clr}]{res.winner.value}s WIN![/color][/size][/b]",
            size_hint_y=0.2, halign="center",
        ))
        self.layout.add_widget(_lbl(res.reason.value, size_hint_y=0.1, halign="center"))

        # Roles reveal
        self.layout.add_widget(_lbl("[b]Player Roles:[/b]", size_hint_y=None, height=dp(30)))
        for p in g.players:
            clr_p = "4499ff" if p.is_liberal else "ff4444"
            dead = " [DEAD]" if not p.is_alive else ""
            you = " ← YOU" if p.player_id == self.state.human_id else ""
            self.layout.add_widget(_lbl(
                f"  {p.name}: [color={clr_p}]{p.role.value}[/color]{dead}{you}",
                size_hint_y=None, height=dp(24),
            ))
        self.layout.add_widget(Label())
        self.layout.add_widget(_btn("Play Again", self._play_again))
        self.layout.add_widget(_btn("Main Menu", self._main_menu))

    def _play_again(self, *_) -> None:
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "setup"

    def _main_menu(self, *_) -> None:
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "menu"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class SecretHitlerApp(App):
    def build(self):
        Window.clearcolor = BG_COLOR
        state = AppState()
        sm = ScreenManager()
        sm.add_widget(MenuScreen(state))
        sm.add_widget(SetupScreen(state))
        sm.add_widget(RoleRevealScreen(state))
        sm.add_widget(GameScreen(state))
        sm.add_widget(ResultScreen(state))
        return sm

    def on_start(self):
        pass


def main():
    SecretHitlerApp().run()


if __name__ == "__main__":
    main()
