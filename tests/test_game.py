"""Tests for the Secret Hitler game engine."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secret_hitler.game import Game, Phase, WinReason
from secret_hitler.roles import Party, Role
from secret_hitler.board import Board
from secret_hitler.deck import PolicyDeck
from secret_hitler.player import Player


# ---------------------------------------------------------------------------
# Deck tests
# ---------------------------------------------------------------------------

class TestPolicyDeck:
    def test_initial_draw(self):
        deck = PolicyDeck()
        tiles = deck.draw(3)
        assert len(tiles) == 3
        assert all(isinstance(t, Party) for t in tiles)

    def test_discard_and_reshuffle(self):
        deck = PolicyDeck()
        # Exhaust the draw pile
        drawn = []
        while deck.draw_pile_size > 0:
            drawn.extend(deck.draw(min(3, deck.draw_pile_size)))
        for t in drawn:
            deck.discard(t)
        # Now drawing should trigger reshuffle
        tiles = deck.draw(3)
        assert len(tiles) == 3

    def test_peek_does_not_consume(self):
        deck = PolicyDeck()
        before = deck.draw_pile_size
        deck.peek(3)
        assert deck.draw_pile_size == before

    def test_total_tile_count(self):
        deck = PolicyDeck()
        all_tiles = deck.draw(17)  # 6 Liberal + 11 Fascist
        assert sum(1 for t in all_tiles if t == Party.LIBERAL) == 6
        assert sum(1 for t in all_tiles if t == Party.FASCIST) == 11


# ---------------------------------------------------------------------------
# Board tests
# ---------------------------------------------------------------------------

class TestBoard:
    def test_enact_liberal(self):
        board = Board(6)
        board.enact(Party.LIBERAL)
        assert board.liberal_policies == 1
        assert board.fascist_policies == 0

    def test_enact_fascist_gives_power(self):
        board = Board(6)
        board.enact(Party.FASCIST)
        board.enact(Party.FASCIST)
        power = board.enact(Party.FASCIST)
        assert power == "policy_peek"

    def test_liberal_win(self):
        board = Board(6)
        for _ in range(5):
            board.enact(Party.LIBERAL)
        assert board.liberal_win

    def test_fascist_win(self):
        board = Board(6)
        for _ in range(6):
            board.enact(Party.FASCIST)
        assert board.fascist_win

    def test_election_tracker_chaos(self):
        board = Board(6)
        assert not board.advance_election_tracker()
        assert not board.advance_election_tracker()
        assert board.advance_election_tracker()  # 3rd → chaos
        assert board.election_tracker == 0  # reset

    def test_veto_unlocked_at_5_fascist_policies(self):
        board = Board(6)
        assert not board.veto_unlocked
        for _ in range(5):
            board.enact(Party.FASCIST)
        assert board.veto_unlocked

    def test_invalid_player_count(self):
        with pytest.raises(ValueError):
            Board(4)
        with pytest.raises(ValueError):
            Board(11)


# ---------------------------------------------------------------------------
# Game construction tests
# ---------------------------------------------------------------------------

class TestGameSetup:
    def test_role_assignment(self):
        game = Game(num_players=6, seed=0)
        roles = [p.role for p in game.players]
        assert roles.count(Role.HITLER) == 1
        assert roles.count(Role.FASCIST) == 1
        assert roles.count(Role.LIBERAL) == 4

    def test_role_assignment_9_players(self):
        game = Game(num_players=9, seed=0)
        roles = [p.role for p in game.players]
        assert roles.count(Role.HITLER) == 1
        assert roles.count(Role.FASCIST) == 3
        assert roles.count(Role.LIBERAL) == 5

    def test_invalid_player_count(self):
        with pytest.raises(ValueError):
            Game(num_players=4)
        with pytest.raises(ValueError):
            Game(num_players=11)

    def test_initial_phase(self):
        game = Game(num_players=6, seed=1)
        assert game.phase == Phase.NOMINATION

    def test_eligible_chancellors_excludes_president(self):
        game = Game(num_players=6, seed=2)
        eligible_ids = [p.player_id for p in game.eligible_chancellors()]
        assert game.president_idx not in eligible_ids


# ---------------------------------------------------------------------------
# Game flow tests
# ---------------------------------------------------------------------------

class TestGameFlow:
    def _full_vote(self, game: Game, vote_value: bool):
        """Cast the same vote for all alive players."""
        for pid in game.alive_ids:
            result = game.cast_vote(pid, vote_value)
            if result is not None:
                return result

    def test_nomination_and_vote_ja(self):
        game = Game(num_players=5, seed=10)
        eligible = game.eligible_chancellors()
        game.nominate_chancellor(eligible[0].player_id)
        assert game.phase == Phase.VOTE
        result = self._full_vote(game, True)
        # Government elected → legislative session
        assert result in (Phase.LEGISLATIVE_PRESIDENT, Phase.GAME_OVER)

    def test_nomination_and_vote_nein(self):
        game = Game(num_players=5, seed=10)
        eligible = game.eligible_chancellors()
        game.nominate_chancellor(eligible[0].player_id)
        result = self._full_vote(game, False)
        assert game.board.election_tracker == 1
        assert result == Phase.NOMINATION

    def test_cannot_nominate_ineligible(self):
        game = Game(num_players=6, seed=5)
        with pytest.raises((ValueError, RuntimeError)):
            game.nominate_chancellor(game.president_idx)

    def test_full_legislative_session(self):
        game = Game(num_players=5, seed=42)
        eligible = game.eligible_chancellors()
        game.nominate_chancellor(eligible[0].player_id)
        self._full_vote(game, True)
        if game.phase == Phase.LEGISLATIVE_PRESIDENT:
            game.president_discard(0)
            if game.phase == Phase.LEGISLATIVE_CHANCELLOR:
                game.chancellor_discard(0)
        # After the session the game either ends or returns to NOMINATION/PRESIDENTIAL_POWER
        assert game.phase in (Phase.NOMINATION, Phase.PRESIDENTIAL_POWER, Phase.GAME_OVER)

    def test_election_tracker_resets_on_ja(self):
        game = Game(num_players=5, seed=99)
        eligible = game.eligible_chancellors()
        game.nominate_chancellor(eligible[0].player_id)
        self._full_vote(game, False)  # nein → tracker 1
        game.nominate_chancellor(game.eligible_chancellors()[0].player_id)
        self._full_vote(game, True)   # ja → reset
        if not game.is_over():
            assert game.board.election_tracker == 0

    def test_three_failed_votes_chaos(self):
        game = Game(num_players=5, seed=100)
        for _ in range(3):
            if game.phase != Phase.NOMINATION:
                break
            eligible = game.eligible_chancellors()
            game.nominate_chancellor(eligible[0].player_id)
            self._full_vote(game, False)
        # After 3 nein votes the election tracker auto-enacts a policy
        total = game.board.liberal_policies + game.board.fascist_policies
        assert total >= 1 or game.is_over()

    def test_term_limits(self):
        game = Game(num_players=6, seed=200)
        eligible = game.eligible_chancellors()
        chan_id = eligible[0].player_id
        game.nominate_chancellor(chan_id)
        self._full_vote(game, True)
        if game.is_over():
            return
        # Run through a quick legislative session
        while game.phase in (Phase.LEGISLATIVE_PRESIDENT, Phase.LEGISLATIVE_CHANCELLOR,
                              Phase.PRESIDENTIAL_POWER):
            if game.phase == Phase.LEGISLATIVE_PRESIDENT:
                game.president_discard(0)
            elif game.phase == Phase.LEGISLATIVE_CHANCELLOR:
                game.chancellor_discard(0)
            elif game.phase == Phase.PRESIDENTIAL_POWER:
                p = game.pending_power
                if p == "policy_peek":
                    game.use_policy_peek()
                elif p == "investigate":
                    t = next(pid for pid in game.alive_ids if pid != game.president_idx)
                    game.use_investigate(t)
                elif p == "special_election":
                    t = next(pid for pid in game.alive_ids if pid != game.president_idx)
                    game.use_special_election(t)
                elif p == "execution":
                    t = next(pid for pid in game.alive_ids if pid != game.president_idx)
                    game.use_execution(t)
                break
        if not game.is_over() and game.phase == Phase.NOMINATION:
            # The previous chancellor should be term-limited
            new_eligible_ids = [p.player_id for p in game.eligible_chancellors()]
            if len(game.alive_players) >= 5:
                assert chan_id not in new_eligible_ids


# ---------------------------------------------------------------------------
# Win-condition tests
# ---------------------------------------------------------------------------

class TestWinConditions:
    def test_liberal_policy_win(self):
        game = Game(num_players=5, seed=0)
        for _ in range(5):
            game.board.enact(Party.LIBERAL)
        assert game.board.liberal_win

    def test_fascist_policy_win(self):
        game = Game(num_players=5, seed=0)
        for _ in range(6):
            game.board.enact(Party.FASCIST)
        assert game.board.fascist_win

    def test_hitler_elected_fascist_win(self):
        game = Game(num_players=5, seed=42)
        # Force 3 fascist policies on the board
        game.board.fascist_policies = 3
        # Find Hitler
        hitler_id = next(p.player_id for p in game.players if p.role == Role.HITLER)
        # Make Hitler eligible chancellor
        game.prev_president_idx = None
        game.prev_chancellor_idx = None
        # Ensure Hitler is not the president
        pres_id = game.president_idx
        if pres_id == hitler_id:
            game._advance_presidency()
        # Nominate Hitler
        if hitler_id in [p.player_id for p in game.eligible_chancellors()]:
            game.nominate_chancellor(hitler_id)
            for pid in game.alive_ids:
                result = game.cast_vote(pid, True)
                if result is not None:
                    break
            assert game.is_over()
            assert game.result.winner == Party.FASCIST
            assert game.result.reason == WinReason.HITLER_ELECTED

    def test_hitler_executed_liberal_win(self):
        game = Game(num_players=5, seed=42)
        # Give the president the execution power manually
        game.pending_power = "execution"
        game.phase = Phase.PRESIDENTIAL_POWER
        hitler_id = next(p.player_id for p in game.players if p.role == Role.HITLER)
        if hitler_id != game.president_idx:
            game.use_execution(hitler_id)
            assert game.is_over()
            assert game.result.winner == Party.LIBERAL
            assert game.result.reason == WinReason.HITLER_SHOT


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------

class TestObservations:
    def test_liberal_does_not_see_other_roles(self):
        game = Game(num_players=6, seed=7)
        lib_id = next(p.player_id for p in game.players if p.role == Role.LIBERAL)
        obs = game.get_observation(lib_id)
        known = obs["known_roles"]
        assert lib_id in known
        assert all(known[k] == Role.LIBERAL.value for k in known)

    def test_fascist_sees_teammates(self):
        game = Game(num_players=6, seed=7)
        fas_id = next(p.player_id for p in game.players if p.role == Role.FASCIST)
        obs = game.get_observation(fas_id)
        known = obs["known_roles"]
        assert Role.HITLER.value in known.values()
        assert Role.FASCIST.value in known.values()
