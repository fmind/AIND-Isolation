# -*- coding: utf-8 -*-
"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
from sample_players import improved_score
from statistics import mean
from itertools import count
from math import sqrt


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def euclidean_distance(a, b):
    """Compute the euclidean distance between two vectors."""
    return sqrt(sum((a - b)**2 for a, b in zip(a, b)))


def unapply_last_move(game, player):
    """Remove the last move of player from the game, return the move and the game."""
    last_board = game.copy()
    last_move = game.__last_player_move__[player]
    last_board.__board_state__[last_move[0]][last_move[1]] = game.BLANK

    return last_board, last_move


def stay_close_to_center(game, player):
    """Compute the distance between the player location and the center of the game."""
    center = (game.width / 2, game.height / 2)
    player_location = game.get_player_location(player)
    distance = euclidean_distance(center, player_location)

    return -distance  # more distance => worst evaluation score


def stay_close_to_opponent(game, player, aggresive=True):
    """Compute the distance between the player and the opponent location.

    If aggresive is True, try to select the legal moves of the opponent."""
    opponent = game.get_opponent(player)

    if aggresive:
        last_board, last_move = unapply_last_move(game, player)
        opponent_moves = last_board.get_legal_moves(opponent)

        if last_move in opponent_moves:
            return 0

    player_location = game.get_player_location(player)
    opponent_location = game.get_player_location(opponent)
    distance = euclidean_distance(player_location, opponent_location)

    return -distance  # more distance => worst evaluation score


def stay_close_to_blank_spaces(game, player):
    """Compute the average distance between the player location and blank spaces."""
    blanks = game.get_blank_spaces()
    player_location = game.get_player_location(player)

    return mean(euclidean_distance(b, player_location) for b in blanks)


def compose_scores(*functions):
    """Compute a list of score functions into a single function by taking their mean."""
    def composition(game, player):
        return mean(f(game, player) for f in functions)

    return composition


improved_opponent = compose_scores(improved_score, stay_close_to_opponent)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
    An instance of `isolation.Board` encoding the current state of the
    game (e.g., player locations and blocked cells).

    player : object
    A player instance in the current game (i.e., an object corresponding to
    one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
    The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")
    elif game.is_winner(player):
        return float("inf")
    else:
        # return stay_close_to_center(game, player)
        # return stay_close_to_opponent(game, player)
        # return stay_close_to_opponent(game, player, False)
        # return stay_close_to_blank_spaces(game, player)
        return improved_opponent(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
    A strictly positive integer (i.e., 1, 2, 3,...) for the number of
    layers in the game tree to explore for fixed-depth search. (i.e., a
    depth of one (1) would only explore the immediate sucessors of the
    current state.)

    score_fn : callable (optional)
    A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
    Flag indicating whether to perform fixed-depth search (False) or
    iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
    The name of the search method to use in get_move().

    timeout : float (optional)
    Time remaining (in milliseconds) when search is aborted. Should be a
    positive value large enough to allow the function to return before the
    timer expires.
    """

    # perform no move
    NO_MOVE = (-1, -1)

    def __init__(self,
                 search_depth=3,
                 score_fn=custom_score,
                 iterative=True,
                 method='minimax',
                 timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
        forfeit the game due to timeout. You must return _before_ the
        timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
        A list containing legal moves. Moves are encoded as tuples of pairs
        of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
        A function that returns the number of milliseconds left in the
        current turn. Returning with any less than 0 ms remaining forfeits
        the game.

        Returns
        -------
        (int, int)
        Board coordinates corresponding to a legal move; may return
        (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # no legal move available
        # => return NO_MOVE (nop)
        if len(legal_moves) == 0:
            return self.NO_MOVE

        # initialization
        score = None
        move = legal_moves[0]
        method = getattr(self, self.method)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                for depth in count(1):  # to infinity
                    score, move = method(game, depth)
            else:
                score, move = method(game, self.search_depth)
        except Timeout:
            pass

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
        An instance of the Isolation game `Board` class representing the
        current game state

        depth : int
        Depth is an integer representing the maximum number of plies to
        search in the game tree before aborting

        maximizing_player : bool
        Flag indicating whether the current search depth corresponds to a
        maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
        The score for the current search branch

        tuple(int, int)
        The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
        (1) You MUST use the `self.score()` method for board evaluation
        to pass the project unit tests; you cannot call any other
        evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()

        # cannot move the player further on the board
        # => return the evaluation of the game state
        if depth == 0 or len(moves) == 0:
            score = self.score(game, self)
            move = self.NO_MOVE
            return score, move

        # initialization
        score = None
        move = None

        # recur/search
        for m in moves:
            # we don't care about the last move performed at the terminal state
            s, _ = self.minimax(
                game.forecast_move(m), depth - 1, not maximizing_player)

            # decide to change the current score and move
            if self.minimax_change_decision(maximizing_player, s, score):
                score = s
                move = m

        return score, move

    def alphabeta(self,
                  game,
                  depth,
                  alpha=float("-inf"),
                  beta=float("inf"),
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
        An instance of the Isolation game `Board` class representing the
        current game state

        depth : int
        Depth is an integer representing the maximum number of plies to
        search in the game tree before aborting

        alpha : float
        Alpha limits the lower bound of search on minimizing layers

        beta : float
        Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
        Flag indicating whether the current search depth corresponds to a
        maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
        The score for the current search branch

        tuple(int, int)
        The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
        (1) You MUST use the `self.score()` method for board evaluation
        to pass the project unit tests; you cannot call any other
        evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()

        # cannot move the player further on the board
        # => return the evaluation of the game state
        if depth == 0 or len(moves) == 0:
            score = self.score(game, self)
            move = self.NO_MOVE
            return score, move

        # initialization
        score = None
        move = None

        # recur/search
        for m in moves:
            # we don't care about the last move performed at the terminal state
            s, _ = self.alphabeta(
                game.forecast_move(m), depth - 1, alpha, beta,
                not maximizing_player)

            # decide to change the current score and move (same as minimax)
            if self.minimax_change_decision(maximizing_player, s, score):
                score = s
                move = m

                # update alpha and beta values
                alpha, beta = self.alphabeta_update(maximizing_player, alpha,
                                                    beta, s)

                # decide to cut the search
                if self.alphabeta_cut_decision(maximizing_player, alpha, beta):
                    break

        return score, move

    @staticmethod
    def minimax_change_decision(maxplayer, proposed, current=None):
        """Return True if minimax should select the proposed move."""
        if current is None:
            return True
        if maxplayer and proposed > current:
            return True
        elif not maxplayer and proposed < current:
            return True
        else:
            return False

    @staticmethod
    def alphabeta_cut_decision(maxplayer, alpha, beta):
        """Return True if alphabeta should stop the current search."""
        # NOTE: the comparison operators are correct (see wikipedia)
        if maxplayer and alpha >= beta:  # beta cut-off
            return True
        elif not maxplayer and alpha >= beta:  # alpha cut-off
            return True
        else:
            return False

    @staticmethod
    def alphabeta_update(maxplayer, alpha, beta, proposed):
        """Encode the update procedure of the alphabeta algorithm."""
        if maxplayer and proposed > alpha:
            return proposed, beta
        elif not maxplayer and proposed < beta:
            return alpha, proposed
        else:
            return alpha, beta
