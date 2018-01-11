"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    '''
    Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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

    '''
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    ownmoves = game.get_legal_moves(player)
    oppmoves = game.get_legal_moves(game.get_opponent(player))

    #memoizing move quality
    #prioritizes picking good neighborhoods over the openmoves difference
    movesvalues = {(0, 0): 2, (1, 0): 3, (2, 0): 4, (3, 0): 4, (4, 0): 4,
                   (5, 0): 3, (6, 0): 2, (0, 1): 3, (1, 1): 4, (2, 1): 6,
                   (3, 1): 6, (4, 1): 6, (5, 1): 4, (6, 1): 3, (0, 2): 4,
                   (1, 2): 6, (2, 2): 8, (3, 2): 8, (4, 2): 8, (5, 2): 6,
                   (6, 2): 4, (0, 3): 4, (1, 3): 6, (2, 3): 8, (3, 3): 8,
                   (4, 3): 8, (5, 3): 6, (6, 3): 4, (0, 4): 4, (1, 4): 6,
                   (2, 4): 8, (3, 4): 8, (4, 4): 8, (5, 4): 6, (6, 4): 4,
                   (0, 5): 3, (1, 5): 4, (2, 5): 6, (3, 5): 6, (4, 5): 6,
                   (5, 5): 4, (6, 5): 3, (0, 6): 2, (1, 6): 3, (2, 6): 4,
                   (3, 6): 4, (4, 6): 4, (5, 6): 3, (6, 6): 2
                   }
    #normalised move-quality
    ownmovescore = float(sum([movesvalues[move] for move in ownmoves]) / (len(ownmoves) + 1))
    oppmovescore = float(sum([movesvalues[move] for move in oppmoves]) / (len(oppmoves) + 1))

    score = len(ownmoves)*ownmovescore - len(oppmoves)*oppmovescore

    return float(score)


class CustomPlayer:
    """Game-playing agent that chooses a move using iterative deepening minimax
        search with alpha-beta pruning. You must finish and test this player to
        make sure it returns a good move before the search time limit expires.
        """

    '''Alpha-Beta Using Sorted Nodes
    '''

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=1.):
        '''

        :param search_depth: int, depth limit for search agent.
        :param score_fn: function, name of the default scoring function.
        :param timeout:  float, 15 millisecs for default, too less and AB will time-out.
        '''
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.absmin = float("-inf")
        self.absmax = float("inf")

    def GetSortedNodes(self, game, isMAX):
        '''

        :param game: object; instance of Board class.
        :param isMAX: BOOL; flag for switching between maximizing
                        and minimizing players
        :return: [(int, int) * len(game.get_legal_moves())]

        returns a list of moves, in order of scores.
        sorted by scores from self.score(game, player)
        Ascending/Descending for MIN/MAX resp.

        Call this instead of game.get_legal_moves()
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        sortedNodes = []
        legals = game.get_legal_moves()
        if legals:
            for move_ in legals:
                nextgame = game.forecast_move(move_)
                appendscore = [move_, self.score(nextgame, self)]
                sortedNodes.append(appendscore)
        else:
            return []

        if isMAX:
            sortedNodes = sorted(sortedNodes, key=lambda node: node[1], reverse=True)
        else:
            sortedNodes = sorted(sortedNodes, key=lambda node: node[1], reverse=False)

        sortedNodes = [node[0] for node in sortedNodes]
        return sortedNodes

    def is_activeplayer(self, game):
        '''

        :param game: object, game, of Isolation.Board
        :return: BOOL
        '''
        # PARITY
        return game.active_player == self

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        # DEFAULT
        bestmove = (-1, -1)
        nexmoves = game.get_legal_moves()
        if len(nexmoves) >= 1:
            bestmove = nexmoves[0]
        else:
            return bestmove
        # MOVE SEARCH
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while self.time_left() > self.TIMER_THRESHOLD:
                nextmove = self.alphabeta(game, depth)
                depth += 1
                if not nextmove:
                    return bestmove
                else:
                    bestmove = nextmove

        except SearchTimeout:
            return bestmove
        # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return bestmove

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # defaults
        bestscore, bestmove = self.absmin, None
        # get moves
        moves = game.get_legal_moves()
        # start search for all legal moves
        for move_ in moves:
            nextgame = game.forecast_move(move_)
            # call search on children
            score = self.ab_min(nextgame, depth - 1, alpha, beta)
            # update on discovery
            if score > bestscore:
                bestmove, bestscore = move_, score
            # update bounds and check
            if bestscore >= beta:
                break

            alpha = max(alpha, bestscore)
        return bestmove

    def ab_max(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # stop search
        if depth == 0:
            return self.score(game, self)
        # set bounds on search
        score = self.absmin
        # start search for all moves in legal
        for move in game.get_legal_moves():
            # recursion
            score = max(score, self.ab_min(game.forecast_move(move), depth - 1, alpha, beta))

            # check and update bounds
            if score >= beta:
                return score

            alpha = max(score, alpha)

        return score

    def ab_min(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # stop search
        if depth == 0:
            return self.score(game, self)
        # set bounds for search
        score = self.absmax
        # recursion
        for move in game.get_legal_moves():
            score = min(score, self.ab_max(game.forecast_move(move), depth - 1, alpha, beta))
            # check and update bounds
            # out of bound
            if score <= alpha:
                return score

            beta = min(beta, score)
        return score
