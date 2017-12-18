"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from isolation import Board
from copy import deepcopy


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    '''

    :param game: object, game, of Isolation.Board
    :param player: object, player, for Isolation.Board._player_1
    :return: int, the score for the board
    '''
    """
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
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score = len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player)))
    return float(score)

def custom_score_2(game, player):
    '''

    :param game: object, game, of Isolation.Board
    :param player: object, player, for Isolation.Board._player_1
    :return: int, the score for the board
    '''
    """
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
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score = len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player)))
    return float(score)

def custom_score_3(game, player):
    '''

    :param game: object, game, of Isolation.Board
    :param player: object, player, for Isolation.Board._player_1
    :return: int, the score for the board
    '''
    """
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
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score = len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player)))
    return float(score)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        '''

        :param search_depth: int, depth limit for search agent.
        :param score_fn: function, name of the default scoring function.
        :param timeout:  float, 15 millisecs for default, too less and MiniMax Search
                            will time-out.
        '''
        super(MinimaxPlayer, self).__init__(search_depth=3, score_fn=custom_score, timeout=10.)
        self.absmin = float("-inf")
        self.absmax = float("inf")

    def activeplayer(self, game):
        '''

        :param game: object, game, of Isolation.Board
        :return: BOOL
        '''
        return game.active_player == self

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

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
        bestmove = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return bestmove

    def minmax_search(self, game, depth):
        '''
        search and helper for minmax
        :param game: object, game, of Isolation.Board
        :param depth: int, search limit
        :return: (int, int) , int : move, and score
         '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #Escape
        bestmove = (-1,-1)
        moves = game.get_legal_moves()
        if not moves:
            return bestmove, self.score(game, self)
        #Parity and Vars
        optimizer, value = None, None
        if self.activeplayer(game):
            optimizer, value = max, self.absmin
        else:
            optimizer, value = min, self.absmax
        #depth-limit checking
        if depth == 0:
            return (game.get_player_location(self), self.score(game, self))
        #recursion and search logic
        for move_ in game.get_legal_moves():
            nextgame = game.forecast_move(move_)
            score = self.minmax_search(nextgame, depth-1)[1]
            if optimizer(value, score) == score:
                bestmove = move_
                value = score

        return (bestmove, value)

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

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
        #Using Helper Function
        return self.minmax_search(game, depth)[0]

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    '''Alpha-Beta Using Sorted Nodes
    '''

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=15.):
        '''

        :param search_depth: int, depth limit for search agent.
        :param score_fn: function, name of the default scoring function.
        :param timeout:  float, 15 millisecs for default, too less and AB will time-out.
        '''
        super(AlphaBetaPlayer, self).__init__(search_depth=3, score_fn=custom_score, timeout=10.)
        self.absmin = float("-inf")
        self.absmax = float("inf")

    def activeplayer(self, game):
        '''

        :param game: object, game, of Isolation.Board
        :return: BOOL
        '''
        #PARITY
        return game.active_player == self

    def isTerminal(self, game):
        '''

        :param game: object, game, of Isolation.Board
        :return: BOOL, yes or no if the game is in end state
        '''
        if len(game.get_legal_moves(self)) <= 1:
            return True
        else:
            return False

    def GetSortedNodes(self, game, player):
        sortedNodes = []
        legals = game.get_legal_moves(player)
        for moves_ in legals:
                    #COPY
                    boardTemp = game.forecast_move(moves_)
                    #EVAL
                    sortedNodes.append((boardTemp, moves_,self.score(game, self)))
        if self.activeplayer(game):
            #DESCENDING FOR MAXIMIZING PLAYER
            sortedNodes = sorted(sortedNodes, key=lambda node: node[2], reverse=True)
        else:
            #ASCENDING FOR MINIMIZNG PLAYER
            sortedNodes = sorted(sortedNodes, key=lambda node: node[2], reverse=False)
        sortedNodes = [node[0] for node in sortedNodes]
        return sortedNodes

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
        #DEFAULT
        bestmove = (-1, -1)
        #ESCAPE
        nmoves = game.get_legal_moves(self)
        if len(nmoves) >= 1:
            bestmove = nmoves[0]
        else:
            return bestmove

        #MOVE SEARCH
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for x in range(3, 100):
                bestmove = self.alphabeta(game, x)
                #print("Search depth:",x, "Moves Played:",game.move_count, "Best:", best_move, EvalBoard(game, self), "\n")

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return bestmove

    def alphabeta_search(self, game, depth, alpha=float("-inf"), beta=float("inf")):
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
        (int, int), float

            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

            The score for the search from float:'-inf' to float:'+inf'
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #DEFAULTS
        bestmove = (-1, -1)
        #PARITY
        playercheck = self.activeplayer(game)
        if playercheck:
            optimizer, value, player = max, self.absmin, game.active_player
        else:
            optimizer, value, player = min, self.absmax, game.active_player
        #ESCAPE
        if self.isTerminal(game):
            if game.is_winner(self):
                return (-1, -1),self.absmax
            elif game.is_loser(self):
                return (-1,-1), self.absmin
        #DEPTH-LIMIT
        if depth == 0:
            return (game.get_player_location(player), self.score(game, player))

        #AB LOGIC and RECURSION
        SortedNodes = self.GetSortedNodes(game, player)
        #sortednodes: boardTemp, moves_, self.score(game, self)
        for games in SortedNodes:
            #debug
            #print("SortedNode Value: ",EvalBoard(games, player), "Player:", type(player), "\n")
            score = self.alphabeta_search(games, depth -1, alpha, beta)[1]
            if score == optimizer(value, score ):
                #debug
                #print("Playercheck",self.activeplayer(games))
                value = score
                bestmove = games.get_player_location(player)
            #AB BOUNDS
            if playercheck:
                alpha = optimizer(alpha, value)
                if beta < alpha:
                    break
            else:
                beta = optimizer(beta, value)
                if beta < alpha:
                    break

        return bestmove, value

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
        #CALL MAIN HELPER FUNCTION
        return self.alphabeta_search(game, depth, alpha, beta)[0]




