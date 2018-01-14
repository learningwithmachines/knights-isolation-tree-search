"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random, math
from isolation import Board
from copy import deepcopy


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def distance(game, move1, move2=None):
    '''
    helper function to return the distance between cells based
    on euclidean distance.
    :param game:  object, of Isolation.Board
    :param move1: (int, int) : move for calculation of distance
    :param move2: (int, int) :optional, if True, replaces centre
                                for distance calculations
    :return:
    '''
    cx, cy = math.ceil((game.width - 1) / 2), math.ceil((game.height - 1) / 2)

    if move2:
        return (move1[0] - move2[0]) ** 2 + (move1[1] - move2[0]) ** 2
    else:
        return (move1[0] - cx) ** 2 + (move1[1] - cy) ** 2

def custom_score(game, player):
    '''

    :param game: object, instance of Isolation.Board
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

    #score 1 : open moves difference #control for AB-improved, not an implemented heurisitic
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    score = len(own_moves) - len(opp_moves)
    return float(score)

def custom_score_2(game, player):
    '''

    :param game: object, instance of Isolation.Board
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
    #score 2: update score 1 score with player centrality

    score = custom_score(game, player)
    score = score * distance(game, game.get_player_location(player))
    return float(score)

def custom_score_3(game, player):
    '''

    :param game: object, instance of Isolation.Board
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
    ownmovescore = float(sum([movesvalues[move] for move in ownmoves]) / 8)
    oppmovescore = float(sum([movesvalues[move] for move in oppmoves]) / 8)

    score = len(ownmoves)*ownmovescore - len(oppmoves)*oppmovescore
    return float(score)

def custom_score_4(game, player):
    '''

    :param game: object, instance of Isolation.Board
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

    blanks = len(game.get_blank_spaces())
    ownmoves = game.get_legal_moves(player)
    ownloc = game.get_player_location(player)
    opploc = game.get_player_location(game.get_opponent(player))

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
    ownmovescore = float(sum([movesvalues[move] for move in ownmoves]) / 8) #normalised move-quality
    score_chase = min(18, distance(game, ownloc, opploc))/18 #penalty, Max(~1) while away, min when near
    endgame_flip = (float(blanks/49)<0.35) * score_chase #True at endgame,
    score = ownmovescore - endgame_flip #stay close
    return float(score)

def custom_score_5(game, player):
    '''

    :param game: object, instance of Isolation.Board
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

    blanks = len(game.get_blank_spaces())
    ownmoves = game.get_legal_moves(player)
    ownloc = game.get_player_location(player)
    opploc = game.get_player_location(game.get_opponent(player))
    openmoves = len(ownmoves)

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

    ownmovescore = float(sum([movesvalues[move] for move in ownmoves]) / 8) #normalised move-quality
    score_chase = min(18, distance(game, ownloc, opploc))/18 #penalty, Max(~1) while away, min when near
    endgame_flip = (float(blanks/49)<0.35) * score_chase #True at endgame,
    if openmoves < 2:
        score = ownmovescore + endgame_flip #away
    else:
        score = ownmovescore - endgame_flip

    return float(score)

def custom_score_6(game, player):
    '''

    :param game: object, instance of Isolation.Board
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

    ownmoves = game.get_legal_moves(player)
    ownloc = game.get_player_location(player)
    opploc = game.get_player_location(game.get_opponent(player))
    openmoves = len(ownmoves)

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

    ownmovescore = float(sum([movesvalues[move] for move in ownmoves]) / 8)
    run = distance(game, ownloc, opploc)

    if openmoves < 2:
        score = ownmovescore + run #away - relaxed
    else:
        score = ownmovescore
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
        super(MinimaxPlayer, self).__init__(search_depth, score_fn, timeout)
        self.absmin = float("-inf")
        self.absmax = float("inf")

    def activeplayer(self, game):
        """
        :param game: object, game, of Isolation.Board
        :return: BOOL
        """
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
        #check depth before moves.
        if depth == 0:
            return (game.get_player_location(self), self.score(game, self))
        #Defaults
        optimizer, value, bestmove = None, self.absmin, (-1,-1)
        #Parity and Vars
        #optimizer, value = None, None
        if self.activeplayer(game):
            optimizer, value = max, self.absmin
        else:
            optimizer, value = min, self.absmax
        #recursion and search logic
        for nextmove_ in game.get_legal_moves():
            nextgame = game.forecast_move(nextmove_)
            score = self.minmax_search(nextgame, depth-1)[1]
            if optimizer(value, score) == score:
                bestmove = nextmove_
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
        super(AlphaBetaPlayer, self).__init__(search_depth, score_fn, timeout)
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
        #PARITY
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
        #DEFAULT
        bestmove = (-1, -1)
        nexmoves = game.get_legal_moves()
        if len(nexmoves) >= 1:
            bestmove = nexmoves[0]
        else:
            return bestmove
        #MOVE SEARCH
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 1
            while self.time_left() > self.TIMER_THRESHOLD:
                nextmove =  self.alphabeta(game, depth)
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
        #defaults
        bestscore, bestmove = self.absmin, None
        #get moves
        moves = game.get_legal_moves()
        #start search for all legal moves
        for move_ in moves:
            nextgame = game.forecast_move(move_)
            #call search on children
            score = self.ab_min(nextgame, depth-1, alpha, beta)
            #update on discovery
            if score > bestscore:
                bestmove, bestscore = move_, score
            #update bounds and check
            if bestscore >= beta:
                break

            alpha = max(alpha, bestscore)
        return bestmove

    def ab_max(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #stop search
        if depth == 0:
            return self.score(game, self)
        #set bounds on search
        score = self.absmin
        #start search for all moves in legal
        for move in game.get_legal_moves():
            #recursion
            score = max(score, self.ab_min(game.forecast_move(move), depth - 1, alpha, beta))

            #check and update bounds
            if score >= beta:
                return score

            alpha =  max(score, alpha)

        return score
    def ab_min(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #stop search
        if depth == 0:
            return  self.score(game, self)
        #set bounds for search
        score = self.absmax
        #recursion
        for move in game.get_legal_moves():
            score = min(score, self.ab_max(game.forecast_move(move), depth - 1, alpha, beta))
            #check and update bounds
            #out of bound
            if score <= alpha:
                return score

            beta = min(beta, score)
        return score