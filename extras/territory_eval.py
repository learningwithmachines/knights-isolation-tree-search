activ# wrapping into a class.

from isolation import Board
from plotter import plotgrid

class movestore():
    '''
    Reverse Moves :
    '''

    def __init__(self, moveloc, level=6):
        self.depthcheck = level
        self.vectorlist = self.seeker(moveloc)

    def dist(self, x, y):
        '''
        returns euclidean distance (int)

        :param x: (int, int) move# 0
        :param y: (int, int) move# 1
        :return: int
        '''
        "return euclidean distance"
        ans = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
        return ans

    def movemark(self, moveslist):
        '''
        mark all moves from moveslist by adding vectors,
        result is a set of algebraic offsets that
        can be added to playerloc to get all possible moves
        :param moveslist: [(int, int),...]
        :return: {DISTANCE: [(int, int),..]}
        '''
        newmoves = moveslist
        results = {}
        vectors = [(-1, 2), (-1, -2), (1, 2), (1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]

        for trials in newmoves:
            for rays in vectors:
                # debug
                # print(type(rays), rays, type(trials), trials, moveslist)
                DIST = self.dist(trials, rays)
                MOV = (rays[0] + trials[0], rays[1] + trials[1])
                temp = {MOV, }
                if DIST <= 98 and DIST > 0:
                    try:
                        temp = temp.union(results[DIST])
                        results[DIST] = temp
                    except KeyError:
                        results[DIST] = temp
        return results

    def flattendict(self, dicto):
        '''
        returns a flattened list
        :param dicto: {:}
        :return: []
        '''

        listres = []
        for keys in dicto.keys():
            for items in dicto[keys]:
                listres.append(items)
        return listres

    def seeker(self, moveloc):
        '''

        :param moveloc: [], list of player location(s)
        :return: {:}, dictionary of alebgraic offsets and depths
        '''
        "take current location as 'moveloc' and try adding all move vectors"
        "stores results into a dictionary of openmoves at corresp. depthlevels"
        tempargs = moveloc
        limit = max = self.depthcheck
        vectordict = {}
        while limit > 0:
            dict = self.movemark(tempargs)
            vectordict[max - limit + 1] = tempargs = tuple(self.flattendict(dict))
            limit -= 1
        return vectordict

    def display(self):
        "display"
        for keys in self.vectorlist:
            print(keys, str(len(self.vectorlist[keys])) + str(" positions"), end='\n')

    def repr_val(self, depth):
        #return entry at correct depth
        return self.vectorlist[depth]

    def regrid(self, x, y):
        #algebra
        return (x[0] + y[0], x[1] + y[1])

    def eval(self, target, checklist, depth):
        openset = set()
        for items in self.repr_val(depth):
            # get pos to add items in vectorlist at chosen depth
            temp = self.regrid(target, items)
            # print(temp)
            if temp in checklist:  # if result is in legals, move is valid
                openset.add(temp)
                # print(openset)

        #return the fraction of the board that is accessible to the player
        return openset


if __name__ == "__main__":

    #testing the class

    vector00 = [(0, 0)]
    MOVX = movestore(vector00)
    gametry = Board('0', '1')
    locinit = gametry.get_blank_spaces()
    #dummy moves
    gametry.apply_move((3, 3))
    gametry.apply_move((4, 4))
    playerloc = gametry.get_player_location('0')
    legals = gametry.get_blank_spaces()
    maxdepth = 6
    #printing figures
    for xstep in range(1, maxdepth + 1):
        #Print Accessibility
        listthing = MOVX.eval(playerloc, legals, xstep)
        pplot = plotgrid(playerloc, locinit, legals, listthing, xstep)
        pplot.start(maxdepth, xstep )