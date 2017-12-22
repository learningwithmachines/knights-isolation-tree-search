import math
import matplotlib.pylab as plt
import numpy as np


class plotgrid:

    def __init__(self, playerloc, fulls, blanks, possibles, stepping):
        self.absmin = -2
        self.absmax = 2
        self.neutral = 0
        self.playerval = -1.9
        self.loc = playerloc
        self.fulls = fulls
        self.blanks = blanks
        self.possibles = possibles
        self.enumdict = {cells: 0 for cells in fulls}
        self.stepping = stepping

    def enum(self):
        full, blanks, possibles = self.fulls, self.blanks, self.possibles
        enumdict = {}
        for keys in self.enumdict.keys():
            if keys == self.loc:
                enumdict[keys] = self.playerval
            elif keys in list(blanks and possibles):
                enumdict[keys] = self.absmax
            elif keys in list(set(full) - set(blanks)):
                enumdict[keys] = self.absmin
            elif keys in list(set(blanks) - set(possibles)):
                enumdict[keys] = self.neutral

        return enumdict

    def makematrix(self, enumdict):
        maxr = maxc = 7
        made = [[enumdict[(x, y)] for y in range(maxc)] for x in range(maxr)]
        return made

    def start(self):
        self.enumdict = self.enum()
        matrix = self.makematrix(self.enumdict)
        self.plotter(matrix)

    def plotter(self, matrix_0):

        matrix = np.matrix(matrix_0)
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1, 1, 1)
        label = [x for x in range(8)]
        ax.set_aspect('equal')
        ax.set_xticklabels(label)
        ax.set_facecolor('white')
        fig.suptitle('Territorial Access Checking: ', fontsize=14, fontweight='bold')
        ax.set_title('Neg = Blocked, 0 = Unreachable, 2 = Inreach, @MaxStep =' + str(self.stepping))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.RdYlGn)
        plt.colorbar()
        plt.show()