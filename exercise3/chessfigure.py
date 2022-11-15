#!/usr/bin/env python3

class ChessFigure:
    def __init__(self, figure_cls, color, position) -> None:
        self.figure_cls = figure_cls # see if needed as self further down the line
        self.color = color
        self.figure_type = figure_cls.FIGURE_TYPE
        self.number = figure_cls.number
        figure_cls.number += 1
        self.name = self.figure_type + ' ' + str(self.number)
        self.position = position
        

    def move(self, position):
        pass

    def beat(self, position):
        pass


class Rook(ChessFigure):
    FIGURE_TYPE = 'rook'

    number = 1
    def __init__(self, color, position) -> None:
        ChessFigure.__init__(self, Rook, color, position)


class Knight(ChessFigure):
    FIGURE_TYPE = 'knight'

    number = 1
    def __init__(self, color, position) -> None:
        ChessFigure.__init__(self, Knight, color, position)


class Bishop(ChessFigure):
    FIGURE_TYPE = 'bishop'

    number = 1
    def __init__(self, color, position) -> None:
        ChessFigure.__init__(self, Bishop, color, position)


class King(ChessFigure):
    FIGURE_TYPE = 'king'

    number = 1
    def __init__(self, color, position) -> None:
        ChessFigure.__init__(self, King, color, position)


class Queen(ChessFigure):
    FIGURE_TYPE = 'queen'

    number = 1
    def __init__(self, color, position) -> None:
        ChessFigure.__init__(self, Queen, color, position)


class Pawn(ChessFigure):
    FIGURE_TYPE = 'pawn'

    number = 1 # make private?
    def __init__(self, color, position) -> None:
        ChessFigure.__init__(self, Pawn, color, position)


if __name__ == '__main__':
    for i in range(5):
        p1 = Pawn('black', 1)
        r1 = Rook('white', 1)
        print(p1.name)
        print(r1.name)
    
