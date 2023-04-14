#!/usr/bin/env python3

import position


class ChessFigure:
    def __init__(self, figure_cls, color, position) -> None:
        self.figure_cls = figure_cls  # see if needed as self further down the line
        self.color = color
        self.figure_type = figure_cls.FIGURE_TYPE
        self.number = figure_cls.number
        figure_cls.number += 1
        self.name = self.color + ' ' + \
            self.figure_type + ' ' + str(self.number)
        self.position = position

    def move(self, new_row, new_column, figure_type):
        self.position = position.Position(new_row, new_column)
        print('Moved ' + figure_type + '!')

    def beat(self, new_row, new_column, figure_type):
        self.position = position.Position(new_row, new_column)
        print('Beaten by ' + figure_type + '!')


class Rook(ChessFigure):
    FIGURE_TYPE = 'rook'

    number = 1

    def __init__(self, color, position) -> None:
        if Rook.number > 2:
            Rook.number = 1
        ChessFigure.__init__(self, Rook, color, position)

    def move(self, new_row, new_column):
        ChessFigure.move(self, new_row, new_column, Rook.FIGURE_TYPE)

    def beat(self, new_row, new_column):
        ChessFigure.beat(self, new_row, new_column, Rook.FIGURE_TYPE)


class Knight(ChessFigure):
    FIGURE_TYPE = 'knight'

    number = 1

    def __init__(self, color, position) -> None:
        if Knight.number > 2:
            Knight.number = 1
        ChessFigure.__init__(self, Knight, color, position)

    def move(self, new_row, new_column):
        ChessFigure.move(self, new_row, new_column, Knight.FIGURE_TYPE)

    def beat(self, new_row, new_column):
        ChessFigure.beat(self, new_row, new_column, Knight.FIGURE_TYPE)


class Bishop(ChessFigure):
    FIGURE_TYPE = 'bishop'

    number = 1

    def __init__(self, color, position) -> None:
        if Bishop.number > 2:
            Bishop.number = 1
        ChessFigure.__init__(self, Bishop, color, position)

    def move(self, new_row, new_column):
        ChessFigure.move(self, new_row, new_column, Bishop.FIGURE_TYPE)

    def beat(self, new_row, new_column):
        ChessFigure.beat(self, new_row, new_column, Bishop.FIGURE_TYPE)


class King(ChessFigure):
    FIGURE_TYPE = 'king'

    number = 1

    def __init__(self, color, position) -> None:
        if King.number > 1:
            King.number = 1
        ChessFigure.__init__(self, King, color, position)

    def castle(self, new_row, new_column):
        self.position = position.Position(new_row, new_column)
        print('And castled!')

    def move(self, new_row, new_column):
        ChessFigure.move(self, new_row, new_column, King.FIGURE_TYPE)

    def beat(self, new_row, new_column):
        ChessFigure.beat(self, new_row, new_column, King.FIGURE_TYPE)


class Queen(ChessFigure):
    FIGURE_TYPE = 'queen'

    number = 1

    def __init__(self, color, position) -> None:
        if Queen.number > 1:
            Queen.number = 1
        ChessFigure.__init__(self, Queen, color, position)

    def move(self, new_row, new_column):
        ChessFigure.move(self, new_row, new_column, Queen.FIGURE_TYPE)

    def beat(self, new_row, new_column):
        ChessFigure.beat(self, new_row, new_column, Queen.FIGURE_TYPE)


class Pawn(ChessFigure):
    FIGURE_TYPE = 'pawn'

    number = 1  # make private?

    def __init__(self, color, position) -> None:
        if Pawn.number > 8:
            Pawn.number = 1
        ChessFigure.__init__(self, Pawn, color, position)

    def move(self, new_row, new_column):
        ChessFigure.move(self, new_row, new_column, Pawn.FIGURE_TYPE)

    def beat(self, new_row, new_column):
        ChessFigure.beat(self, new_row, new_column, Pawn.FIGURE_TYPE)


if __name__ == '__main__':
    for i in range(5):
        p1 = Pawn('black', 1)
        r1 = Rook('white', 1)
        print(p1.name)
        print(r1.name)
