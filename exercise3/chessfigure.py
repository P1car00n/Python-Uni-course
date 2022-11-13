#!/usr/bin/env python3

class ChessFigure:
    def __init__(self, color, name, position) -> None:
        self.color = color
        self.name = name
        self.position = position

    def move(self, position):
        pass

    def beat(self, position):
        pass

class Rook(ChessFigure):
    pass

class Knight(ChessFigure):
    pass

class Bishop(ChessFigure):
    pass

class King(ChessFigure):
    pass

class Queen(ChessFigure):
    pass

class Pawn(ChessFigure):
    pass


if __name__ == '__main__':
    pass