#!/usr/bin/env python3

import string  # string.ascii_lowercase; map it later using a dictionary?
import position
import chessfigure


class Chessboard:
    def __init__(self) -> None:
        self.initialize_board()
        self.existing_figures = {}
        self.create_figures()
        self.killed_figures = {}

    def print_board(self):
        for i in self.board:
            print(' | ', end='')
            for j in i:
                print(j, end=' | ')
            print()

        for i in self.board:
            print(i)

    def initialize_board(self):
        if not position.Position.taken_pisitions:
            self.board = [[column + str(row) for column in string.ascii_lowercase[:8]]
                          for row in range(1, 9)]  # {a=1, b =2} map it this way
        else:
            # untested
            self.board = [[column + str(row) if position.Position.check_free(row, column) else column + str(
                row) for column in string.ascii_lowercase[:8]] for row in range(1, 9)]  # + name from taken positions

    def create_figures(self):
        for i in range(1, 9):
            self.existing_figures[f'({2}, {i})'] = chessfigure.Pawn(
                'white', position.Position(2, i))
        for i in range(1, 9):
            self.existing_figures[f'({7}, {i})'] = chessfigure.Pawn(
                'black', position.Position(7, i))
        for i in range(1, 9, 7):
            self.existing_figures[f'({1}, {i})'] = chessfigure.Rook(
                'white', position.Position(1, i))
        for i in range(1, 9, 7):
            self.existing_figures[f'({8}, {i})'] = chessfigure.Rook(
                'black', position.Position(8, i))
        for i in range(2, 8, 6):
            self.existing_figures[f'({1}, {i})'] = chessfigure.Knight(
                'white', position.Position(1, i))
        for i in range(2, 8, 6):
            self.existing_figures[f'({8}, {i})'] = chessfigure.Knight(
                'black', position.Position(8, i))
        for i in range(3, 7, 4):
            self.existing_figures[f'({1}, {i})'] = chessfigure.Bishop(
                'white', position.Position(1, i))
        for i in range(3, 7, 4):
            self.existing_figures['f({8}, {i})'] = chessfigure.Bishop(
                'black', position.Position(8, i))
        self.existing_figures[f'({1}, {4})'] = chessfigure.Queen(
            'white', position.Position(1, 4))
        self.existing_figures[f'({8}, {4})'] = chessfigure.Queen(
            'black', position.Position(8, 4))
        self.existing_figures[f'({1}, {5})'] = chessfigure.King(
            'white', position.Position(1, 5))
        self.existing_figures[f'({8}, {5})'] = chessfigure.King(
            'black', position.Position(8, 5))


if __name__ == '__main__':
    pass
