#!/usr/bin/env python3

import string  # string.ascii_lowercase; map it later using a dictionary?
import position
import chessfigure


class Chessboard:
    def __init__(self) -> None:
        self.existing_figures = {}
        self.create_figures()
        self.initialize_board()
        #self.killed_figures = {}

    def print_board(self):
        for i in self.board:
            print(' | ', end='')
            for j in i:
                print(j, end=' | ')
            print()

        # for i in self.board:
        #     print(i)

    def initialize_board(self):
        # if not position.Position.taken_pisitions:
        #     self.board = [[column + str(row) for column in string.ascii_lowercase[:8]]
        #                   for row in range(1, 9)]  # {a=1, b =2} map it this way
        # else:
        #     # untested
        #     self.board = [[column + str(row) if position.Position.check_free(row, column) else column + str(
        # row) for column in string.ascii_lowercase[:8]] for row in range(1,
        # 9)]  # + name from taken positions
        self.board = [[column + str(row) + ' ' + self.existing_figures[f'({row}, {string.ascii_lowercase.index(column) + 1})'].name if f'({row}, {string.ascii_lowercase.index(column) + 1})' in self.existing_figures else column + str(
            row) for column in string.ascii_lowercase[:8]] for row in range(1, 9)]  # can and should be greatly optimized

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
        for i in range(2, 9, 5):
            self.existing_figures[f'({1}, {i})'] = chessfigure.Knight(
                'white', position.Position(1, i))
        for i in range(2, 9, 5):
            self.existing_figures[f'({8}, {i})'] = chessfigure.Knight(
                'black', position.Position(8, i))
        for i in range(3, 8, 3):
            self.existing_figures[f'({1}, {i})'] = chessfigure.Bishop(
                'white', position.Position(1, i))
        for i in range(3, 8, 3):
            self.existing_figures[f'({8}, {i})'] = chessfigure.Bishop(
                'black', position.Position(8, i))
        self.existing_figures[f'({1}, {4})'] = chessfigure.Queen(
            'white', position.Position(1, 4))
        self.existing_figures[f'({8}, {4})'] = chessfigure.Queen(
            'black', position.Position(8, 4))
        self.existing_figures[f'({1}, {5})'] = chessfigure.King(
            'white', position.Position(1, 5))
        self.existing_figures[f'({8}, {5})'] = chessfigure.King(
            'black', position.Position(8, 5))

    def move_piece(self, command):
        old_position, new_position = command
        #print(old_position, new_position)
        old_column = string.ascii_lowercase.index(old_position[0]) + 1
        old_row = old_position[1]
        new_column = string.ascii_lowercase.index(new_position[0]) + 1
        new_row = new_position[1]
        #print(old_column, old_row, new_column, new_row)
        #print('before')
        #for key, value in self.existing_figures.items():
        #    print(key, value)

        #print(f'{new_row}, {new_column}')
        #print(f'({new_row}, {new_column})' in self.existing_figures)
        #print(self.existing_figures)
        if (f'({new_row}, {new_column})') in self.existing_figures:
            self.existing_figures[(f'({old_row}, {old_column})')].beat(
                new_row, new_column)
            self.existing_figures[f'({new_row}, {new_column})'] = self.existing_figures[(f'({old_row}, {old_column})')]
            del self.existing_figures[(f'({old_row}, {old_column})')]
        else:
            self.existing_figures[(f'({old_row}, {old_column})')].move(
                new_row, new_column)
            self.existing_figures[f'({new_row}, {new_column})'] = self.existing_figures[(f'({old_row}, {old_column})')]
            del self.existing_figures[(f'({old_row}, {old_column})')]
        #print('after')
        #for key, value in self.existing_figures.items():
        #    print(key, value)


if __name__ == '__main__':
    pass
