#!/usr/bin/env python3

#import string

class Position():

    taken_pisitions = {}  # {(x=4, y=7): pawn7}

    def __init__(self, row, column) -> None:  # use decorators and make private x, y?
        Position.check_free(row, column)
        self.row = row
        self.column = column

    def set_position():
        pass

    def change_position(self):
        pass

    def check_free(row, column):
        try:
            (row, column) in Position.taken_pisitions
        except KeyError:
            print('Alala')  # think of how to undo the operation for the user


if __name__ == '__main__':
    pass
