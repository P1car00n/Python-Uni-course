#!/usr/bin/env python3

#import string

class Position():

    taken_pisitions = {}  # {(x=4, y=7): pawn7} # access with property

    def __init__(self, row, column) -> None:  # use decorators and make private x, y?
        Position.check_free(row, column)
        self.row = row
        self.column = column

    def check_free(row, column):
        if (row, column) in Position.taken_pisitions:
            return False
        else:
            return True


if __name__ == '__main__':
    pass
