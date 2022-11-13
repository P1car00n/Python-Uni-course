#!/usr/bin/env python3

import string # string.ascii_lowercase; map it later using a dictionary?

class Chessboard:
    def __init__(self) -> None:
        self.board = [[column+str(row) for column in string.ascii_lowercase[:8]] for row in range(1, 9)]


if __name__ == '__main__':
    print('''

  _         _   _             _                                               _                   _ 
 | |    ___| |_( )___   _ __ | | __ _ _   _   ___  ___  _ __ ___   ___    ___| |__   ___  ___ ___| |
 | |   / _ \ __|// __| | '_ \| |/ _` | | | | / __|/ _ \| '_ ` _ \ / _ \  / __| '_ \ / _ \/ __/ __| |
 | |__|  __/ |_  \__ \ | |_) | | (_| | |_| | \__ \ (_) | | | | | |  __/ | (__| | | |  __/\__ \__ \_|
 |_____\___|\__| |___/ | .__/|_|\__,_|\__, | |___/\___/|_| |_| |_|\___|  \___|_| |_|\___||___/___(_)
                       |_|            |___/                                                         

''')
    ch1 = Chessboard()
    for i in ch1.board:
        print(' | ', end='')
        for j in i:
            print(j, end=' | ')
        print()
    for i in ch1.board:
        print(i)
        