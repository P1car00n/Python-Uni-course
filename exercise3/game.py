#!/usr/bin/env python3

import chessboard


class Game():
    def __init__(self) -> None:
        self.print_introduction()

    def print_introduction(self):
        print('''

  _         _   _             _                                               _                   _
 | |    ___| |_( )___   _ __ | | __ _ _   _   ___  ___  _ __ ___   ___    ___| |__   ___  ___ ___| |
 | |   / _ \\ __|// __| | '_ \\| |/ _` | | | | / __|/ _ \\| '_ ` _ \\ / _ \\  / __| '_ \\ / _ \\/ __/ __| |
 | |__|  __/ |_  \\__ \\ | |_) | | (_| | |_| | \\__ \\ (_) | | | | | |  __/ | (__| | | |  __/\\__ \\__ \\_|
 |_____\\___|\\__| |___/ | .__/|_|\\__,_|\\__, | |___/\\___/|_| |_| |_|\\___|  \\___|_| |_|\\___||___/___(_)
                       |_|            |___/

''')
        print(
            'I hope you know the rules. Grab another player, choose your colour and enjoy!',
            '\n')

    def set_players(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        # temporary
        print(f'{self.player1} plays white, {self.player2} plays black\n')


if __name__ == "__main__":
    game1 = Game()
    game1.set_players(
        input('Name the 1st player: '),
        input('Name the 2nd player: '))

    ch1 = chessboard.Chessboard()
    ch1.print_board()
