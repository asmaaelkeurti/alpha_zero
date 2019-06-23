from __future__ import print_function
import sys
from Game import Game
from GobangLogic import Board
import numpy as np
sys.path.append('..')


class GobangGame(Game):
    def __init__(self, n=10, nir=5):
        self.n = n
        self.n_in_row = nir

    def get_init_board(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self):
        # (a,b) tuple
        return self.n, self.n

    def get_action_size(self):
        # return number of actions
        return self.n * self.n + 1

    def get_next_state(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return board, -player
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return b.pieces, -player

    # modified
    def get_valid_moves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.get_action_size()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legal_moves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    # modified
    def get_game_ended(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        n = self.n_in_row

        for w in range(self.n):
            for h in range(self.n):
                if (w in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[i][h] for i in range(w, w + n))) == 1):
                    return board[w][h]
                if (h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w][j] for j in range(h, h + n))) == 1):
                    return board[w][h]
                if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w + k][h + k] for k in range(n))) == 1):
                    return board[w][h]
                if (w in range(self.n - n + 1) and h in range(n - 1, self.n) and board[w][h] != 0 and
                        len(set(board[w + l][h - l] for l in range(n))) == 1):
                    return board[w][h]
        if b.has_legal_moves():
            return 0
        return 1e-4

    def get_canonical_form(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    # modified
    def get_symmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                l += [(new_b, list(new_pi.ravel()) + [pi[-1]])]
        return l

    def string_representation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()


def display(board):
    # n = board.shape[0]
    n = len(board)

    for y in range(n):
        print(y, "|", end="  ")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|", end="  ")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1:
                print("O ", end="  ")
            elif piece == 1:
                print("X ", end="  ")
            else:
                if x == n:
                    print("-", end="  ")
                else:
                    print("- ", end="  ")
        print("|")

    print("   -----------------------")


def display_pi(board):
    # n = board.shape[0]
    n = len(board)

    for y in range(n):
        print(y, "|", end="  ")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|", end="  ")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            print("%.3f" % piece, end="  ")
        print("|")

    print("   -----------------------")