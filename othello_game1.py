# Othello game engine
import torch
import pygame
from pygame.locals import *

size = 8

class Board:

    def __init__(self, display=False):
        self.board = []
        self.player = "0"
        self.display = display

        for i in range(size):
            self.board.append([])
            for j in range(size):
                self.board[i].append(Chip(j + 1, i + 1))

        if self.display:
            pygame.init()
            self.window = pygame.display.set_mode((456, 456))
            self.window.fill((0, 255, 0))
            for i in range(9):
                pygame.draw.rect(self.window, (58, 58, 58),
                                 [56 * i, 0, 8, 464], 0)
                pygame.draw.rect(self.window, (58, 58, 58),
                                 [0, 56 * i, 464, 8], 0)

        self._place_starting_pieces()

    def _place_starting_pieces(self):
        """Place the 4 starting pieces without triggering flips."""
        self.player = "0"
        self.board[3][3].place_chip("0")
        self.board[4][4].place_chip("0")
        self.player = "1"
        self.board[3][4].place_chip("1")
        self.board[4][3].place_chip("1")
        self.player = "0"

    def new_game(self):
        """Reset the board to the starting position."""
        for i in range(size):
            for j in range(size):
                self.board[i][j].symbol = " "
        self.player = "0"
        self._place_starting_pieces()

    def return_board(self):
        """Return board as a (2, 8, 8) tensor. Channel 0 = current player, Channel 1 = opponent."""
        state = torch.zeros((2, 8, 8), dtype=torch.float32)
        for i in range(size):
            for j in range(size):
                piece = self.board[i][j].return_chip()
                if piece == self.player:
                    state[0, i, j] = 1.0
                elif piece != " ":
                    state[1, i, j] = 1.0
        return state

    def game_over(self):
        move0 = len(self.find_legal_moves('0'))
        move1 = len(self.find_legal_moves('1'))
        return move0 == 0 and move1 == 0

    def find_legal_moves(self, curplayer=None):
        if curplayer is None:
            curplayer = self.player

        legal_moves = []
        for i in range(size):
            for j in range(size):
                if self.board[i][j].is_empty():
                    if self.check_legal_move(j + 1, i + 1, curplayer):
                        legal_moves.append((i, j))  # (row, col) in 0-indexed
        return legal_moves

    def check_legal_move(self, x, y, curplayer):
        """x, y are 1-indexed board coordinates."""
        directions = [
            (x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y),              (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)
        ]
        good_moves = [tup for tup in directions if all(0 < ele <= size for ele in tup)]

        for i in good_moves:
            tile = self.board[i[1]-1][i[0]-1]
            if (not tile.is_empty()) and (tile.return_chip() != curplayer):
                direction = (i[0] - x, i[1] - y)
                step = 2
                xcon = x + (direction[0] * step)
                ycon = y + (direction[1] * step)
                while (0 < xcon <= size) and (0 < ycon <= size):
                    cell = self.board[ycon-1][xcon-1]
                    if cell.return_chip() == curplayer:
                        return True
                    elif cell.is_empty():
                        break
                    step += 1
                    xcon = x + (direction[0] * step)
                    ycon = y + (direction[1] * step)
        return False

    def make_play(self, row, col):
        """row, col are 0-indexed."""
        x = col + 1  # convert to 1-indexed for internal logic
        y = row + 1
        self.board[row][col].place_chip(self.player)

        directions = [
            (x-1, y-1), (x, y-1), (x+1, y-1),
            (x-1, y),              (x+1, y),
            (x-1, y+1), (x, y+1), (x+1, y+1)
        ]
        good_moves = [tup for tup in directions if all(0 < ele <= size for ele in tup)]

        for i in good_moves:
            tile = self.board[i[1]-1][i[0]-1]
            if (not tile.is_empty()) and (tile.return_chip() != self.player):
                direction = (i[0] - x, i[1] - y)
                step = 2
                xcon = x + (direction[0] * step)
                ycon = y + (direction[1] * step)
                while (0 < xcon <= size) and (0 < ycon <= size):
                    cell = self.board[ycon-1][xcon-1]
                    if cell.return_chip() == self.player:
                        self.flip_chip(xcon, ycon, direction)
                        break
                    elif cell.is_empty():
                        break
                    step += 1
                    xcon = x + (direction[0] * step)
                    ycon = y + (direction[1] * step)

    def flip_chip(self, x, y, direction):
        step = 1
        xcon = x - (direction[0] * step)
        ycon = y - (direction[1] * step)
        while self.board[ycon-1][xcon-1].return_chip() != self.player:
            self.board[ycon-1][xcon-1].place_chip(self.player)
            step += 1
            xcon = x - (direction[0] * step)
            ycon = y - (direction[1] * step)

    def play_swap(self):
        self.player = "1" if self.player == "0" else "0"

    def who_winner(self):
        """
        Returns:
            1  if player '0' wins  (player 0 is encoded as +1 in training)
           -1  if player '1' wins  (player 1 is encoded as -1 in training)
            0  if draw
        """
        player0 = sum(
            1 for i in range(size) for j in range(size)
            if self.board[i][j].return_chip() == '0'
        )
        player1 = sum(
            1 for i in range(size) for j in range(size)
            if self.board[i][j].return_chip() == '1'
        )
        if player0 > player1:
            return 1
        elif player1 > player0:
            return -1
        return 0

    def print_board(self):
        if not self.display:
            return
        for i in range(size):
            for j in range(size):
                chip = self.board[i][j].return_chip()
                if chip == '0':
                    pygame.draw.circle(self.window, (0, 0, 0),
                                       [56 * j + 32, 56 * i + 32], 24, 0)
                elif chip == '1':
                    pygame.draw.circle(self.window, (255, 255, 255),
                                       [56 * j + 32, 56 * i + 32], 24, 0)
        pygame.display.update()
        input()


class Chip:
    def __init__(self, xpos, ypos):
        self.x = xpos
        self.y = ypos
        self.symbol = " "

    def return_chip(self):
        return self.symbol

    def place_chip(self, player):
        self.symbol = player

    def is_empty(self):
        return self.symbol == " "
