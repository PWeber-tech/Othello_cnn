"""
play_othello.py

Play Othello against a trained model using a pygame window.

Usage:
    python play_othello.py
    python play_othello.py --difficulty hard
    python play_othello.py --difficulty medium --color w
"""

import argparse
import os
import sys
import torch
import pygame
from pygame.locals import *
from neural_net import OthelloNet, select_move_from_net, random_move
from othello_game1 import Board

# ── Constants ─────────────────────────────────────────────────────────────────
SIZE        = 8
CELL        = 64           # pixels per cell
MARGIN      = 8            # board inset
BOARD_PX    = SIZE * CELL  # 512
INFO_HEIGHT = 100          # status bar below board
WIN_W       = BOARD_PX + MARGIN * 2
WIN_H       = BOARD_PX + MARGIN * 2 + INFO_HEIGHT

# Colours
GREEN_DARK  = (20,  120,  60)
GREEN_LIGHT = (34,  139,  74)
LINE_COL    = (10,   80,  40)
BLACK_PIECE = (15,   15,  15)
WHITE_PIECE = (240, 240, 240)
LEGAL_COL   = (80,  180, 120)
HINT_ALPHA  = 140           # transparency of legal move dots
BG_COL      = (30,   30,  30)
TEXT_COL    = (230, 230, 230)
ACCENT_COL  = (100, 160, 240)
WIN_COL     = (255, 215,   0)
SHADOW_COL  = (0,     0,   0, 80)

# ── Difficulty presets ────────────────────────────────────────────────────────
DIFFICULTIES = {
    "random": {
        "label":       "Random",
        "checkpoint":  None,
        "epsilon":     1.0,
        "description": "Picks moves completely at random"
    },
    "easy": {
        "label":       "20k",
        "checkpoint":  "checkpoints/checkpoint_game_20000.pt",
        "epsilon":     0.6,
        "description": "Early model, high randomness"
    },
    "medium": {
        "label":       "60k",
        "checkpoint":  "checkpoints/checkpoint_game_60000.pt",
        "epsilon":     0.2,
        "description": "Mid-training model"
    },
    "hard": {
        "label":       "final 100k",
        "checkpoint":  "checkpoints/final_model.pt",
        "epsilon":     0.0,
        "description": "Final model, always plays best move"
    },
}

# ── Network helpers ───────────────────────────────────────────────────────────
def move_to_index(move):
    row, col = move
    return row * SIZE + col

def index_to_move(index):
    return index // SIZE, index % SIZE

def load_net(path):
    if not path or not os.path.exists(path):
        return None
    net = OthelloNet()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net

def get_ai_move(net, board, legal_moves, epsilon):
    legal_indices = [move_to_index(m) for m in legal_moves]
    if net is None or epsilon >= 1.0:
        return index_to_move(random_move(legal_indices))
    state_tensor = board.return_board()
    return index_to_move(
        select_move_from_net(net, state_tensor, legal_indices, epsilon)
    )

def count_pieces(board):
    black = white = 0
    for r in range(SIZE):
        for c in range(SIZE):
            ch = board.board[r][c].return_chip()
            if ch == "0": black += 1
            elif ch == "1": white += 1
    return black, white

# ── Drawing helpers ───────────────────────────────────────────────────────────
def cell_rect(row, col):
    """Pixel rect for a board cell."""
    x = MARGIN + col * CELL
    y = MARGIN + row * CELL
    return pygame.Rect(x, y, CELL, CELL)

def cell_center(row, col):
    r = cell_rect(row, col)
    return r.centerx, r.centery

def draw_board(surface):
    """Draw the green board with grid lines."""
    board_rect = pygame.Rect(MARGIN, MARGIN, BOARD_PX, BOARD_PX)
    pygame.draw.rect(surface, GREEN_DARK, board_rect)

    for r in range(SIZE):
        for c in range(SIZE):
            rect = cell_rect(r, c)
            shade = GREEN_LIGHT if (r + c) % 2 == 0 else GREEN_DARK
            pygame.draw.rect(surface, shade, rect)
            pygame.draw.rect(surface, LINE_COL, rect, 1)

def draw_pieces(surface, board):
    """Draw all placed pieces with a subtle shadow."""
    radius = CELL // 2 - 6
    for r in range(SIZE):
        for c in range(SIZE):
            ch = board.board[r][c].return_chip()
            if ch == " ":
                continue
            cx, cy = cell_center(r, c)
            # shadow
            shadow_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, (0, 0, 0, 70),
                                (CELL//2 + 2, CELL//2 + 3), radius)
            surface.blit(shadow_surf, (cx - CELL//2, cy - CELL//2))
            # piece
            color = BLACK_PIECE if ch == "0" else WHITE_PIECE
            edge  = (80, 80, 80) if ch == "0" else (160, 160, 160)
            pygame.draw.circle(surface, color, (cx, cy), radius)
            pygame.draw.circle(surface, edge,  (cx, cy), radius, 2)

def draw_legal_moves(surface, legal_moves):
    """Draw semi-transparent green dots on legal squares."""
    dot_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
    pygame.draw.circle(dot_surf, (*LEGAL_COL, HINT_ALPHA),
                       (CELL//2, CELL//2), CELL//6)
    for (r, c) in legal_moves:
        rect = cell_rect(r, c)
        surface.blit(dot_surf, rect.topleft)

def draw_highlight(surface, row, col):
    """Highlight the last move played."""
    rect = cell_rect(row, col).inflate(-4, -4)
    pygame.draw.rect(surface, ACCENT_COL, rect, 3, border_radius=4)

def draw_info(surface, board, human_player, difficulty_label,
              status_msg, font_lg, font_sm):
    """Draw the status bar below the board."""
    info_y = MARGIN * 2 + BOARD_PX
    info_rect = pygame.Rect(0, info_y, WIN_W, INFO_HEIGHT)
    pygame.draw.rect(surface, BG_COL, info_rect)

    black, white = count_pieces(board)
    human_sym  = "●" if human_player == "0" else "○"
    ai_sym     = "○" if human_player == "0" else "●"
    human_cnt  = black if human_player == "0" else white
    ai_cnt     = white if human_player == "0" else black

    score_txt = font_lg.render(
        f"{human_sym} You: {human_cnt}    {ai_sym} AI ({difficulty_label}): {ai_cnt}",
        True, TEXT_COL
    )
    surface.blit(score_txt, (MARGIN, info_y + 10))

    status_surf = font_sm.render(status_msg, True, ACCENT_COL)
    surface.blit(status_surf, (MARGIN, info_y + 48))

    hint = font_sm.render("Click a green dot to move  |  ESC to quit", True, (120, 120, 120))
    surface.blit(hint, (MARGIN, info_y + 72))

def draw_game_over(surface, board, human_player, font_lg, font_sm):
    """Overlay showing final result."""
    black, white = count_pieces(board)
    winner = board.who_winner()

    if winner == 0:
        msg = "It's a draw!"
        col = TEXT_COL
    elif (winner == 1 and human_player == "0") or \
         (winner == -1 and human_player == "1"):
        msg = "You win!"
        col = WIN_COL
    else:
        msg = "AI wins!"
        col = (220, 80, 80)

    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    txt1 = font_lg.render(msg, True, col)
    txt2 = font_sm.render(f"Final score — ● Black: {black}   ○ White: {white}", True, TEXT_COL)
    txt3 = font_sm.render("Press R to play again or ESC to quit", True, (160, 160, 160))

    cx = WIN_W // 2
    cy = WIN_H // 2
    surface.blit(txt1, txt1.get_rect(center=(cx, cy - 40)))
    surface.blit(txt2, txt2.get_rect(center=(cx, cy + 10)))
    surface.blit(txt3, txt3.get_rect(center=(cx, cy + 50)))

# ── Menu screen ───────────────────────────────────────────────────────────────
def run_menu(surface, clock, font_lg, font_sm, font_xl):
    """Simple difficulty + color selector. Returns (difficulty_key, human_player)."""
    difficulty_keys = list(DIFFICULTIES.keys())
    selected_diff   = 2   # default: medium
    selected_color  = 0   # 0 = Black, 1 = White

    while True:
        surface.fill(BG_COL)

        title = font_xl.render("OthelloNet", True, WIN_COL)
        surface.blit(title, title.get_rect(center=(WIN_W//2, 80)))

        sub = font_sm.render("Play against a trained neural network", True, (150, 150, 150))
        surface.blit(sub, sub.get_rect(center=(WIN_W//2, 125)))

        # Difficulty buttons
        diff_label = font_sm.render("Difficulty:", True, TEXT_COL)
        surface.blit(diff_label, (MARGIN + 20, 180))

        btn_w, btn_h = 120, 44
        gap = 12
        total_w = len(difficulty_keys) * btn_w + (len(difficulty_keys)-1) * gap
        start_x = (WIN_W - total_w) // 2

        diff_rects = []
        for i, key in enumerate(difficulty_keys):
            bx = start_x + i * (btn_w + gap)
            by = 210
            rect = pygame.Rect(bx, by, btn_w, btn_h)
            diff_rects.append(rect)
            color = ACCENT_COL if i == selected_diff else (60, 60, 60)
            pygame.draw.rect(surface, color, rect, border_radius=8)
            pygame.draw.rect(surface, (100, 100, 100), rect, 1, border_radius=8)
            lbl = font_sm.render(DIFFICULTIES[key]["label"], True, TEXT_COL)
            surface.blit(lbl, lbl.get_rect(center=rect.center))

        # Description of selected difficulty
        desc = font_sm.render(
            DIFFICULTIES[difficulty_keys[selected_diff]]["description"],
            True, (160, 160, 160)
        )
        surface.blit(desc, desc.get_rect(center=(WIN_W//2, 280)))

        # Color selector
        col_label = font_sm.render("You play as:", True, TEXT_COL)
        surface.blit(col_label, (MARGIN + 20, 320))

        color_options = [("● Black (first)", "0"), ("○ White (second)", "1")]
        color_rects = []
        cbtn_w = 200
        cstart = (WIN_W - (2*cbtn_w + gap)) // 2
        for i, (lbl_str, _) in enumerate(color_options):
            bx = cstart + i * (cbtn_w + gap)
            rect = pygame.Rect(bx, 350, cbtn_w, btn_h)
            color_rects.append(rect)
            color = ACCENT_COL if i == selected_color else (60, 60, 60)
            pygame.draw.rect(surface, color, rect, border_radius=8)
            pygame.draw.rect(surface, (100, 100, 100), rect, 1, border_radius=8)
            lbl = font_sm.render(lbl_str, True, TEXT_COL)
            surface.blit(lbl, lbl.get_rect(center=rect.center))

        # Start button
        start_rect = pygame.Rect(WIN_W//2 - 100, 430, 200, 52)
        pygame.draw.rect(surface, (50, 160, 80), start_rect, border_radius=10)
        start_txt = font_lg.render("Play", True, (255, 255, 255))
        surface.blit(start_txt, start_txt.get_rect(center=start_rect.center))

        esc_hint = font_sm.render("ESC to quit", True, (80, 80, 80))
        surface.blit(esc_hint, esc_hint.get_rect(center=(WIN_W//2, WIN_H - 20)))

        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == K_RETURN:
                    return difficulty_keys[selected_diff], color_options[selected_color][1]
            if event.type == MOUSEBUTTONDOWN:
                mx, my = event.pos
                for i, rect in enumerate(diff_rects):
                    if rect.collidepoint(mx, my):
                        selected_diff = i
                for i, rect in enumerate(color_rects):
                    if rect.collidepoint(mx, my):
                        selected_color = i
                if start_rect.collidepoint(mx, my):
                    return difficulty_keys[selected_diff], color_options[selected_color][1]

# ── Main game loop ────────────────────────────────────────────────────────────
def run_game(surface, clock, font_lg, font_sm,
             net, epsilon, difficulty_label, human_player):
    board = Board(display=False)
    last_move = None
    ai_player = "1" if human_player == "0" else "0"
    status = "Your turn" if board.player == human_player else "AI is thinking..."
    game_done = False
    ai_delay  = 0   # small delay so AI doesn't move instantly

    while True:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return "menu"
                if event.key == K_r and game_done:
                    return "menu"
            if event.type == MOUSEBUTTONDOWN and not game_done:
                if board.player == human_player:
                    mx, my = event.pos
                    col = (mx - MARGIN) // CELL
                    row = (my - MARGIN) // CELL
                    legal = board.find_legal_moves()
                    if 0 <= row < SIZE and 0 <= col < SIZE and (row, col) in legal:
                        board.make_play(row, col)
                        last_move = (row, col)
                        board.play_swap()
                        status = "AI is thinking..."
                        ai_delay = pygame.time.get_ticks() + 400

        # ── AI move ───────────────────────────────────────────────────────────
        if not game_done and board.player == ai_player:
            now = pygame.time.get_ticks()
            if now >= ai_delay:
                legal = board.find_legal_moves()
                if legal:
                    row, col = get_ai_move(net, board, legal, epsilon)
                    board.make_play(row, col)
                    last_move = (row, col)
                    board.play_swap()
                    status = "Your turn"
                else:
                    board.play_swap()
                    status = "AI has no moves — your turn"

        # ── Handle skipped human turn ─────────────────────────────────────────
        if not game_done and board.player == human_player:
            legal = board.find_legal_moves()
            if not legal:
                status = "No legal moves — turn skipped"
                board.play_swap()
                ai_delay = pygame.time.get_ticks() + 600

        # ── Game over check ───────────────────────────────────────────────────
        if not game_done and board.game_over():
            game_done = True
            status = ""

        # ── Draw ──────────────────────────────────────────────────────────────
        surface.fill((50, 50, 50))
        draw_board(surface)

        if last_move:
            draw_highlight(surface, *last_move)

        legal_moves = board.find_legal_moves() if not game_done else []
        if board.player == human_player:
            draw_legal_moves(surface, legal_moves)

        draw_pieces(surface, board)
        draw_info(surface, board, human_player, difficulty_label,
                  status, font_lg, font_sm)

        if game_done:
            draw_game_over(surface, board, human_player, font_lg, font_sm)

        pygame.display.flip()
        clock.tick(60)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Play Othello vs OthelloNet")
    parser.add_argument("--difficulty", choices=DIFFICULTIES.keys())
    parser.add_argument("--color", choices=["b", "w"])
    args = parser.parse_args()

    pygame.init()
    surface = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Othello — vs OthelloNet")
    clock = pygame.time.Clock()

    font_xl = pygame.font.SysFont("Arial", 42, bold=True)
    font_lg = pygame.font.SysFont("Arial", 22, bold=True)
    font_sm = pygame.font.SysFont("Arial", 16)

    while True:
        # Menu (or skip if args provided)
        if args.difficulty and args.color:
            difficulty = args.difficulty
            human_player = "0" if args.color == "b" else "1"
            args.difficulty = None   # only skip menu once
        else:
            difficulty, human_player = run_menu(surface, clock, font_lg, font_sm, font_xl)

        preset = DIFFICULTIES[difficulty]
        checkpoint = preset["checkpoint"]
        epsilon    = preset["epsilon"]

        if checkpoint and not os.path.exists(checkpoint):
            print(f"Warning: checkpoint not found: {checkpoint} — falling back to random")
            checkpoint = None
            epsilon    = 1.0

        net = load_net(checkpoint)

        result = run_game(
            surface, clock, font_lg, font_sm,
            net, epsilon, preset["label"], human_player
        )
        if result == "menu":
            continue

if __name__ == "__main__":
    main()
