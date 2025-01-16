from minichess.chess.fastchess import Chess
from minichess.chess.fastchess_utils import inv_color, piece_matrix_to_legal_moves
import numpy as np


def in_check(chess: Chess):
    king_pos = chess.single_to_bitboard(*chess.find_king(chess.turn))
    return king_pos & chess.get_attacked_squares(inv_color(chess.turn), False)
    if chess.legal_move_cache is None:
        chess.legal_moves()
    return chess.any_checkers


def has_contested_open_file(chess: Chess):
    # First, find positions of own queens and rooks:
    squares_to_look_at = []
    for i in range(chess.dims[0]):
        for j in range(chess.dims[1]):
            piece_at = chess.piece_at(i, j, chess.turn)
            if piece_at == 4 or piece_at == 5:
                squares_to_look_at.append((i, j))

    for (i, j) in squares_to_look_at:
        # If the file, i, only has enemy and own rooks and queens on it, it is open and contested
        enemy_occupant_found = False
        for k in range(chess.dims[0]):
            own_piece_at = chess.piece_at(k, j, chess.turn)
            # Own piece on this file that is not a rook or queen
            if own_piece_at != -1 and own_piece_at != 4 and own_piece_at != 5:
                break
            enemy_piece_at = chess.piece_at(k, j, inv_color(chess.turn))
            # Enemy occupant on the file, ok...
            if enemy_piece_at == 4 or enemy_piece_at == 5:
                enemy_occupant_found = True
            # Some other enemy piece on the file
            if enemy_piece_at != 4 and enemy_piece_at != 5 and enemy_piece_at != -1:
                break
        else:
            # This means the file is open, and if it has an enemy oppucant as well, it is contested
            if enemy_occupant_found:
                return True
    return False


def opponent_has_mate_threat(chess: Chess):
    # Essentially, pass the turn over, and see if enemy (now player to move) has mate
    to_check = chess.copy()
    to_check.make_null_move()
    return has_mate_threat(to_check)


def has_mate_threat(chess: Chess):
    moves, proms = chess.legal_moves()
    legal_moves = piece_matrix_to_legal_moves(moves, proms)
    for move in legal_moves:
        (i, j), (dx, dy), prom = move
        potential_mate = chess.copy()
        potential_mate.make_move(i, j, dx, dy, prom)
        if potential_mate.game_result() is not None and abs(potential_mate.game_result()) == 1:
            return True
    return False


def threat_opp_queen(chess: Chess):
    if chess.legal_move_cache is None:
        chess.legal_moves()

    enemy_turn = inv_color(chess.turn)
    if chess.bitboards[enemy_turn, 4] == 0:
        return False

    queen_pos = chess.find_queen(enemy_turn)

    moves, proms = chess.legal_moves()
    legal_moves = piece_matrix_to_legal_moves(moves, proms)
    for move in legal_moves:
        (i, j), (dx, dy), prom = move
        if i + dx == queen_pos[0] and j + dy == queen_pos[1]:
            return True

    return False


def material_advantage(position: Chess):
    total = sum_of_pieces(position, position.turn) - sum_of_pieces(position, inv_color(position.turn))
    return total >= 3


def random(position: Chess):
    return np.random.random() > 0.5


def sum_of_pieces(position: Chess, color: bool):
    piece_values = {
        0: 1,
        1: 2,
        2: 2,
        3: 4,
        4: 9,
        5: 0
    }
    total = 0
    for i in range(position.dims[0]):
        for j in range(position.dims[1]):
            if position.piece_lookup[color, i, j] == -1:
                continue
            total += piece_values[position.piece_lookup[color, i, j]]

    return total


# custom concepts

def center_control(chess: Chess):
    # luam dimensiunile tablei
    rows, cols = chess.dims

    # Calculam zona centrala a tablei de sah
    center_rows = range(rows // 2 - 1, rows // 2 + 1) if rows % 2 == 0 else range(rows // 2, rows // 2 + 1)
    center_cols = range(cols // 2 - 1, cols // 2 + 1) if cols % 2 == 0 else range(cols // 2, cols // 2 + 1)

    # Calculam pozitiile patratelor centrale
    center_squares = [(i, j) for i in center_rows for j in center_cols]

    # Numaram cate patrate centrale sunt controlate de jucator
    control_count = 0
    for square in center_squares:
        if chess.single_to_bitboard(*square) & chess.get_attacked_squares(chess.turn, False):
            control_count += 1

    #n_cen_center_squares = len(center_squares) / 2 if len(center_squares) % 2 == 0 else len(center_squares) / 2 + 1
    return control_count >= 2



def king_safety(chess: Chess):
    king_pos = chess.find_king(chess.turn)
    if not king_pos:
        return 0  # Nu exista rege (edge case)

    # Deltas pentru a verifica patratele din jurul regelui
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    safety_score = 0
    for dx, dy in deltas:
        x, y = king_pos[0] + dx, king_pos[1] + dy
        if 0 <= x < chess.dims[0] and 0 <= y < chess.dims[1]:
            piece_at = chess.piece_at(x, y, chess.turn)
            if piece_at != -1:
                safety_score += 1  # Exista o piesa a jucatorului in jurul regelui
    return safety_score > 4

def material_imbalance(chess: Chess):
    # Pion=1, Cal=Nebun=3, Tura=5, Regina=9, Rege=0 -- la fel cum sunt in proiect evaluate
    piece_values = {0: 1, 1: 3, 2: 3, 3: 5, 4: 9, 5: 0}
    imbalance_score = 0

    #calculam diferenta de material
    for i in range(chess.dims[0]):
        for j in range(chess.dims[1]):
            piece = chess.piece_at(i, j, chess.turn)
            # Daca exista o piesa la pozitia respectiva adaugam valoarea ei la scor
            if piece != -1:
                imbalance_score += piece_values.get(piece, 0)

    return imbalance_score < 11

def threaten_high_value_pieces(chess: Chess):
    high_value_pieces = {5, 4, 0} # Regina, Tura, Rege
    threatened_pieces = 0 # Numarul de piese importante amenintate

    enemy_turn = inv_color(chess.turn) # Jucatorul inamic
    for i in range(chess.dims[0]):
        for j in range(chess.dims[1]):
            # Daca piesa este una importanta
            if chess.piece_at(i, j, enemy_turn) in high_value_pieces:
                # Daca piesa este amenintata
                if chess.single_to_bitboard(i, j) & chess.get_attacked_squares(chess.turn, False):
                    threatened_pieces += 1
    return threatened_pieces

