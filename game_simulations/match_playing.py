from scipy.stats import norm
import numpy as np
import tensorflow as tf
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
from minichess.chess.move_utils import move_to_index, index_to_move
from minichess.rl.chess_helpers import get_initial_chess_object

def calculate_elo_update(rating_a, rating_b, result, k=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    rating_a_new = rating_a + k * (result - expected_a)
    rating_b_new = rating_b + k * ((1 - result) - (1 - expected_a))
    return rating_a_new, rating_b_new

def play_match_and_calculate_elo(agents, full_name, dims, move_cap, all_moves, all_moves_inv, elo_ratings, k=32):
    chess = get_initial_chess_object(full_name)
    to_start = 1 if np.random.random() > 0.5 else 0
    current = to_start

    while chess.game_result() is None:
        agent_to_play = agents[current]
        dist, value = agent_to_play.predict(chess.agent_board_state())

        moves, proms = chess.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        legal_moves_mask = np.zeros((dims[0], dims[1], all_moves_inv.shape[0]))
        for move in legal_moves:
            (i, j), (dx, dy), promotion = move
            ind = move_to_index(all_moves, dx, dy, promotion, chess.turn)
            legal_moves_mask[i, j, ind] = 1

        move_dims = dist.shape

        dist = (dist + 0.5 * np.random.uniform(size=dist.shape)) * legal_moves_mask.flatten()
        dist /= dist.sum()
        move_to_play = np.argmax(dist)

        i, j, ind = np.unravel_index(move_to_play, (dims[0], dims[1], move_cap))
        dx, dy, promotion = index_to_move(all_moves_inv, ind, chess.turn)
        chess.make_move(i, j, dx, dy, promotion)
        current = (current + 1) % 2

    result = chess.game_result()
    if result == 1:  # White wins
        result_for_elo = 1 if to_start == 0 else 0
        #print("WIN", end=" ")
    elif result == 2:  # Black wins
        result_for_elo = 0 if to_start == 0 else 1
        #print("LOSE", end=" ")
    else:
        result_for_elo = 0.5
        #print("EQUAL", end=" ")

    elo_ratings[0], elo_ratings[1] = calculate_elo_update(
        elo_ratings[0], elo_ratings[1], result_for_elo, k=k
    )

    return elo_ratings