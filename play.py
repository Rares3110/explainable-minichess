import minichess.chess.fastchess_utils
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
# Example of using piece_matrix_to_legal_moves to get legal moves for an agent

# Assume `bitboards` is the current state of the board (as bitboards) and `promotions` is the promotion mask
legal_moves = piece_matrix_to_legal_moves(bitboards[turn], promotions)

# The `legal_moves` list will contain valid moves for the agent
for move in legal_moves:
    origin, deltas, promotion = move
    # Handle the move (you can convert this to UCI or use it directly in a game engine)
    print(f"Move from {origin} to {origin[0] + deltas[0], origin[1] + deltas[1]}, promotion: {promotion}")