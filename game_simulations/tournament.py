from itertools import combinations
from game_simulations.match_playing import play_match_and_calculate_elo

def simulate_round_robin_tournament(
    agents, full_name, dims, move_cap, all_moves, all_moves_inv, num_games_per_pair=3, initial_elo=1500, max_k=64, min_k=16
):
    # ELO ratings initialization
    num_agents = len(agents)
    elo_ratings = {i: initial_elo for i in range(num_agents)}

    # All combinations of matches
    pairings = list(combinations(range(num_agents), 2))

    # Match simulation for each pair
    for agent1_id, agent2_id in pairings:
        #print(f"Playing matches between Agent {agent1_id} and Agent {agent2_id}")

        for game in range(num_games_per_pair):
            # Dynamically adjusted K-factor
            k = max_k - (game * (max_k - min_k) / (num_games_per_pair - 1)) if num_games_per_pair > 1 else max_k

            # Match simulation
            elo_ratings[agent1_id], elo_ratings[agent2_id] = play_match_and_calculate_elo(
                [agents[agent1_id], agents[agent2_id]],
                full_name,
                dims,
                move_cap,
                all_moves,
                all_moves_inv,
                [elo_ratings[agent1_id], elo_ratings[agent2_id]],
                k=k
            )

            '''print(
                f"Game {game + 1} (K={k:.2f}) between Agent {agent1_id} and Agent {agent2_id}: "
                f"Agent {agent1_id} ELO = {elo_ratings[agent1_id]:.2f}, "
                f"Agent {agent2_id} ELO = {elo_ratings[agent2_id]:.2f}"
            )'''

    return elo_ratings