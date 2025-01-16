import numpy as np
from minichess.concepts.concepts import in_check, random, threat_opp_queen, material_advantage, has_mate_threat, opponent_has_mate_threat, has_contested_open_file, center_control, king_safety, material_imbalance, threaten_high_value_pieces
import tensorflow as tf
from minichess.agents.lite_model import LiteModel
from minichess.agents.predictor_convnet import PredictorConvNet
from minichess.chess.move_utils import calculate_all_moves
from game_simulations.tournament import simulate_round_robin_tournament
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from minichess.chess.fastchess_utils import piece_matrix_to_legal_moves
from minichess.chess.move_utils import calculate_all_moves, index_to_move, move_to_index
from minichess.rl.chess_helpers import get_initial_chess_object
from tqdm import tqdm
import tensorflow.keras as keras
from minichess.agents.convnet import ConvNet


dataList = [
    {
        "full_name": "8x8standard",
        "model_name": "jimmy",
        'levels': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'dims': (8, 8),
        'concept_functions': [threat_opp_queen, in_check, has_mate_threat, material_advantage, center_control, king_safety, material_imbalance, threaten_high_value_pieces]
    }
]

def play_match(agents, full_name, dims, move_cap, all_moves, all_moves_inv, concept_function):
    chess = get_initial_chess_object(full_name)
    to_start = 1 if np.random.random() > 0.5 else 0
    current = to_start
    positive_cases = []
    negative_cases = []
    SAMPLING_RATIO = 0.2

    while chess.game_result() is None:
        if np.random.random() < SAMPLING_RATIO:
            if concept_function(chess):
                positive_cases.append(chess.agent_board_state())
            else:
                negative_cases.append(chess.agent_board_state())


        agent_to_play = agents[current]
        dist, value = agent_to_play.predict(chess.agent_board_state())

        moves, proms = chess.legal_moves()
        legal_moves = piece_matrix_to_legal_moves(moves, proms)
        legal_moves_mask = np.zeros((dims[0], dims[1], all_moves_inv.shape[0]))
        for move in legal_moves:
            (i, j), (dx, dy), promotion = move
            ind = move_to_index(all_moves, dx, dy, promotion, chess.turn)
            legal_moves_mask[i, j, ind] = 1

        dist = (dist + 0.5 * np.random.uniform(size=dist.shape)) * legal_moves_mask.flatten()

        dist /= dist.sum()
        move_to_play = np.argmax(dist)

        # move_to_play = np.random.choice(np.arange(dist.shape[0]), p=dist)
        i, j, ind = np.unravel_index(move_to_play, (dims[0], dims[1], move_cap))
        dx, dy, promotion = index_to_move(all_moves_inv, ind, chess.turn)
        chess.make_move(i, j, dx, dy, promotion)
        current = (current + 1) % 2
    return positive_cases, negative_cases

def load_model(full_name, model_name, epoch):
    keras_model = tf.keras.models.load_model("minichess/agents/checkpoints/{}/{}/{}".format(full_name, model_name, epoch))
    simple_model = PredictorConvNet(LiteModel.from_keras_model(keras_model))
    del keras_model
    return simple_model

if __name__ == "__main__":

    for data in dataList:
        full_name = data["full_name"]
        model_name = data["model_name"]
        levels = data["levels"]
        dims = data["dims"]
        CONCEPT_FUNCTIONS = data["concept_functions"]

        agents = [load_model(full_name, model_name, epoch) for epoch in levels]

        all_moves, all_moves_inv = calculate_all_moves(dims)
        move_cap = all_moves_inv.shape[0]

        # Run the round-robin tournament
        final_elo = simulate_round_robin_tournament(
            agents, full_name, dims, move_cap, all_moves, all_moves_inv, num_games_per_pair=60, initial_elo=800, max_k=600, min_k=40
        )

        for CONCEPT_FUNC in CONCEPT_FUNCTIONS:
            concept_name = CONCEPT_FUNC.__name__
            positive_cases = []
            negative_cases = []

            CASES_TO_COLLECT = 2000
            pbar = tqdm(total=CASES_TO_COLLECT)
            while len(positive_cases) < CASES_TO_COLLECT:
                pos, neg = play_match([agents[0], agents[2]], full_name, dims, move_cap, all_moves, all_moves_inv, CONCEPT_FUNC)
                positive_cases.extend(pos)
                negative_cases.extend(neg)
                pbar.update(len(pos))

            positive_cases = positive_cases[:CASES_TO_COLLECT]
            negative_cases = negative_cases[:CASES_TO_COLLECT]

            positive_cases = np.array(positive_cases)
            negative_cases = np.array(negative_cases)

            for epoch_to_look_at in levels:
                predictor_model = ConvNet(None, None, init=False)
                predictor_model.model = keras.models.load_model("minichess/agents/checkpoints/{}/{}/{}".format(full_name, model_name, epoch_to_look_at))
                all_cases = np.concatenate([positive_cases, negative_cases])
                all_labels = [1] * positive_cases.shape[0] + [0] * negative_cases.shape[0]
                all_labels = np.array(all_labels)
                shuffled_indices = np.arange(all_labels.shape[0])
                np.random.shuffle(shuffled_indices)
                all_cases = all_cases[shuffled_indices]
                all_labels = all_labels[shuffled_indices]
                POSITIONS_TO_CONSIDER = 3200
                VALIDATION_POSITIONS = 800
                from minichess.concepts.linear_regression import perform_linear_regression, perform_logistic_regression, perform_regression

                concept_presences = {}

                outputs = predictor_model.get_all_resblock_outputs(all_cases)
                merged_outputs = []
                for output_batch in outputs:
                    for i, output_layer in enumerate(output_batch):
                        if len(merged_outputs) <= i:
                            merged_outputs.append([])
                        merged_outputs[i].extend(output_layer)

                for i, layer_output in enumerate(merged_outputs):
                    merged_outputs[i] = np.array(merged_outputs[i])
                outputs = merged_outputs
                concept_presence_per_layer = []
                for (i, output) in enumerate(outputs):
                    points = output.reshape((output.shape[0], np.prod(output.shape[1:])))
                    score = perform_regression(
                        points[:POSITIONS_TO_CONSIDER],
                        all_labels[:POSITIONS_TO_CONSIDER],
                        points[POSITIONS_TO_CONSIDER:],
                        all_labels[POSITIONS_TO_CONSIDER:],
                        True
                    )
                    concept_presence_per_layer.append(score)

                concept_presences[concept_name] = concept_presence_per_layer
                import os
                import string
                from random import choices
                import json

                os.makedirs("concept_presences", exist_ok=True)
                os.makedirs("concept_presences/{}".format(full_name), exist_ok=True)
                os.makedirs("concept_presences/{}/{}".format(full_name, model_name), exist_ok=True)
                os.makedirs("concept_presences/{}/{}/{}".format(full_name, model_name, concept_name), exist_ok=True)
                os.makedirs("concept_presences/{}/{}/{}/{}".format(full_name, model_name, concept_name, epoch_to_look_at), exist_ok=True)

                random_suffix = ''.join(choices(string.ascii_uppercase + string.digits, k=10))

                with open("concept_presences/{}/{}/{}/{}/{}.json".format(full_name, model_name, concept_name, epoch_to_look_at, random_suffix), "w") as f:
                    json.dump(concept_presences[concept_name], f)

        for CONCEPT_FUNC in CONCEPT_FUNCTIONS:
            concept = CONCEPT_FUNC.__name__
            z = []

            for level in levels:
                presences = []
                for file in os.listdir("concept_presences/{}/{}/{}/{}".format(full_name, model_name, concept, level)):
                    with open(os.path.join("concept_presences/{}/{}/{}/{}".format(full_name, model_name, concept, level), file)) as f:
                        data = json.load(f)
                        presences.append(data)

                y = []

                for (i, presence) in enumerate(presences):
                    y = []
                    for j, ind_presence in enumerate(presence):
                        y.append(ind_presence)
                    z.append(y)
            z = np.array(z)

            mpl.style.use("seaborn-muted")
            mpl.rcParams['figure.figsize'] = (10, 11)
            mpl.rcParams['lines.linewidth'] = 20.0

            mpl.rcParams['font.family'] = "serif"
            mpl.rcParams["axes.axisbelow"] = True
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            # Make data.
            X = np.arange(len(levels))
            Y = np.arange(z.shape[1])
            X, Y = np.meshgrid(Y, X)

            # Plot the surface.
            surf = ax.plot_surface(X, Y, z, cmap=cm.plasma, edgecolor="white", linewidth=0.25, vmin=0.1, vmax=0.9)
            # Customize the z axis.
            ax.set_zlim(0, 1.00)
            ax.set_axisbelow(False)
            ax.set_title(f"{model_name}: {concept}", fontsize=20)
            ax.set_xlabel("Layer number", labelpad=15, fontsize=20, zorder=10)
            ax.set_ylabel("Checkpoint", labelpad=55,fontsize=20,zorder=10)
            plt.xticks(fontsize=15, rotation=0)
            plt.xticks(np.arange(z.shape[1]))
            plt.yticks(fontsize=15, rotation=-40)
            ax.tick_params('z', labelsize=15, pad=10, reset=True)
            fig.patch.set_facecolor("white")

            ax.zaxis.set_major_locator(LinearLocator(5))

            labels = [""] + [f"{level} (ELO {int(round(final_elo.get(idx, 0), 0))})" for idx, level in enumerate(levels)]
            plt.yticks(np.arange(len(labels)), labels)
            yticks = ax.yaxis.get_major_ticks()

            yticks[-1].label1.set_visible(True)
            # A StrMethodFormatter is used automatically
            ax.zaxis.set_major_formatter('{x:.02f}')
            ax.view_init(30, -30)

            os.makedirs("plots/{}/{}".format(full_name, model_name), exist_ok=True)
            plt.savefig("plots/{}/{}/{}_{}.png".format(full_name, model_name, model_name, concept), transparent=False)
