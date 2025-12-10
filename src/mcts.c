#include "mcts.h"

// UCB1 node selection logic
double ucb1(Node *node) {
    if (node->visits == 0) return INFINITY;
    double exploitation = node->wins / node->visits;
    double exploration = UCB_CONSTANT * sqrt(log(node->parent->visits) / node->visits);
    return exploitation + exploration;
}

// MCTS selection phase
Node* select_child(Node *node) {
    Node *best = NULL;
    double best_ucb = -INFINITY;
    for (int i = 0; i < node->num_children; i++) {
        double ucb = ucb1(node->children[i]);
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best = node->children[i];
        }
    }
    return best;
}

// MCTS expansion phase
void expand(Node *node) {
    GameState *state = &node->state;
    int count = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j))
                count++;

    if (count == 0) return;

    node->children = malloc(count * sizeof(Node*));
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (is_valid_move(state, i, j)) {
                GameState new_state = *state;
                make_move(&new_state, i, j);
                node->children[node->num_children++] = create_node(&new_state, i, j, node);
            }
        }
    }
}

// MCTS simulation phase
double simulate(GameState *state, int original_player, unsigned int *seed, int include_seed) {
    GameState sim = *state;
    while (1) {
        if (!has_valid_moves(&sim)) {
            sim.player = opponent(sim.player);
            if (!has_valid_moves(&sim))
                break;
        }
        int board_size = SIZE * SIZE;
        int moves[board_size][2], count = 0;
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                if (is_valid_move(&sim, i, j)) {
                    moves[count][0] = i;
                    moves[count][1] = j;
                    count++;
                }

        if (count == 0) break;

        int idx;
        if (include_seed)
            idx = rand_r(seed) % count;
        else
            idx = rand() % count;
        make_move(&sim, moves[idx][0], moves[idx][1]);
    }

    int black, white;
    get_score(&sim, &black, &white);

    // Return win value from perspective of original_player
    if (original_player == BLACK) {
        if (black > white) return 1.0;
        else if (black < white) return 0.0;
        else return 0.5;
    } else {
        if (white > black) return 1.0;
        else if (white < black) return 0.0;
        else return 0.5;
    }
}

// MCTS backpropagation approach
void backpropagate(Node *node, double result) {
    int sim_player = node->state.player;
    while (node != NULL) {
        node->visits++;
        if (node->player_just_moved == sim_player) node->wins += result;
        else node->wins += (1.0 - result);
        node = node->parent;
    }
}

// MCTS sequential approach
MCTSTiming mcts_sequential(Node *root, int iterations) {
    MCTSTiming timing = {0};

    struct timespec start, end;
    
    for (int i = 0; i < iterations; i++) {
        Node *node = root;
        
        // Selection
        clock_gettime(CLOCK_MONOTONIC, &start);
        while (node->num_children > 0) {
            node = select_child(node);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        timing.selection += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Expansion
        clock_gettime(CLOCK_MONOTONIC, &start);
        if (node->visits > 0 && has_valid_moves(&node->state)) {
            expand(node);
            if (node->num_children > 0) {
                node = node->children[rand() % node->num_children];
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        timing.expansion += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Simulation
        clock_gettime(CLOCK_MONOTONIC, &start);
        double result = simulate(&node->state, node->state.player, 0, 0);
        clock_gettime(CLOCK_MONOTONIC, &end);
        timing.simulation += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        
        // Backpropagation
        clock_gettime(CLOCK_MONOTONIC, &start);
        backpropagate(node, result);
        clock_gettime(CLOCK_MONOTONIC, &end);
        timing.backpropagation += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    timing.total = timing.selection + timing.expansion + timing.simulation + timing.backpropagation;
    return timing;
}
