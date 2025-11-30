#include <omp.h>
#include <stdint.h>
#include "mcts.h"

#define ROLLOUTS 20

double ucb1(Node *node) {
    if (node->visits == 0) return INFINITY;
    double exploitation = node->wins / node->visits;
    double exploration = UCB_CONSTANT * sqrt(log(node->parent->visits) / node->visits);
    return exploitation + exploration;
}

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

void expand(Node *node) {
    GameState *state = &node->state;
    int count = 0;
    
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j)) count++;
    
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


double simulate(GameState *state, int original_player, unsigned int *seed, int include_seed) {
    GameState sim = *state;
    
    while (1) {
        if (!has_valid_moves(&sim)) {
            sim.player = opponent(sim.player);
            if (!has_valid_moves(&sim)) break;
        }
        
        int moves[64][2], count = 0;
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

/*void backpropagate_rollouts(Node *node, double result, int visits) {
    // result is total wins from perspective of the player at the LEAF where simulation started
    // We need to propagate this up, flipping perspective at each level
    
    while (node != NULL) {
        node->visits += visits;
        node->wins += result;
        
        // Flip the result for the parent (opponent's perspective)
        result = visits - result;
        
        node = node->parent;
    }
}*/

void backpropagate_rollouts(Node *node, double total_result, int rollouts, int original_player) {
    if (node == NULL) return;
    double wins_from_leaf_player = total_result;      // wins as seen from the leaf's player (sum of 0..1)
    int r = rollouts;

    // Walk up the tree, flipping perspective at each step.
    while (node != NULL) {
        node->visits += r;

        if (node->player_just_moved == original_player) {
            node->wins += wins_from_leaf_player;
        } else {
            node->wins += (double)r - wins_from_leaf_player;
        }

        // Flip perspective for the parent node:
        wins_from_leaf_player = (double)r - wins_from_leaf_player;
        node = node->parent;
    }
}

void backpropagate(Node *node, double result) {
    // Start from the leaf where simulation was run
    // result is from perspective of state.player at that node
    int sim_player = node->state.player;
    
    while (node != NULL) {
        node->visits++;
        
        // Each node stores stats from perspective of player_just_moved
        // (the player who created this node by making a move)
        if (node->player_just_moved == sim_player) {
            node->wins += result;
        } else {
            node->wins += (1.0 - result);
        }
        
        node = node->parent;
    }
}

void mcts(Node *root, int iterations) {
    for (int i = 0; i < iterations; i++) {
        Node *node = root;
        
        // Selection
        while (node->num_children > 0) {
            node = select_child(node);
        }
        
        // Expansion
        if (node->visits > 0 && has_valid_moves(&node->state)) {
            expand(node);
            if (node->num_children > 0) {
                node = node->children[rand() % node->num_children];
            }
        }
        
        // Simulation - the result is from the perspective of the current player
        double result = simulate(&node->state, node->state.player, 0, 0);
        
        // Backpropagation
        backpropagate(node, result);
    }
}


void mcts_leaf_parallel(Node *root, int iterations) {
    if (root == NULL) return;

    int groups = iterations / ROLLOUTS;
    if (groups == 0) groups = 1;

    for (int g = 0; g < groups; g++) {
        Node *node = root;

        // Selection - single-threaded (same as sequential)
        while (node->num_children > 0) {
            node = select_child(node);
        }

        // Expansion - single-threaded
        if (node->visits > 0 && has_valid_moves(&node->state)) {
            expand(node);
            if (node->num_children > 0) {
                node = node->children[rand() % node->num_children];
            }
        }
        GameState base_state = node->state;
        int original_player = base_state.player;
        unsigned int seed_base = (unsigned int)rand() ^ (unsigned int)time(NULL) ^ (unsigned int)(g * 0x9e3779b9u);
        #pragma omp parallel for schedule(dynamic) firstprivate(base_state, original_player, seed_base)
        for (int r = 0; r < ROLLOUTS; r++) {
            unsigned int thread_seed = seed_base ^ (unsigned int)(r * 0x9e3779b9u) ^ (unsigned int)omp_get_thread_num();
            GameState state_copy = base_state;

            double result = simulate(&state_copy, original_player, &thread_seed, 1); // result in [0,1] w.r.t. original_player
            Node *n = node;
            while (n != NULL) {
                double add;
                if (n->player_just_moved == original_player) {
                    add = result;           // wins for player_just_moved
                } else {
                    add = 1.0 - result;     // opponent's wins
                }

                // Atomic updates to avoid races (int visits, double wins)
                #pragma omp atomic
                n->visits += 1;

                #pragma omp atomic
                n->wins += add;

                n = n->parent;
            }
        } 
    } 
}
