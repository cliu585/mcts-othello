#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mcts.h"

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

void backpropagate(Node *node, double result) {
    // Start from the leaf where simulation was run
    int sim_player = node->state.player;
    while (node != NULL) {
        node->visits++;
        if (node->player_just_moved == sim_player) {
            node->wins += result;
        } else {
            node->wins += (1.0 - result);
        }
        node = node->parent;
    }
}

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

// Initialize aggregator
void init_timing_aggregator(MCTSTimingAggregator *agg) {
    agg->total_selection = 0.0;
    agg->total_expansion = 0.0;
    agg->total_simulation = 0.0;
    agg->total_backpropagation = 0.0;
    agg->num_runs = 0;
}

// Add timing to aggregator
void add_timing(MCTSTimingAggregator *agg, const MCTSTiming *timing) {
    agg->total_selection += timing->selection;
    agg->total_expansion += timing->expansion;
    agg->total_simulation += timing->simulation;
    agg->total_backpropagation += timing->backpropagation;
    agg->num_runs++;
}

// Get average timing
MCTSTiming get_average_timing(const MCTSTimingAggregator *agg) {
    MCTSTiming avg = {0};
    if (agg->num_runs > 0) {
        avg.selection = agg->total_selection / agg->num_runs;
        avg.expansion = agg->total_expansion / agg->num_runs;
        avg.simulation = agg->total_simulation / agg->num_runs;
        avg.backpropagation = agg->total_backpropagation / agg->num_runs;
        avg.total = avg.selection + avg.expansion + avg.simulation + avg.backpropagation;
    }
    return avg;
}

// Print timing statistics
void print_timing(const MCTSTiming *timing, int iterations, const char *label) {
    printf("\n=== MCTS Phase Timing: %s (%d iterations) ===\n", label, iterations);
    printf("Selection:       %.6f s (%.2f%%)\n", timing->selection, 
           100.0 * timing->selection / timing->total);
    printf("Expansion:       %.6f s (%.2f%%)\n", timing->expansion, 
           100.0 * timing->expansion / timing->total);
    printf("Simulation:      %.6f s (%.2f%%)\n", timing->simulation, 
           100.0 * timing->simulation / timing->total);
    printf("Backpropagation: %.6f s (%.2f%%)\n", timing->backpropagation, 
           100.0 * timing->backpropagation / timing->total);
    printf("Total:           %.6f s\n", timing->total);
    printf("=================================================\n\n");
}

// void free_node_state(Node *node) {
//     if (node->state.board != NULL) {
//         for (int i = 0; i < node->state.size; i++) {
//             free(node->state.board[i]);
//         }
//         free(node->state.board);
//         node->state.board = NULL;
//     }
// }

// Helper function to free a tree
void free_tree(Node *node) {
    if (node == NULL) return;
    for (int i = 0; i < node->num_children; i++)
        free_tree(node->children[i]);
    free(node->children);
    free(node);
}