#include "mcts_util.h"

// NODE FUNCTIONS

// Deep copy a MCTS node and its entire subtree
Node* clone_node(Node *original, Node *new_parent) {
    if (original == NULL) return NULL;

    Node *clone = malloc(sizeof(Node));
    clone->state = original->state;
    clone->move_row = original->move_row;
    clone->move_col = original->move_col;
    clone->visits = 0;
    clone->wins = 0.0;
    clone->parent = new_parent;
    clone->player_just_moved = original->player_just_moved;
    clone->num_children = 0;
    clone->children = NULL;

    return clone;
}

// Make a MCTS tree node
Node* create_node(GameState *state, int r, int c, Node *parent) {
    Node *node = malloc(sizeof(Node));
    node->state = *state;
    node->move_row = r;
    node->move_col = c;
    node->visits = 0;
    node->wins = 0.0;
    node->parent = parent;
    node->children = NULL;
    node->num_children = 0;
    node->player_just_moved = parent ? opponent(state->player) : BLACK;
    omp_init_lock(&node->lock);
    return node;
}

// Free a MCTS tree
void free_tree(Node *node) {
    if (node == NULL) return;
    for (int i = 0; i < node->num_children; i++)
        free_tree(node->children[i]);
    free(node->children);
    free(node);
}


// TIMING FUNCTIONS

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
