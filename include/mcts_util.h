#ifndef MCTS_UTIL_H
#define MCTS_UTIL_H

#include <stdio.h>

#include "othello.h"

typedef struct {
    double selection;
    double expansion;
    double simulation;
    double backpropagation;
    double total;
} MCTSTiming;

typedef struct {
    double total_selection;
    double total_expansion;
    double total_simulation;
    double total_backpropagation;
    int num_runs;
} MCTSTimingAggregator;

typedef struct Node {
    GameState state;
    int move_row, move_col;
    int visits;
    double wins;  
    struct Node *parent;
    struct Node **children;
    int num_children;
    int player_just_moved;  
    omp_lock_t lock;  
} Node;

// Timing functions
void init_timing_aggregator(MCTSTimingAggregator *agg);
void add_timing(MCTSTimingAggregator *agg, const MCTSTiming *timing);
MCTSTiming get_average_timing(const MCTSTimingAggregator *agg);
void print_timing(const MCTSTiming *timing, int iterations, const char *label);

// Node functions
Node* create_node(GameState *state, int r, int c, Node *parent);
Node* clone_node(Node *original, Node *new_parent);
void free_tree(Node *node);

#endif