#ifndef MCTS_H
#define MCTS_H

#include "othello.h"

// virtual loss amount (tunable)
#define VIRTUAL_LOSS 1.0

// maximum path length for a single simulation (adjust if necessary)
#define MAX_PATH_LEN 1024

typedef struct {
    Node *root;
    int iterations;
} ThreadData;

double ucb1(Node *node);

Node* select_child(Node *node);

void expand(Node *node);

double simulate(GameState *state, int original_player, unsigned int *seed, int include_seed);

void backpropagate(Node *node, double result);

void backpropagate_rollouts(Node *node, double total_result, int rollouts, int original_player);

void mcts(Node *root, int iterations);

void mcts_leaf_parallel(Node *root, int iterations);

Node* clone_node(Node *original, Node *new_parent);

// Standard MCTS iteration (same as your sequential version)
void mcts_iteration(Node *root, unsigned int *seed);

// Root parallelization implementation
void mcts_root_parallel(Node *root, int total_iterations);

// Alternative: Virtual loss approach (more efficient, no tree copying)
// This allows threads to share the same tree with synchronization
void mcts_root_parallel_virtual_loss(Node *root, int total_iterations);

void free_tree(Node *node);

#endif