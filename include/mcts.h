#ifndef MCTS_H
#define MCTS_H

#include "othello.h"
#include "mcts_leaf.h"
#include "mcts_root.h"

// virtual loss amount (tunable)
#define VIRTUAL_LOSS 1.0

// maximum path length for a single simulation (adjust if necessary)
#define MAX_PATH_LEN 1024

#define ROLLOUTS 20

typedef struct {
    Node *root;
    int iterations;
} ThreadData;

double ucb1(Node *node);

Node* select_child(Node *node);

void expand(Node *node);

double simulate(GameState *state, int original_player, unsigned int *seed, int include_seed);

void backpropagate(Node *node, double result);

void mcts_sequential(Node *root, int iterations);

void free_tree(Node *node);

#endif