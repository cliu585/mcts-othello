#ifndef MCTS_H
#define MCTS_H

#include "othello.h"

double ucb1(Node *node);

Node* select_child(Node *node);

void expand(Node *node);

double simulate(GameState *state, int original_player, unsigned int *seed, int include_seed);

void backpropagate(Node *node, double result);

void backpropagate_rollouts(Node *node, double total_result, int rollouts, int original_player);

void mcts(Node *root, int iterations);

void mcts_leaf_parallel(Node *root, int iterations);

#endif