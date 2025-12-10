#ifndef MCTS_H
#define MCTS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "othello.h"
#include "mcts_util.h"
#include "mcts_leaf.h"
#include "mcts_root.h"

#define VIRTUAL_LOSS 1.0    // Virtual loss amount
#define MAX_PATH_LEN 1024   // Maximum path length for a single simulation
#define UCB_CONSTANT 1.414
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
MCTSTiming mcts_sequential(Node *root, int iterations);

#endif