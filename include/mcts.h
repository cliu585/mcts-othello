#ifndef MCTS_H
#define MCTS_H

#include "othello.h"


double ucb1(Node *node);

Node* select_child(Node *node);

void expand(Node *node);

double simulate(GameState *state, int original_player);

void backpropagate(Node *node, double result);

void mcts(Node *root, int iterations);
#endif