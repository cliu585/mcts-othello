#ifndef MCTS_LEAF_H
#define MCTS_LEAF_H

#include "mcts.h"

MCTSTiming mcts_leaf_parallel(Node *root, int iterations);

#endif