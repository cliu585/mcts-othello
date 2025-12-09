#ifndef MCTS_ROOT_H
#define MCTS_ROOT_H

#include "mcts.h"

MCTSTiming mcts_root_parallel(Node *root, int total_iterations);
MCTSTiming mcts_root_parallel_virtual_loss(Node *root, int total_iterations);

#endif