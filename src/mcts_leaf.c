#include <omp.h>
#include <stdint.h>
#include "mcts_leaf.h"

void mcts_leaf_parallel(Node *root, int iterations) {
    if (root == NULL) return;

    int groups = iterations / ROLLOUTS;
    if (groups == 0) groups = 1;

    for (int g = 0; g < groups; g++) {
        Node *node = root;

        // Selection - single-threaded 
        while (node->num_children > 0) {
            node = select_child(node);
        }

        // Expansion - single-threaded
        if (node->visits > 0 && has_valid_moves(&node->state)) {
            expand(node);
            if (node->num_children > 0) {
                node = node->children[rand() % node->num_children];
            }
        }
        GameState base_state = node->state;
        int original_player = base_state.player;
        unsigned int seed_base = (unsigned int)rand() ^ (unsigned int)time(NULL) ^ (unsigned int)(g * 0x9e3779b9u);
        #pragma omp parallel for schedule(dynamic) firstprivate(base_state, original_player, seed_base)
        for (int r = 0; r < ROLLOUTS; r++) {
            unsigned int thread_seed = seed_base ^ (unsigned int)(r * 0x9e3779b9u) ^ (unsigned int)omp_get_thread_num();
            GameState state_copy = base_state;

            double result = simulate(&state_copy, original_player, &thread_seed, 1); 
            Node *n = node;
            while (n != NULL) {
                double add;
                if (n->player_just_moved == original_player) {
                    add = result;           // wins for player_just_moved
                } else {
                    add = 1.0 - result;     // opponent's wins
                }

                #pragma omp atomic
                n->visits += 1;

                #pragma omp atomic
                n->wins += add;

                n = n->parent;
            }
        }
    }
}