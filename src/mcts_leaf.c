#include <omp.h>
#include <stdint.h>
#include "mcts_leaf.h"


MCTSTiming mcts_leaf_parallel(Node *root, int iterations) {
    MCTSTiming timing = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    if (root == NULL) return timing;
    
    double total_start = omp_get_wtime();
    
    int groups = iterations / ROLLOUTS;
    if (groups == 0) groups = 1;

    for (int g = 0; g < groups; g++) {
        Node *node = root;

        // Selection - single-threaded
        double sel_start = omp_get_wtime();
        while (node->num_children > 0) {
            node = select_child(node);
        }
        double sel_end = omp_get_wtime();
        timing.selection += (sel_end - sel_start);

        // Expansion - single-threaded
        double exp_start = omp_get_wtime();
        if (node->visits > 0 && has_valid_moves(&node->state)) {
            expand(node);
            if (node->num_children > 0) {
                node = node->children[rand() % node->num_children];
            }
        }
        double exp_end = omp_get_wtime();
        timing.expansion += (exp_end - exp_start);

        GameState base_state = node->state;
        int original_player = base_state.player;
        unsigned int seed_base = (unsigned int)rand() ^ (unsigned int)time(NULL) ^ (unsigned int)(g * 0x9e3779b9u);
        
        // Track simulation and backpropagation times separately
        double sim_time = 0.0;
        double back_time = 0.0;
        
        double parallel_start = omp_get_wtime();
        #pragma omp parallel reduction(+:sim_time,back_time) firstprivate(base_state, original_player, seed_base)
        {
            #pragma omp for schedule(dynamic)
            for (int r = 0; r < ROLLOUTS; r++) {
                unsigned int thread_seed = seed_base ^ (unsigned int)(r * 0x9e3779b9u) ^ (unsigned int)omp_get_thread_num();
                GameState state_copy = base_state;

                // Simulation phase
                double sim_start = omp_get_wtime();
                double result = simulate(&state_copy, original_player, &thread_seed, 1);
                double sim_end = omp_get_wtime();
                sim_time += (sim_end - sim_start);
                
                // Backpropagation phase
                double back_start = omp_get_wtime();
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
                double back_end = omp_get_wtime();
                back_time += (back_end - back_start);
            }
        }
        double parallel_end = omp_get_wtime();
        
        timing.simulation += sim_time;
        timing.backpropagation += back_time;
    }
    
    double total_end = omp_get_wtime();
    timing.total = total_end - total_start;
    
    return timing;
}