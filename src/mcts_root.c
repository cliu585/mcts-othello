#include <omp.h>
#include <stdint.h>
#include "mcts_root.h"

static inline double atomic_load_int(volatile int *p) {
    int v;
    #pragma omp atomic read
    v = *p;
    return (double) v;
}

static inline double atomic_load_double(volatile double *p) {
    double v;
    #pragma omp atomic read
    v = *p;
    return v;
}

// Atomic version of UCB1 selection logic
double ucb1_atomic(Node *child, Node *parent) {
    double child_visits    = atomic_load_int(&child->visits);
    double child_wins      = atomic_load_double(&child->wins);
    double parent_visits   = atomic_load_int(&parent->visits);

    if (child_visits == 0) return INFINITY;
    if (parent_visits == 0) parent_visits = 1; // prevent log(0)


    double exploitation = child_wins / child_visits;
    double exploration  = UCB_CONSTANT *
                          sqrt(log(parent_visits) / child_visits);

    return exploitation + exploration;
}

// MCTS selection phase using atomic UCB1
int select_child_index_parallel(Node *parent) {
    int nc = parent->num_children;
    Node **kids = parent->children;

    double best = -INFINITY;
    int best_idx = 0;

    for (int i = 0; i < nc; i++) {
        Node *c = kids[i];
        if (c == NULL) continue;
        double u = ucb1_atomic(c, parent);
        if (u > best) { best = u; best_idx = i; }
    }

    return best_idx;
}

// MCTS expansion phase with node children array allocated locally
void expand_parallel(Node *node) {
    GameState *state = &node->state;

    int count = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j))
                count++;

    if (count == 0) return;

    Node **new_children = malloc(count * sizeof(Node*));
    int idx = 0;

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j)) {
                GameState new_state = *state;
                make_move(&new_state, i, j);
                new_children[idx++] = create_node(&new_state, i, j, node);
            }

    node->children = new_children;
    node->num_children = count;
}

// Single MCTS iteration
MCTSTiming mcts_iteration(Node *root, unsigned int *seed) {
    MCTSTiming timing = {0.0, 0.0, 0.0, 0.0, 0.0};
    Node *node = root;

    // Selection
    double sel_start = omp_get_wtime();
    while (node->num_children > 0) {
        node = select_child(node);
    }
    double sel_end = omp_get_wtime();
    timing.selection = sel_end - sel_start;

    // Expansion
    double exp_start = omp_get_wtime();
    if (node->visits > 0 && has_valid_moves(&node->state)) {
        expand(node);
        if (node->num_children > 0) {
            node = node->children[rand_r(seed) % node->num_children];
        }
    }
    double exp_end = omp_get_wtime();
    timing.expansion = exp_end - exp_start;

    // Simulation
    double sim_start = omp_get_wtime();
    double result = simulate(&node->state, node->state.player, seed, 1);
    double sim_end = omp_get_wtime();
    timing.simulation = sim_end - sim_start;

    // Backpropagation
    double back_start = omp_get_wtime();
    backpropagate(node, result);
    double back_end = omp_get_wtime();
    timing.backpropagation = back_end - back_start;
    
    return timing;
}

// MCTS root parallel approach
MCTSTiming mcts_root_parallel(Node *root, int total_iterations) {
    MCTSTiming timing = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    if (root == NULL) return timing;

    double total_start = omp_get_wtime();

    int num_threads = omp_get_max_threads();
    int iters_per_thread = total_iterations / num_threads;

    // Create thread-local root copies
    Node **thread_roots = malloc(num_threads * sizeof(Node*));
    
    // Arrays to accumulate timing from each thread
    double *sel_times = calloc(num_threads, sizeof(double));
    double *exp_times = calloc(num_threads, sizeof(double));
    double *sim_times = calloc(num_threads, sizeof(double));
    double *back_times = calloc(num_threads, sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)tid ^ 0x9e3779b9u;

        // Each thread clones the root and works independently
        thread_roots[tid] = clone_node(root, NULL);

        // Run MCTS iterations on thread-local tree
        for (int i = 0; i < iters_per_thread; i++) {
            MCTSTiming iter_timing = mcts_iteration(thread_roots[tid], &seed);
            sel_times[tid] += iter_timing.selection;
            exp_times[tid] += iter_timing.expansion;
            sim_times[tid] += iter_timing.simulation;
            back_times[tid] += iter_timing.backpropagation;
        }
    }

    // Aggregate timing across all threads
    for (int t = 0; t < num_threads; t++) {
        timing.selection += sel_times[t];
        timing.expansion += exp_times[t];
        timing.simulation += sim_times[t];
        timing.backpropagation += back_times[t];
    }

    if (root->num_children == 0) {
        for (int t = 0; t < num_threads; t++) {
            if (thread_roots[t]->num_children > 0) {
                expand(root);
                break;
            }
        }
    }

    // Aggregate statistics from all thread-local roots
    if (root->num_children > 0) {
        for (int t = 0; t < num_threads; t++) {
            Node *thread_root = thread_roots[t];

            // Merge child statistics
            for (int i = 0; i < thread_root->num_children; i++) {
                Node *thread_child = thread_root->children[i];

                for (int j = 0; j < root->num_children; j++) {
                    Node *main_child = root->children[j];

                    if (main_child->move_row == thread_child->move_row &&
                        main_child->move_col == thread_child->move_col) {
                        main_child->visits += thread_child->visits;
                        main_child->wins += thread_child->wins;
                        break;
                    }
                }
            }
            free_tree(thread_root);
        }
    }

    free(thread_roots);
    free(sel_times);
    free(exp_times);
    free(sim_times);
    free(back_times);

    double total_end = omp_get_wtime();
    timing.total = total_end - total_start;
    
    return timing;
}

// MCTS root parallel with virtual loss approach
MCTSTiming mcts_root_parallel_virtual_loss(Node *root, int total_iterations) {
    MCTSTiming timing = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    if (root == NULL || total_iterations <= 0) return timing;

    double total_start = omp_get_wtime();

    // Track cumulative times across all threads
    double sel_time = 0.0, exp_time = 0.0, sim_time = 0.0, back_time = 0.0;

    #pragma omp parallel reduction(+:sel_time,exp_time,sim_time,back_time)
    {
        unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)omp_get_thread_num() ^ 0x9e3779b9u;

        #pragma omp for schedule(dynamic)
        for (int iter = 0; iter < total_iterations; ++iter) {
            Node *path[MAX_PATH_LEN];
            int path_len = 0;

            Node *node = root;

            // Selection phase
            double sel_start = omp_get_wtime();
            int depth = 0;
            while (depth++ < 2000) {
                if (path_len >= MAX_PATH_LEN) break;
                
                path[path_len++] = node;

                // Apply virtual loss
                #pragma omp atomic
                node->visits += 1;
                #pragma omp atomic
                node->wins -= VIRTUAL_LOSS;

                int nc;
                Node **children;
                
                nc = node->num_children;
                children = node->children;

                if (nc == 0 || children == NULL) break;

                int idx = select_child_index_parallel(node);
                if (idx < 0 || idx >= nc) break;
                
                Node *child = children[idx];
                if (child == NULL) break;
                
                node = child;
            }
            double sel_end = omp_get_wtime();
            sel_time += (sel_end - sel_start);

            // Expansion
            double exp_start = omp_get_wtime();
            if (node != NULL && has_valid_moves(&node->state)) {
                omp_set_lock(&node->lock);
                // Check if expansion still needed (another thread might have expanded)
                if (node->num_children == 0 && has_valid_moves(&node->state)) {
                    expand_parallel(node);
                }
                omp_unset_lock(&node->lock);

                // If node gained children due to expansion, pick child
                if (node->num_children > 0 && node->children != NULL) {
                    int pick = rand_r(&seed) % node->num_children;
                    Node *child = node->children[pick];
                    
                    if (child != NULL && path_len < MAX_PATH_LEN) {
                        node = child;
                        path[path_len++] = node;

                        #pragma omp atomic
                        node->visits += 1;
                        #pragma omp atomic
                        node->wins -= VIRTUAL_LOSS;
                    }
                }
            }
            double exp_end = omp_get_wtime();
            exp_time += (exp_end - exp_start);

            // Simulation
            double sim_start = omp_get_wtime();
            double result = (node != NULL) ? 
                simulate(&node->state, node->state.player, &seed, 1) : 0.5;
            double sim_end = omp_get_wtime();
            sim_time += (sim_end - sim_start);

            // Backpropagation
            double back_start = omp_get_wtime();
            if (path_len > 0) {
                int original_player = path[path_len - 1]->state.player;
                
                for (int p = 0; p < path_len; ++p) {
                    Node *n = path[p];
                    if (n == NULL) continue;

                    double add;
                    if (n->player_just_moved == original_player) 
                        add = result;
                    else 
                        add = 1.0 - result;

                    double wins_inc = VIRTUAL_LOSS + add;

                    #pragma omp atomic
                    n->wins += wins_inc;
                }
            }
            double back_end = omp_get_wtime();
            back_time += (back_end - back_start);
        } 
    }
    
    timing.selection = sel_time;
    timing.expansion = exp_time;
    timing.simulation = sim_time;
    timing.backpropagation = back_time;
    
    double total_end = omp_get_wtime();
    timing.total = total_end - total_start;
    
    return timing;
}
