#include <omp.h>
#include <stdint.h>
#include "mcts_root.h"

// Deep copy a node and its entire subtree
Node* clone_node(Node *original, Node *new_parent) {
    if (original == NULL) return NULL;

    Node *clone = malloc(sizeof(Node));
    clone->state = original->state;
    clone->move_row = original->move_row;
    clone->move_col = original->move_col;
    clone->visits = 0;  // Start fresh for each thread
    clone->wins = 0.0;
    clone->parent = new_parent;
    clone->player_just_moved = original->player_just_moved;
    clone->num_children = 0;
    clone->children = NULL;

    return clone;
}


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

int select_child_index_parallel(Node *parent)
{
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

// Standard MCTS iteration 
void mcts_iteration(Node *root, unsigned int *seed) {
    Node *node = root;

    // Selection
    while (node->num_children > 0) {
        node = select_child(node);
    }

    // Expansion
    if (node->visits > 0 && has_valid_moves(&node->state)) {
        expand(node);
        if (node->num_children > 0) {
            node = node->children[rand_r(seed) % node->num_children];
        }
    }

    // Simulation
    double result = simulate(&node->state, node->state.player, seed, 1);

    // Backpropagation
    backpropagate(node, result);
}

void expand_parallel(Node *node) {
    GameState *state = &node->state;

    int count = 0;
    // First pass: count moves
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j))
                count++;

    if (count == 0) return;

    // Allocate children array locally
    Node **new_children = malloc(count * sizeof(Node*));
    int idx = 0;

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j)) {
                GameState new_state = *state;
                make_move(&new_state, i, j);
                new_children[idx++] =
                    create_node(&new_state, i, j, node);
            }

    // Now publish them atomically under the struct lock
    node->children = new_children;
    node->num_children = count;
}

// Root parallelization: Each thread builds its own independent tree from root,
// then we merge statistics to select the best move
void mcts_root_parallel(Node *root, int total_iterations) {
    if (root == NULL) return;

    int num_threads = omp_get_max_threads();
    int iters_per_thread = total_iterations / num_threads;

    // Create thread-local root copies
    Node **thread_roots = malloc(num_threads * sizeof(Node*));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)tid ^ 0x9e3779b9u;

        // Each thread clones the root and works independently
        thread_roots[tid] = clone_node(root, NULL);

        // Run MCTS iterations on thread-local tree
        for (int i = 0; i < iters_per_thread; i++) {
            mcts_iteration(thread_roots[tid], &seed);
        }
    }

    // Merge results: aggregate statistics from all thread-local roots
    for (int t = 0; t < num_threads; t++) {
        Node *thread_root = thread_roots[t];

        // If this is the first expansion, expand the main root
        if (root->num_children == 0 && thread_root->num_children > 0) {
            expand(root);
        }

        // Merge child statistics
        for (int i = 0; i < thread_root->num_children; i++) {
            Node *thread_child = thread_root->children[i];

            // Find corresponding child in main root
            for (int j = 0; j < root->num_children; j++) {
                Node *main_child = root->children[j];

                if (main_child->move_row == thread_child->move_row &&
                    main_child->move_col == thread_child->move_col) {
                    // Aggregate statistics
                    main_child->visits += thread_child->visits;
                    main_child->wins += thread_child->wins;
                    break;
                }
            }
        }
        free_tree(thread_root);
    }

    free(thread_roots);
}

void mcts_root_parallel_virtual_loss(Node *root, int total_iterations) {
    if (root == NULL || total_iterations <= 0) return;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)omp_get_thread_num() ^ 0x9e3779b9u;

        #pragma omp for schedule(dynamic)
        for (int iter = 0; iter < total_iterations; ++iter) {
            Node *path[MAX_PATH_LEN];
            int path_len = 0;

            Node *node = root;

            int depth = 0;
            while (1) {
                if (depth++ > 2000) { printf("Stuck!\n"); break; }
                if (path_len < MAX_PATH_LEN) path[path_len++] = node;
                else break; // path overflow (shouldn't happen in normal games)

                // Apply virtual loss to this node (atomically)
                #pragma omp atomic
                node->visits += 1;
                #pragma omp atomic
                node->wins -= VIRTUAL_LOSS;

                int nc = node->num_children;
                Node **children_snapshot = node->children;

                if (nc == 0 || children_snapshot == NULL) break;

                // Choose best child index using snapshot statistics
                int idx = select_child_index_parallel(node);
                if (idx < 0 || idx >= nc) {
                    printf("BAD INDEX: idx=%d nc=%d\n", idx, nc);
                    break; 
                }
                Node *child = children_snapshot[idx];
                if (!child) {
                    printf("NULL CHILD at idx=%d\n", idx);
                    break;
                }
                node = child;

                if (node == NULL) break;
            }

            if (has_valid_moves(&node->state)) {
                omp_set_lock(&node->lock);

                // Double-check if expansion still needed (another thread might have expanded)
                if (node->num_children == 0 && has_valid_moves(&node->state)) {
                    expand_parallel(node);
                }
                omp_unset_lock(&node->lock);

                // If node gained children due to expansion, pick one child to continue (and apply virtual loss to it)
                if (node->num_children > 0 && node->children != NULL) {
                    int pick = rand_r(&seed) % node->num_children;
                    node = node->children[pick];

                    if (path_len < MAX_PATH_LEN) path[path_len++] = node;

                    // virtual loss on newly chosen child
                    #pragma omp atomic
                    node->visits += 1;
                    #pragma omp atomic
                    node->wins -= VIRTUAL_LOSS;
                }
            }

            double result = simulate(&node->state, node->state.player, &seed, 1);

            int original_player = path[path_len - 1]->state.player; 
            for (int p = 0; p < path_len; ++p) {
                Node *n = path[p];

                // compute contribution depending on who just moved
                double add;
                if (n->player_just_moved == original_player) add = result;
                else add = 1.0 - result;

                double wins_inc = VIRTUAL_LOSS + add;

                #pragma omp atomic
                n->wins += wins_inc;
            }
        } 
    } 
}
