#include <omp.h>
#include <stdint.h>
#include "mcts.h"

#define ROLLOUTS 20

double ucb1(Node *node) {
    if (node->visits == 0) return INFINITY;
    double exploitation = node->wins / node->visits;
    double exploration = UCB_CONSTANT * sqrt(log(node->parent->visits) / node->visits);
    return exploitation + exploration;
}

Node* select_child(Node *node) {
    Node *best = NULL;
    double best_ucb = -INFINITY;
    
    for (int i = 0; i < node->num_children; i++) {
        double ucb = ucb1(node->children[i]);
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best = node->children[i];
        }
    }
    return best;
}

void expand(Node *node) {
    GameState *state = &node->state;
    int count = 0;
    
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j)) count++;
    
    if (count == 0) return;
    
    node->children = malloc(count * sizeof(Node*));
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (is_valid_move(state, i, j)) {
                GameState new_state = *state;
                make_move(&new_state, i, j);
                node->children[node->num_children++] = create_node(&new_state, i, j, node);
            }
        }
    }
}


double simulate(GameState *state, int original_player, unsigned int *seed, int include_seed) {
    GameState sim = *state;
    
    while (1) {
        if (!has_valid_moves(&sim)) {
            sim.player = opponent(sim.player);
            if (!has_valid_moves(&sim)) break;
        }
        
        int moves[64][2], count = 0;
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                if (is_valid_move(&sim, i, j)) {
                    moves[count][0] = i;
                    moves[count][1] = j;
                    count++;
                }
        
        if (count == 0) break;
        int idx;
        if (include_seed)
            idx = rand_r(seed) % count;
        else    
            idx = rand() % count;
        make_move(&sim, moves[idx][0], moves[idx][1]);
    }
    
    int black, white;
    get_score(&sim, &black, &white);
    
    // Return win value from perspective of original_player
    if (original_player == BLACK) {
        if (black > white) return 1.0;
        else if (black < white) return 0.0;
        else return 0.5;
    } else {
        if (white > black) return 1.0;
        else if (white < black) return 0.0;
        else return 0.5;
    }
}

/*void backpropagate_rollouts(Node *node, double result, int visits) {
    // result is total wins from perspective of the player at the LEAF where simulation started
    // We need to propagate this up, flipping perspective at each level
    
    while (node != NULL) {
        node->visits += visits;
        node->wins += result;
        
        // Flip the result for the parent (opponent's perspective)
        result = visits - result;
        
        node = node->parent;
    }
}*/

void backpropagate_rollouts(Node *node, double total_result, int rollouts, int original_player) {
    if (node == NULL) return;
    double wins_from_leaf_player = total_result;      // wins as seen from the leaf's player (sum of 0..1)
    int r = rollouts;

    // Walk up the tree, flipping perspective at each step.
    while (node != NULL) {
        node->visits += r;

        if (node->player_just_moved == original_player) {
            node->wins += wins_from_leaf_player;
        } else {
            node->wins += (double)r - wins_from_leaf_player;
        }

        // Flip perspective for the parent node:
        wins_from_leaf_player = (double)r - wins_from_leaf_player;
        node = node->parent;
    }
}

void backpropagate(Node *node, double result) {
    // Start from the leaf where simulation was run
    // result is from perspective of state.player at that node
    int sim_player = node->state.player;
    
    while (node != NULL) {
        node->visits++;
        
        // Each node stores stats from perspective of player_just_moved
        // (the player who created this node by making a move)
        if (node->player_just_moved == sim_player) {
            node->wins += result;
        } else {
            node->wins += (1.0 - result);
        }
        
        node = node->parent;
    }
}

// Standard MCTS iteration (same as your sequential version)
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

void mcts(Node *root, int iterations) {
    for (int i = 0; i < iterations; i++) {
        Node *node = root;

        // Selection
        while (node->num_children > 0) {
            node = select_child(node);
        }
        
        // Expansion
        if (node->visits > 0 && has_valid_moves(&node->state)) {
            expand(node);
            if (node->num_children > 0) {
                node = node->children[rand() % node->num_children];
            }
        }
        
        // Simulation - the result is from the perspective of the current player
        double result = simulate(&node->state, node->state.player, 0, 0);
        
        // Backpropagation
        backpropagate(node, result);
    }
}

void mcts_leaf_parallel(Node *root, int iterations) {
    if (root == NULL) return;

    int groups = iterations / ROLLOUTS;
    if (groups == 0) groups = 1;

    for (int g = 0; g < groups; g++) {
        Node *node = root;

        // Selection - single-threaded (same as sequential)
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

            double result = simulate(&state_copy, original_player, &thread_seed, 1); // result in [0,1] w.r.t. original_player
            Node *n = node;
            while (n != NULL) {
                double add;
                if (n->player_just_moved == original_player) {
                    add = result;           // wins for player_just_moved
                } else {
                    add = 1.0 - result;     // opponent's wins
                }

                // Atomic updates to avoid races (int visits, double wins)
                #pragma omp atomic
                n->visits += 1;

                #pragma omp atomic
                n->wins += add;

                n = n->parent;
            }
        } 
    } 
}

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

int select_child_index_uct(Node *parent)
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
                else break; // path overflow -> break (shouldn't happen in normal games)

                // Apply virtual loss to this node (atomically)
                #pragma omp atomic
                node->visits += 1;
                #pragma omp atomic
                node->wins -= VIRTUAL_LOSS;

                int nc = node->num_children;
                Node **children_snapshot = node->children;

                if (nc == 0 || children_snapshot == NULL) break;

                // Choose best child index (UCT) using snapshot statistics
                int idx = select_child_index_uct(node);
                if (idx < 0 || idx >= nc) {
                    printf("BAD UCT INDEX: idx=%d nc=%d\n", idx, nc);
                    break; // force leaf
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

            int original_player = path[path_len - 1]->state.player; // the player at the leaf (as your original)
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
        } // end for iter
    } // end parallel
}

// Helper function to free a tree
void free_tree(Node *node) {
    if (node == NULL) return;
    for (int i = 0; i < node->num_children; i++)
        free_tree(node->children[i]);
    free(node->children);
    free(node);
}