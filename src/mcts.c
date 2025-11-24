#include "mcts.h"

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

double simulate(GameState *state, int original_player) {
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
        int idx = rand() % count;
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
        double result = simulate(&node->state, node->state.player);
        
        // Backpropagation
        backpropagate(node, result);
    }
}