#ifndef OTHELLO_H
#define OTHELLO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>   // add for parallelism


#define SIZE 8
#define EMPTY 0
#define BLACK 1
#define WHITE 2
#define UCB_CONSTANT 1.414

typedef struct {
    int board[SIZE][SIZE];
    int player;
} GameState;

typedef struct Node {
    GameState state;
    int move_row, move_col;
    int visits;
    double wins;  // Changed to double for better accuracy
    struct Node *parent;
    struct Node **children;
    int num_children;
    int player_just_moved;  // Track which player made this move
    omp_lock_t lock;  // ← ADD THIS LINE FOR PARALLEL VERSION
} Node;

int is_valid(int r, int c);

int opponent(int player);

void init_board(GameState *state);

int is_valid_move(GameState *state, int r, int c);

void make_move(GameState *state, int r, int c);

int has_valid_moves(GameState *state);

void get_score(GameState *state, int *black, int *white);

int get_winner(GameState *state);

Node* create_node(GameState *state, int r, int c, Node *parent);
#endif