#ifndef OTHELLO_H
#define OTHELLO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define SIZE 8
#define EMPTY 0
#define BLACK 1
#define WHITE 2

typedef struct {
    int board[SIZE][SIZE];
    int player;
} GameState;

void init_board(GameState *state);
int is_valid(int r, int c);
int opponent(int player);
int is_valid_move(GameState *state, int r, int c);
int has_valid_moves(GameState *state);
void make_move(GameState *state, int r, int c);
void get_score(GameState *state, int *black, int *white);
int get_winner(GameState *state);
GameState* clone_game_state(const GameState* original);

#endif