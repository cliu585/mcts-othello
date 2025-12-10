#include "othello.h"

int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

void init_board(GameState *state) {
    memset(state->board, EMPTY, sizeof(state->board));
    state->board[3][3] = WHITE;
    state->board[3][4] = BLACK;
    state->board[4][3] = BLACK;
    state->board[4][4] = WHITE;
    state->player = BLACK;
}

int is_valid(int r, int c) {
    return r >= 0 && r < SIZE && c >= 0 && c < SIZE;
}

int opponent(int player) {
    return player == BLACK ? WHITE : BLACK;
}

int is_valid_move(GameState *state, int r, int c) {
    if (!is_valid(r, c) || state->board[r][c] != EMPTY)
        return 0;
    
    int opp = opponent(state->player);
    for (int d = 0; d < 8; d++) {
        int nr = r + dx[d], nc = c + dy[d];
        int found_opp = 0;
        
        while (is_valid(nr, nc) && state->board[nr][nc] == opp) {
            found_opp = 1;
            nr += dx[d];
            nc += dy[d];
        }
        
        if (found_opp && is_valid(nr, nc) && state->board[nr][nc] == state->player)
            return 1;
    }
    return 0;
}

int has_valid_moves(GameState *state) {
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j))
                return 1;
    return 0;
}

void make_move(GameState *state, int r, int c) {
    state->board[r][c] = state->player;
    int opp = opponent(state->player);
    
    for (int d = 0; d < 8; d++) {
        int nr = r + dx[d], nc = c + dy[d];
        int flips = 0;
        
        while (is_valid(nr, nc) && state->board[nr][nc] == opp) {
            flips++;
            nr += dx[d];
            nc += dy[d];
        }
        
        if (flips > 0 && is_valid(nr, nc) && state->board[nr][nc] == state->player) {
            nr = r + dx[d];
            nc = c + dy[d];
            for (int i = 0; i < flips; i++) {
                state->board[nr][nc] = state->player;
                nr += dx[d];
                nc += dy[d];
            }
        }
    }
    state->player = opp;
}

void get_score(GameState *state, int *black, int *white) {
    *black = 0;
    *white = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (state->board[i][j] == BLACK) (*black)++;
            else if (state->board[i][j] == WHITE) (*white)++;
        }
    }
}

int get_winner(GameState *state) {
    int black, white;
    get_score(state, &black, &white);
    return black > white ? BLACK : (white > black ? WHITE : 0);
}

GameState* clone_game_state(const GameState* original) {
    if (original == NULL) {
        return NULL;
    }
    
    GameState* clone = (GameState*)malloc(sizeof(GameState));
    if (clone == NULL) {
        return NULL;  // Memory allocation failed
    }
    
    memcpy(clone->board, original->board, sizeof(original->board));
    clone->player = original->player;
    
    return clone;
}