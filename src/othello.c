#include "othello.h"

int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

int is_valid(int r, int c) {
    return r >= 0 && r < SIZE && c >= 0 && c < SIZE;
}

int opponent(int player) {
    return player == BLACK ? WHITE : BLACK;
}

void init_board(GameState *state) {
    memset(state->board, EMPTY, sizeof(state->board));
    state->board[3][3] = WHITE;
    state->board[3][4] = BLACK;
    state->board[4][3] = BLACK;
    state->board[4][4] = WHITE;
    state->player = BLACK;
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

int has_valid_moves(GameState *state) {
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j))
                return 1;
    return 0;
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

Node* create_node(GameState *state, int r, int c, Node *parent) {
    Node *node = malloc(sizeof(Node));
    node->state = *state;
    node->move_row = r;
    node->move_col = c;
    node->visits = 0;
    node->wins = 0.0;
    node->parent = parent;
    node->children = NULL;
    node->num_children = 0;
    // Store which player made the move to get to this state
    node->player_just_moved = parent ? opponent(state->player) : BLACK;
    return node;
}