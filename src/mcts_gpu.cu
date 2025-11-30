#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include "mcts.h"

#define MAX_MOVES 64
#define THREADS_PER_BLOCK 256

// Directions
__device__ __constant__ int dx[8] = {-1,-1,-1,0,0,1,1,1};
__device__ __constant__ int dy[8] = {-1,0,1,-1,1,-1,0,1};

// Compact GPU state
typedef struct {
    int board[SIZE][SIZE];
    int player;
} SmallState;

// --- Device helpers ---
__device__ int opponent_dev(int player) {
    return player == BLACK ? WHITE : BLACK;
}

__device__ int is_valid_move_dev(SmallState *state, int r, int c) {
    if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) return 0;
    if (state->board[r][c] != EMPTY) return 0;
    int opp = opponent_dev(state->player);
    for (int d = 0; d < 8; d++) {
        int nr = r + dx[d], nc = c + dy[d], found = 0;
        while (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && state->board[nr][nc] == opp) {
            found = 1; nr += dx[d]; nc += dy[d];
        }
        if (found && nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && state->board[nr][nc] == state->player)
            return 1;
    }
    return 0;
}

__device__ int has_valid_moves_dev(SmallState *state) {
    for (int r = 0; r < SIZE; r++)
        for (int c = 0; c < SIZE; c++)
            if (is_valid_move_dev(state, r, c)) return 1;
    return 0;
}

__device__ void make_move_dev(SmallState *state, int r, int c) {
    int player = state->player;
    state->board[r][c] = player;
    int opp = opponent_dev(player);
    for (int d = 0; d < 8; d++) {
        int nr = r + dx[d], nc = c + dy[d], flips = 0;
        while (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && state->board[nr][nc] == opp) {
            flips++; nr += dx[d]; nc += dy[d];
        }
        if (flips > 0 && nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE && state->board[nr][nc] == player) {
            nr = r + dx[d]; nc = c + dy[d];
            for (int i = 0; i < flips; i++) {
                state->board[nr][nc] = player;
                nr += dx[d]; nc += dy[d];
            }
        }
    }
    state->player = opp;
}

__device__ void get_score_dev(SmallState *state, int *black, int *white) {
    *black = 0; *white = 0;
    for (int r = 0; r < SIZE; r++)
        for (int c = 0; c < SIZE; c++) {
            if (state->board[r][c] == BLACK) (*black)++;
            else if (state->board[r][c] == WHITE) (*white)++;
        }
}

// --- Kernel: one thread per playout ---
__global__ void playout_kernel(SmallState *leaves, int leaf_count, int playouts_per_leaf, float *results, unsigned long long seed) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_playouts = leaf_count * playouts_per_leaf;
    if (thread_id >= total_playouts) return;

    int leaf_idx = thread_id / playouts_per_leaf;

    // Initialize RNG
    curandState rng;
    curand_init(seed ^ thread_id, thread_id, 0, &rng);

    SmallState state = leaves[leaf_idx];
    int original_player = state.player;

    int black, white;

    // Run one playout
    while (1) {
        if (!has_valid_moves_dev(&state)) {
            state.player = opponent_dev(state.player);
            if (!has_valid_moves_dev(&state)) break;
        }

        int moves[MAX_MOVES][2], count = 0;
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                if (is_valid_move_dev(&state, r, c)) {
                    moves[count][0] = r;
                    moves[count][1] = c;
                    count++;
                }

        if (count == 0) break;

        int pick = curand(&rng) % count;
        make_move_dev(&state, moves[pick][0], moves[pick][1]);
    }

    get_score_dev(&state, &black, &white);

    float result = 0.0f;
    if (original_player == BLACK) result = (black > white) ? 1.0f : ((black < white) ? 0.0f : 0.5f);
    else result = (white > black) ? 1.0f : ((white < black) ? 0.0f : 0.5f);

    atomicAdd(&results[leaf_idx], result);
}

// --- Host wrapper ---
void run_playouts_on_gpu(Node **leaves, int leaf_count, int playouts_per_leaf, float *results) {
    SmallState *h_leaves = (SmallState*)malloc(sizeof(SmallState) * leaf_count);
    for (int i = 0; i < leaf_count; i++) {
        Node *leaf = leaves[i];
        for (int r = 0; r < SIZE; r++)
            for (int c = 0; c < SIZE; c++)
                h_leaves[i].board[r][c] = leaf->state.board[r][c];
        h_leaves[i].player = leaf->state.player;
    }

    SmallState *d_leaves;
    float *d_results;
    cudaMalloc(&d_leaves, sizeof(SmallState) * leaf_count);
    cudaMalloc(&d_results, sizeof(float) * leaf_count);
    cudaMemcpy(d_leaves, h_leaves, sizeof(SmallState) * leaf_count, cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, sizeof(float) * leaf_count);

    unsigned long long seed = time(NULL);
    int total_playouts = leaf_count * playouts_per_leaf;
    int blocks = (total_playouts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    playout_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_leaves, leaf_count, playouts_per_leaf, d_results, seed);
    cudaDeviceSynchronize();

    cudaMemcpy(results, d_results, sizeof(float) * leaf_count, cudaMemcpyDeviceToHost);

    // Average results per leaf
    for (int i = 0; i < leaf_count; i++)
        results[i] /= playouts_per_leaf;

    cudaFree(d_leaves);
    cudaFree(d_results);
    free(h_leaves);
}
