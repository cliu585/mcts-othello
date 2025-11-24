#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "othello.h"
#include "mcts.h"

void free_tree(Node *node) {
    if (node == NULL) return;
    for (int i = 0; i < node->num_children; i++)
        free_tree(node->children[i]);
    free(node->children);
    free(node);
}

int get_random_move(GameState *state, int *r, int *c) {
    int moves[64][2], count = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (is_valid_move(state, i, j)) {
                moves[count][0] = i;
                moves[count][1] = j;
                count++;
            }
    
    if (count == 0) return 0;
    int idx = rand() % count;
    *r = moves[idx][0];
    *c = moves[idx][1];
    return 1;
}

int get_mcts_move(GameState *state, int simulations, int *r, int *c) {
    Node *root = create_node(state, -1, -1, NULL);
    root->player_just_moved = opponent(state->player);
    expand(root);
    
    if (root->num_children == 0) {
        free_tree(root);
        return 0;
    }
    
    mcts(root, simulations);
    
    // Choose based on HIGHEST WIN RATE, not just most visits
    Node *best = NULL;
    double best_winrate = -1.0;
    for (int i = 0; i < root->num_children; i++) {
        if (root->children[i]->visits > 0) {
            double winrate = root->children[i]->wins / root->children[i]->visits;
            if (winrate > best_winrate) {
                best_winrate = winrate;
                best = root->children[i];
            }
        }
    }
    
    if (best) {
        *r = best->move_row;
        *c = best->move_col;
    }
    
    free_tree(root);
    return best != NULL;
}

// Benchmark 1: MCTS vs Random Player
void benchmark_vs_random(int mcts_sims, int num_games) {
    printf("\n=== Benchmark 1: MCTS(%d) vs Random (%d games) ===\n", mcts_sims, num_games);
    
    int mcts_wins = 0, random_wins = 0, draws = 0;
    int total_mcts_score = 0, total_random_score = 0;
    
    for (int game = 0; game < num_games; game++) {
        GameState state;
        init_board(&state);
        
        int mcts_player = (game % 2 == 0) ? BLACK : WHITE;
        
        while (1) {
            if (!has_valid_moves(&state)) {
                state.player = opponent(state.player);
                if (!has_valid_moves(&state)) break;
            }
            
            int r, c;
            if (state.player == mcts_player) {
                if (!get_mcts_move(&state, mcts_sims, &r, &c)) break;
            } else {
                if (!get_random_move(&state, &r, &c)) break;
            }
            make_move(&state, r, c);
        }
        
        int black_score, white_score;
        get_score(&state, &black_score, &white_score);
        
        int mcts_score = (mcts_player == BLACK) ? black_score : white_score;
        int random_score = (mcts_player == BLACK) ? white_score : black_score;
        
        total_mcts_score += mcts_score;
        total_random_score += random_score;
        
        int winner = get_winner(&state);
        if (winner == mcts_player) mcts_wins++;
        else if (winner == opponent(mcts_player)) random_wins++;
        else draws++;
        
        if ((game + 1) % 10 == 0) {
            printf("  Progress: %d/%d games completed\n", game + 1, num_games);
        }
    }
    
    printf("\nResults:\n");
    printf("  MCTS Wins:   %d (%.1f%%)\n", mcts_wins, 100.0 * mcts_wins / num_games);
    printf("  Random Wins: %d (%.1f%%)\n", random_wins, 100.0 * random_wins / num_games);
    printf("  Draws:       %d (%.1f%%)\n", draws, 100.0 * draws / num_games);
    printf("  Avg MCTS Score:   %.1f\n", (double)total_mcts_score / num_games);
    printf("  Avg Random Score: %.1f\n", (double)total_random_score / num_games);
}

// Benchmark 2: MCTS vs Random, Scaling with Number of Simulations
void benchmark_simulation_scaling(int num_games) {
    printf("\n=== Benchmark 2: Simulation Count Scaling (%d games each) ===\n",
           num_games);

    int sim_counts[] = {100, 500, 1000, 2000, 5000};
    int num_configs = sizeof(sim_counts) / sizeof(sim_counts[0]);

    for (int cfg = 0; cfg < num_configs; cfg++) {
        int sims = sim_counts[cfg];
        int wins = 0;
        int draws = 0;
        clock_t total_time = 0;

        for (int game = 0; game < num_games; game++) {
            GameState state;
            init_board(&state);

            int mcts_player = BLACK;  // ALWAYS MCTS as Black
            int r, c;

            while (1) {
                if (!has_valid_moves(&state)) {
                    state.player = opponent(state.player);
                    if (!has_valid_moves(&state)) break;
                }

                clock_t start = clock();

                if (state.player == mcts_player) {
                    if (!get_mcts_move(&state, sims, &r, &c)) break;
                } else {
                    if (!get_random_move(&state, &r, &c)) break;
                }

                total_time += (clock() - start);
                make_move(&state, r, c);
            }

            int winner = get_winner(&state);
            if (winner == mcts_player) wins++;
            else if (winner == 0) draws++;
        }

        double avg_time = (double)total_time / CLOCKS_PER_SEC / num_games;
        printf("  %5d sims: Win rate vs random = %.1f%%, Draws = %.1f%%, Avg time/game = %.3fs\n",
               sims,
               100.0 * wins / num_games,
               100.0 * draws / num_games,
               avg_time);
    }
}


// Benchmark 3: MCTS vs MCTS with different strengths
void benchmark_mcts_vs_mcts(int weak_sims, int strong_sims, int num_games) {
    printf("\n=== Benchmark 3: MCTS(%d) vs MCTS(%d) (%d games) ===\n", 
           weak_sims, strong_sims, num_games);
    
    int weak_wins = 0, strong_wins = 0, draws = 0;
    
    for (int game = 0; game < num_games; game++) {
        GameState state;
        init_board(&state);
        
        int weak_player = (game % 2 == 0) ? BLACK : WHITE;
        
        while (1) {
            if (!has_valid_moves(&state)) {
                state.player = opponent(state.player);
                if (!has_valid_moves(&state)) break;
            }
            
            int r, c;
            int sims = (state.player == weak_player) ? weak_sims : strong_sims;
            if (!get_mcts_move(&state, sims, &r, &c)) break;
            make_move(&state, r, c);
        }
        
        int winner = get_winner(&state);
        if (winner == weak_player) weak_wins++;
        else if (winner == opponent(weak_player)) strong_wins++;
        else draws++;
        
        if ((game + 1) % 5 == 0) {
            printf("  Progress: %d/%d games completed\n", game + 1, num_games);
        }
    }
    
    printf("\nResults:\n");
    printf("  Weak MCTS Wins:   %d (%.1f%%)\n", weak_wins, 100.0 * weak_wins / num_games);
    printf("  Strong MCTS Wins: %d (%.1f%%)\n", strong_wins, 100.0 * strong_wins / num_games);
    printf("  Draws:            %d (%.1f%%)\n", draws, 100.0 * draws / num_games);
}

// Benchmark 4: Move quality consistency
void benchmark_move_consistency(int simulations, int trials) {
    printf("\n=== Benchmark 4: Move Consistency (%d trials with %d sims) ===\n",
           trials, simulations);
    
    GameState state;
    init_board(&state);
    
    // Make a few moves to get to mid-game
    int setup_moves[][2] = {{2,3}, {2,2}, {2,4}};
    for (int i = 0; i < 3; i++) {
        if (is_valid_move(&state, setup_moves[i][0], setup_moves[i][1])) {
            make_move(&state, setup_moves[i][0], setup_moves[i][1]);
        }
    }
    
    int move_counts[SIZE][SIZE] = {0};
    
    for (int trial = 0; trial < trials; trial++) {
        int r, c;
        if (get_mcts_move(&state, simulations, &r, &c)) {
            move_counts[r][c]++;
        }
    }
    
    printf("\nMove distribution (should heavily favor best move):\n");
    int max_count = 0, total = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (move_counts[i][j] > max_count) max_count = move_counts[i][j];
            total += move_counts[i][j];
        }
    }
    
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (move_counts[i][j] > 0) {
                printf("  Move (%d,%d): %3d times (%.1f%%)\n", 
                       i, j, move_counts[i][j], 100.0 * move_counts[i][j] / total);
            }
        }
    }
    
    double consistency = 100.0 * max_count / total;
    printf("\nConsistency score: %.1f%% (higher is better)\n", consistency);
    if (consistency > 80) printf("  ✓ Excellent - AI is very confident\n");
    else if (consistency > 60) printf("  ✓ Good - AI has clear preference\n");
    else if (consistency > 40) printf("  ~ Fair - Some variation in choices\n");
    else printf("  ✗ Poor - AI is too random\n");
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║     Othello MCTS Benchmark Suite              ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    
    if (argc > 1 && strcmp(argv[1], "quick") == 0) {
        printf("\n[QUICK MODE - Reduced test counts]\n");
        benchmark_vs_random(1000, 20);
        benchmark_simulation_scaling(10);
        benchmark_mcts_vs_mcts(500, 2000, 10);
        benchmark_move_consistency(1000, 20);
    } else if (argc > 1 && strcmp(argv[1], "full") == 0) {
        printf("\n[FULL MODE - Comprehensive testing]\n");
        benchmark_vs_random(1000, 100);
        benchmark_vs_random(5000, 50);
        benchmark_simulation_scaling(20);
        benchmark_mcts_vs_mcts(500, 2000, 20);
        benchmark_mcts_vs_mcts(1000, 5000, 20);
        benchmark_move_consistency(5000, 50);
    } else {
        printf("\n[STANDARD MODE]\n");
        benchmark_vs_random(1000, 50);
        benchmark_simulation_scaling(15);
        benchmark_mcts_vs_mcts(500, 2000, 15);
        benchmark_move_consistency(1000, 30);
    }
    
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║     Benchmark Complete!                        ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("\nInterpretation Guide:\n");
    printf("• MCTS should beat random >80%% (shows it's learning)\n");
    printf("• More simulations = higher win rate (shows scaling)\n");
    printf("• Strong MCTS should dominate weak MCTS\n");
    printf("• High consistency = confident, quality decisions\n");
    
    return 0;
}