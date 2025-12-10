#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include "mcts.h"

typedef enum {
    MCTS_SEQUENTIAL,
    MCTS_LEAF_PARALLEL,
    MCTS_ROOT_PARALLEL,
    MCTS_ROOT_PARALLEL_VIRTUAL_LOSS
} MCTSMode;

const char* mode_names[] = {
    "Sequential",
    "Leaf Parallel",
    "Root Parallel",
    "Root Parallel + Virtual Loss"
};

typedef struct {
    int wins;
    int losses;
    int draws;
    double total_time;
    int move_count;
    MCTSTimingAggregator agg;
} ModeStats;

int get_random_move(GameState *state, int *r, int *c) {
    int board_size = SIZE * SIZE;
    int moves[board_size][2], count = 0;
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

int get_mcts_move(GameState *state, int simulations, int *r, int *c, 
                  MCTSMode mode, MCTSTiming *timing_out) {
    Node *root = create_node(state, -1, -1, NULL);
    root->player_just_moved = opponent(state->player);
    expand(root);

    if (root->num_children == 0) {
        free_tree(root);
        return 0;
    }

    MCTSTiming timing;
    switch (mode) {
        case MCTS_SEQUENTIAL:
            timing = mcts_sequential(root, simulations);
            break;
        case MCTS_LEAF_PARALLEL:
            timing = mcts_leaf_parallel(root, simulations);
            break;
        case MCTS_ROOT_PARALLEL:
            timing = mcts_root_parallel(root, simulations);
            break;
        case MCTS_ROOT_PARALLEL_VIRTUAL_LOSS:
            timing = mcts_root_parallel_virtual_loss(root, simulations);
            break;
        default:
            timing = mcts_sequential(root, simulations);
            break;
    }

    if (timing_out != NULL) {
        *timing_out = timing;
    }

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

// Benchmark 1: All modes vs random player
void benchmark_all_modes_vs_random(int mcts_sims, int num_games) {
    printf("\n=== Benchmark 1: All MCTS Modes vs Random Player (%d sims, %d games) ===\n", 
           mcts_sims, num_games);

    ModeStats stats[4];
    
    // Initialize all stats
    for (int mode = 0; mode < 4; mode++) {
        stats[mode].wins = 0;
        stats[mode].draws = 0;
        stats[mode].total_time = 0.0;
        stats[mode].move_count = 0;
        init_timing_aggregator(&stats[mode].agg);
    }
    
    // Test each mode
    for (int mode = 0; mode < 4; mode++) {
        printf("\nTesting %s MCTS...\n", mode_names[mode]);
        
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
                    MCTSTiming timing;
                    double start = omp_get_wtime();
                    int res = get_mcts_move(&state, mcts_sims, &r, &c, mode, &timing);
                    add_timing(&stats[mode].agg, &timing);
                    if (!res) break;
                    stats[mode].total_time += (omp_get_wtime() - start);
                    stats[mode].move_count++;
                } else {
                    if (!get_random_move(&state, &r, &c)) break;
                }
                make_move(&state, r, c);
            }
            
            int winner = get_winner(&state);
            if (winner == mcts_player) stats[mode].wins++;
            else if (winner == 0) stats[mode].draws++;
            
            if ((game + 1) % 10 == 0) {
                printf("  Progress: %d/%d games\n", game + 1, num_games);
            }
        }
        
        MCTSTiming avg = get_average_timing(&stats[mode].agg);
        print_timing(&avg, 1000, mode_names[mode]);
    }
    
    // Compare results
    printf("\n╔════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    PERFORMANCE COMPARISON                          ║\n");
    printf("╚════════════════════════════════════════════════════════════════════╝\n");
    
    printf("%-30s | %8s | %8s | %8s | %12s | %10s\n",
           "Mode", "Wins", "Draws", "Losses", "Win Rate", "Time/Move");
    printf("-------------------------------|----------|----------|----------|--------------|------------\n");
    
    double baseline_time = stats[0].total_time / stats[0].move_count;
    
    for (int mode = 0; mode < 4; mode++) {
        int losses = num_games - stats[mode].wins - stats[mode].draws;
        double win_rate = 100.0 * stats[mode].wins / num_games;
        double avg_time = stats[mode].total_time / stats[mode].move_count;
        
        printf("%-30s | %4d/%3d | %4d/%3d | %4d/%3d | %11f%% | %10f s\n",
               mode_names[mode],
               stats[mode].wins, num_games,
               stats[mode].draws, num_games,
               losses, num_games,
               win_rate,
               avg_time);
    }
    
    printf("\nSpeedup vs Sequential:\n");
    for (int mode = 1; mode < 4; mode++) {
        double avg_time = stats[mode].total_time / stats[mode].move_count;
        double speedup = baseline_time / avg_time;
        printf("  %s: %.2fx\n", mode_names[mode], speedup);
    }
}

// Benchmark 2: head-to-head all modes
void benchmark_head_to_head_all_modes(int simulations, int num_games) {
    printf("\n=== Benchmark 2: Head-to-Head All Modes (%d sims, %d games each matchup) ===\n",
           simulations, num_games);

    typedef struct {
        int mode1_wins;
        int mode2_wins;
        int draws;
        double mode1_total_time;
        double mode2_total_time;
        int mode1_move_count;
        int mode2_move_count;
        MCTSTimingAggregator mode1_agg;
        MCTSTimingAggregator mode2_agg;
    } MatchupStats;

    MatchupStats matchups[4][4];

    // Initialize stats
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matchups[i][j].mode1_wins = 0;
            matchups[i][j].mode2_wins = 0;
            matchups[i][j].draws = 0;
            matchups[i][j].mode1_total_time = 0.0;
            matchups[i][j].mode2_total_time = 0.0;
            matchups[i][j].mode1_move_count = 0;
            matchups[i][j].mode2_move_count = 0;
            init_timing_aggregator(&matchups[i][j].mode1_agg);
            init_timing_aggregator(&matchups[i][j].mode2_agg);
        }
    }

    // Run matchups (only upper triangle to avoid duplicates)
    for (int mode1 = 0; mode1 < 4; mode1++) {
        for (int mode2 = mode1 + 1; mode2 < 4; mode2++) {
            printf("\nTesting %s vs %s...\n", mode_names[mode1], mode_names[mode2]);

            for (int game = 0; game < num_games; game++) {
                GameState state;
                init_board(&state);

                int player1 = (game % 2 == 0) ? BLACK : WHITE;
                int player2 = opponent(player1);

                while (1) {
                    if (!has_valid_moves(&state)) {
                        state.player = opponent(state.player);
                        if (!has_valid_moves(&state)) break;
                    }

                    int r, c;
                    MCTSTiming timing;
                    double start = omp_get_wtime();

                    if (state.player == player1) {
                        // CRITICAL: Ensure threads from previous move are done
                        #pragma omp barrier
                        
                        int res = get_mcts_move(&state, simulations, &r, &c, mode1, &timing);
                        add_timing(&matchups[mode1][mode2].mode1_agg, &timing);
                        if (!res) break;
                        matchups[mode1][mode2].mode1_total_time += (omp_get_wtime() - start);
                        matchups[mode1][mode2].mode1_move_count++;
                    } else {
                        // CRITICAL: Ensure threads from previous move are done
                        #pragma omp barrier
                        
                        int res = get_mcts_move(&state, simulations, &r, &c, mode2, &timing);
                        add_timing(&matchups[mode1][mode2].mode2_agg, &timing);
                        if (!res) break;
                        matchups[mode1][mode2].mode2_total_time += (omp_get_wtime() - start);
                        matchups[mode1][mode2].mode2_move_count++;
                    }

                    make_move(&state, r, c);
                }

                int winner = get_winner(&state);
                if (winner == player1) {
                    matchups[mode1][mode2].mode1_wins++;
                } else if (winner == player2) {
                    matchups[mode1][mode2].mode2_wins++;
                } else {
                    matchups[mode1][mode2].draws++;
                }
            }

            printf("  %s: %d-%d-%d (W-L-D)\n",
                   mode_names[mode1],
                   matchups[mode1][mode2].mode1_wins,
                   matchups[mode1][mode2].mode2_wins,
                   matchups[mode1][mode2].draws);
            printf("  %s: %d-%d-%d (W-L-D)\n",
                   mode_names[mode2],
                   matchups[mode1][mode2].mode2_wins,
                   matchups[mode1][mode2].mode1_wins,
                   matchups[mode1][mode2].draws);
        }
    }

    printf("\n╔════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    HEAD-TO-HEAD RESULTS                            ║\n");
    printf("╚════════════════════════════════════════════════════════════════════╝\n");

    for (int mode1 = 0; mode1 < 4; mode1++) {
        for (int mode2 = mode1 + 1; mode2 < 4; mode2++) {
            printf("\n%s vs %s:\n", mode_names[mode1], mode_names[mode2]);
            printf("  %s: %d wins, %d losses, %d draws (%.1f%% win rate)\n",
                   mode_names[mode1],
                   matchups[mode1][mode2].mode1_wins,
                   matchups[mode1][mode2].mode2_wins,
                   matchups[mode1][mode2].draws,
                   100.0 * matchups[mode1][mode2].mode1_wins / num_games);
            printf("  %s: %d wins, %d losses, %d draws (%.1f%% win rate)\n",
                   mode_names[mode2],
                   matchups[mode1][mode2].mode2_wins,
                   matchups[mode1][mode2].mode1_wins,
                   matchups[mode1][mode2].draws,
                   100.0 * matchups[mode1][mode2].mode2_wins / num_games);

            if (matchups[mode1][mode2].mode1_move_count > 0) {
                printf("  %s avg time: %.4f s/move\n",
                       mode_names[mode1],
                       matchups[mode1][mode2].mode1_total_time / matchups[mode1][mode2].mode1_move_count);
            }
            if (matchups[mode1][mode2].mode2_move_count > 0) {
                printf("  %s avg time: %.4f s/move\n",
                       mode_names[mode2],
                       matchups[mode1][mode2].mode2_total_time / matchups[mode1][mode2].mode2_move_count);
            }
        }
    }
}

// Benchmark 3: thread scaling for parallel modes
void benchmark_thread_scaling(int simulations, int num_games) {
    printf("\n=== Benchmark 3: Thread Scaling for Parallel Modes (%d sims, %d games) ===\n",
           simulations, num_games);
    
    int thread_counts[] = {1, 2, 4, 8};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    // Test each parallel mode
    for (int mode = 1; mode < 4; mode++) {
        printf("\n%s:\n", mode_names[mode]);
        printf("%-8s | %-12s | %-10s | %-12s\n", "Threads", "Time/Move", "Speedup", "Efficiency");
        printf("---------|--------------|------------|-------------\n");
        
        double baseline_time = 0.0;
        
        for (int cfg = 0; cfg < num_configs; cfg++) {
            int threads = thread_counts[cfg];
            omp_set_num_threads(threads);
            
            MCTSTimingAggregator agg;
            init_timing_aggregator(&agg);
            
            double total_time = 0.0;
            int move_count = 0;
            
            for (int game = 0; game < num_games; game++) {
                GameState state;
                init_board(&state);
                
                while (1) {
                    if (!has_valid_moves(&state)) {
                        state.player = opponent(state.player);
                        if (!has_valid_moves(&state)) break;
                    }
                    
                    int r, c;
                    MCTSTiming timing;
                    double start = omp_get_wtime();
                    int res = get_mcts_move(&state, simulations, &r, &c, mode, &timing);
                    add_timing(&agg, &timing);
                    if (!res) break;
                    total_time += (omp_get_wtime() - start);
                    move_count++;
                    
                    make_move(&state, r, c);
                }
            }
            
            double avg_time = total_time / move_count;
            if (cfg == 0) baseline_time = avg_time;
            
            double speedup = baseline_time / avg_time;
            double efficiency = (speedup / threads) * 100.0;
            
            printf("%-8d | %10.4f s | %8.2fx | %10.1f%%\n",
                   threads, avg_time, speedup, efficiency);
        }
    }
}

// Benchmark 4: simulation scaling for all modes
void benchmark_simulation_scaling(int num_games) {
    printf("\n=== Benchmark 4: Simulation Scaling All Modes (%d games each) ===\n", num_games);

    int sim_counts[] = {500, 1000, 2000, 5000};
    int num_configs = sizeof(sim_counts) / sizeof(sim_counts[0]);
    
    printf("\n%-8s | ", "Sims");
    for (int mode = 0; mode < 4; mode++) {
        printf("%-18s | ", mode_names[mode]);
    }
    printf("\n");
    
    printf("---------|");
    for (int mode = 0; mode < 4; mode++) {
        printf("--------------------|");
    }
    printf("\n");
    
    printf("%-8s | ", "");
    for (int mode = 0; mode < 4; mode++) {
        printf("%-8s | %-7s | ", "Time", "Wins");
    }
    printf("\n");

    for (int cfg = 0; cfg < num_configs; cfg++) {
        int sims = sim_counts[cfg];
        
        typedef struct {
            double time;
            int moves;
            int wins;
            MCTSTimingAggregator agg;
        } ModeResult;
        
        ModeResult results[4];
        for (int mode = 0; mode < 4; mode++) {
            results[mode].time = 0.0;
            results[mode].moves = 0;
            results[mode].wins = 0;
            init_timing_aggregator(&results[mode].agg);
        }
        
        // Test each mode
        for (int mode = 0; mode < 4; mode++) {
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
                        MCTSTiming timing;
                        double start = omp_get_wtime();
                        int res = get_mcts_move(&state, sims, &r, &c, mode, &timing);
                        add_timing(&results[mode].agg, &timing);
                        if (!res) break;
                        results[mode].time += (omp_get_wtime() - start);
                        results[mode].moves++;
                    } else {
                        if (!get_random_move(&state, &r, &c)) break;
                    }
                    make_move(&state, r, c);
                }
                
                int winner = get_winner(&state);
                if (winner == mcts_player) results[mode].wins++;
            }
        }
        
        printf("%-8d | ", sims);
        for (int mode = 0; mode < 4; mode++) {
            double avg_time = results[mode].time / results[mode].moves;
            printf("%7.4fs | %3d/%3d | ", avg_time, results[mode].wins, num_games);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Othello MCTS: All Modes Comparison            ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("\nModes tested:\n");
    for (int i = 0; i < 4; i++) {
        printf("  %d. %s\n", i, mode_names[i]);
    }
    
    if (argc > 1 && strcmp(argv[1], "quick") == 0) {
        printf("\n[QUICK MODE - Fast testing]\n");
        benchmark_all_modes_vs_random(1000, 20);
        benchmark_head_to_head_all_modes(1000, 10);
        benchmark_thread_scaling(1000, 5);
        benchmark_simulation_scaling(10);
    } else if (argc > 1 && strcmp(argv[1], "full") == 0) {
        printf("\n[FULL MODE - Comprehensive testing]\n");
        benchmark_all_modes_vs_random(2000, 50);
        benchmark_head_to_head_all_modes(2000, 30);
        benchmark_thread_scaling(2000, 15);
        benchmark_simulation_scaling(30);
    } else {
        printf("\n[STANDARD MODE]\n");
        benchmark_all_modes_vs_random(1000, 30);
        benchmark_head_to_head_all_modes(1000, 20);
        benchmark_thread_scaling(1000, 10);
        benchmark_simulation_scaling(15);
    }
    
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║  Benchmark Complete!                           ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    
    return 0;
}