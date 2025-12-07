#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "include/mcts.h"
#include "othello.h"
#include "mcts.h"

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

int get_mcts_move(GameState *state, int simulations, int *r, int *c, int use_parallel, int root_parallel) {
    Node *root = create_node(state, -1, -1, NULL);
    root->player_just_moved = opponent(state->player);
    expand(root);

    if (root->num_children == 0) {
        free_tree(root);
        return 0;
    }

    if (root_parallel) {
        mcts_root_parallel_virtual_loss(root, simulations);
    } else if (use_parallel) {
        mcts_leaf_parallel(root, simulations);
    } else {
        mcts_sequential(root, simulations);
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

// Benchmark 1: Sequential vs Parallel Performance
void benchmark_seq_vs_parallel(int mcts_sims, int num_games) {
    printf("\n=== Benchmark 1: Sequential vs Parallel (%d sims, %d games) ===\n", 
           mcts_sims, num_games);
    
    // Test Sequential
    printf("\nTesting SEQUENTIAL MCTS...\n");
    int seq_wins = 0, seq_draws = 0;
    double seq_total_time = 0.0;
    int seq_move_count = 0;
    
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
                double start = omp_get_wtime();
                if (!get_mcts_move(&state, mcts_sims, &r, &c, 0, 0)) break;
                seq_total_time += (omp_get_wtime() - start);
                seq_move_count++;
            } else {
                if (!get_random_move(&state, &r, &c)) break;
            }
            make_move(&state, r, c);
        }
        
        int winner = get_winner(&state);
        if (winner == mcts_player) seq_wins++;
        else if (winner == 0) seq_draws++;
        
        if ((game + 1) % 10 == 0) {
            printf("  Progress: %d/%d games\n", game + 1, num_games);
        }
    }
    
    // Test Parallel
    printf("\nTesting PARALLEL MCTS...\n");
    int par_wins = 0, par_draws = 0;
    double par_total_time = 0.0;
    int par_move_count = 0;
    
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
                double start = omp_get_wtime();
                if (!get_mcts_move(&state, mcts_sims, &r, &c, 0, 1)) break;
                par_total_time += (omp_get_wtime() - start);
                par_move_count++;
            } else {
                if (!get_random_move(&state, &r, &c)) break;
            }
            make_move(&state, r, c);
        }
        
        int winner = get_winner(&state);
        if (winner == mcts_player) par_wins++;
        else if (winner == 0) par_draws++;
        
        if ((game + 1) % 10 == 0) {
            printf("  Progress: %d/%d games\n", game + 1, num_games);
        }
    }
    
    // Compare Results
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║            PERFORMANCE COMPARISON              ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("\nQuality (vs Random Player):\n");
    printf("  Sequential MCTS:\n");
    printf("    Wins:  %3d (%.1f%%)\n", seq_wins, 100.0 * seq_wins / num_games);
    printf("    Draws: %3d (%.1f%%)\n", seq_draws, 100.0 * seq_draws / num_games);
    printf("    Loses: %3d (%.1f%%)\n", num_games - seq_wins - seq_draws, 
           100.0 * (num_games - seq_wins - seq_draws) / num_games);
    
    printf("  Parallel MCTS:\n");
    printf("    Wins:  %3d (%.1f%%)\n", par_wins, 100.0 * par_wins / num_games);
    printf("    Draws: %3d (%.1f%%)\n", par_draws, 100.0 * par_draws / num_games);
    printf("    Loses: %3d (%.1f%%)\n", num_games - par_wins - par_draws,
           100.0 * (num_games - par_wins - par_draws) / num_games);
    
    printf("\nSpeed Performance:\n");
    printf("  Sequential: %.4f s/move | Total: %.2f s | %d moves\n", 
           seq_total_time / seq_move_count, seq_total_time, seq_move_count);
    printf("  Parallel:   %.4f s/move | Total: %.2f s | %d moves\n",
           par_total_time / par_move_count, par_total_time, par_move_count);
    printf("  ⚡ SPEEDUP: %.2fx faster\n", seq_total_time / par_total_time);
    
    printf("\nQuality Assessment:\n");
    double quality_diff = fabs((100.0 * seq_wins / num_games) - (100.0 * par_wins / num_games));
    if (quality_diff < 5.0) {
        printf("  ✓ Win rates are similar (%.1f%% difference)\n", quality_diff);
    } else {
        printf("  ⚠ Win rates differ significantly (%.1f%% difference)\n", quality_diff);
    }
    
    printf("\nSpeed Assessment:\n");
    double speedup = seq_total_time / par_total_time;
    if (speedup > 3.0) {
        printf("  ✓ Excellent speedup (%.1fx)\n", speedup);
    } else if (speedup > 1.5) {
        printf("  ✓ Good speedup (%.1fx)\n", speedup);
    } else {
        printf("  ⚠ Limited speedup (%.1fx) - may need tuning\n", speedup);
    }
}

// Benchmark 2: Head-to-Head Sequential vs Parallel
void benchmark_head_to_head(int simulations, int num_games) {
    printf("\n=== Benchmark 2: Sequential vs Parallel Head-to-Head (%d sims, %d games) ===\n",
           simulations, num_games);
    
    int seq_wins = 0, par_wins = 0, draws = 0;
    double seq_total_time = 0.0, par_total_time = 0.0;
    
    for (int game = 0; game < num_games; game++) {
        GameState state;
        init_board(&state);
        
        // Alternate who plays first
        int seq_player = (game % 2 == 0) ? BLACK : WHITE;
        int par_player = opponent(seq_player);
        
        while (1) {
            if (!has_valid_moves(&state)) {
                state.player = opponent(state.player);
                if (!has_valid_moves(&state)) break;
            }
            
            int r, c;
            double start = omp_get_wtime();
            
            if (state.player == seq_player) {
                if (!get_mcts_move(&state, simulations, &r, &c, 0, 0)) break;
                seq_total_time += (omp_get_wtime() - start);
            } else {
                if (!get_mcts_move(&state, simulations, &r, &c, 0, 1)) break;
                par_total_time += (omp_get_wtime() - start);
            }
            
            make_move(&state, r, c);
        }
        
        int winner = get_winner(&state);
        if (winner == seq_player) seq_wins++;
        else if (winner == par_player) par_wins++;
        else draws++;
        
        if ((game + 1) % 5 == 0) {
            printf("  Progress: %d/%d games\n", game + 1, num_games);
        }
    }
    
    printf("\nResults:\n");
    printf("  Sequential Wins: %d (%.1f%%)\n", seq_wins, 100.0 * seq_wins / num_games);
    printf("  Parallel Wins:   %d (%.1f%%)\n", par_wins, 100.0 * par_wins / num_games);
    printf("  Draws:           %d (%.1f%%)\n", draws, 100.0 * draws / num_games);
    printf("\nTiming:\n");
    printf("  Sequential Avg: %.4f s/move\n", seq_total_time / (num_games * 30));
    printf("  Parallel Avg:   %.4f s/move\n", par_total_time / (num_games * 30));
    printf("  Speedup: %.2fx\n", seq_total_time / par_total_time);
}

// Benchmark 3: Thread Scaling
void benchmark_thread_scaling(int simulations, int num_games) {
    printf("\n=== Benchmark 3: Thread Scaling (%d sims, %d games) ===\n",
           simulations, num_games);
    
    int thread_counts[] = {1, 2, 4, 8};
    int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);
    double baseline_time = 0.0;
    
    for (int cfg = 0; cfg < num_configs; cfg++) {
        int threads = thread_counts[cfg];
        omp_set_num_threads(threads);
        
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
                double start = omp_get_wtime();
                if (!get_mcts_move(&state, simulations, &r, &c, 0, 1)) break;
                total_time += (omp_get_wtime() - start);
                move_count++;
                
                make_move(&state, r, c);
            }
        }
        
        double avg_time = total_time / move_count;
        if (cfg == 0) baseline_time = avg_time;
        
        double speedup = baseline_time / avg_time;
        double efficiency = (speedup / threads) * 100.0;
        
        printf("  %d thread%s: %.4f s/move | Speedup: %.2fx | Efficiency: %.1f%%\n",
               threads, threads > 1 ? "s" : " ",
               avg_time, speedup, efficiency);
    }
}

void benchmark_simulation_scaling(int num_games) {
    printf("\n=== Benchmark 4: Simulation Scaling + GPU vs Random (%d games each) ===\n", num_games);

    int sim_counts[] = {500, 1000, 2000, 5000};
    int num_configs = sizeof(sim_counts) / sizeof(sim_counts[0]);

    printf("\n%-8s | %-15s | %-20s | %-15s | %-20s | %-15s | %-20s\n",
           "Sims", "Seq Time", "Seq Wins", "CPU Time", "CPU Wins", "Root Time", "Root Wins");
    printf("---------|-----------------|--------------------|-----------------|--------------------|-----------------|--------------------\n");

    for (int cfg = 0; cfg < num_configs; cfg++) {
        int sims = sim_counts[cfg];

        double seq_time = 0.0, par_time = 0.0, gpu_time = 0.0;
        int seq_moves = 0, par_moves = 0, gpu_moves = 0;
        int seq_wins = 0, par_wins = 0, gpu_wins = 0;

        for (int game = 0; game < num_games; game++) {
            GameState state_seq, state_par, state_gpu;
            init_board(&state_seq);
            init_board(&state_par);
            init_board(&state_gpu);

            int seq_player = (game % 2 == 0) ? BLACK : WHITE;
            int par_player = seq_player;
            int gpu_player = seq_player;

            while (1) {
                // Check if game over
                if (!has_valid_moves(&state_seq) && !has_valid_moves(&state_par) && !has_valid_moves(&state_gpu)) break;

                int r, c;

                // Sequential MCTS vs Random
                if (state_seq.player == seq_player) {
                    double start = omp_get_wtime();
                    if (!get_mcts_move(&state_seq, sims, &r, &c, 0, 0)) break;
                    seq_time += (omp_get_wtime() - start);
                } else {
                    if (!get_random_move(&state_seq, &r, &c)) break;
                }
                seq_moves++;
                make_move(&state_seq, r, c);

                // Parallel CPU MCTS vs Random
                if (state_par.player == par_player) {
                    double start = omp_get_wtime();
                    if (!get_mcts_move(&state_par, sims, &r, &c, 1, 0)) break;
                    par_time += (omp_get_wtime() - start);
                } else {
                    if (!get_random_move(&state_par, &r, &c)) break;
                }
                par_moves++;
                make_move(&state_par, r, c);

                // GPU MCTS vs Random
                if (state_gpu.player == gpu_player) {
                    double start = omp_get_wtime();
                    if (!get_mcts_move(&state_gpu, sims, &r, &c, 0, 1)) break;
                    gpu_time += (omp_get_wtime() - start);
                } else {
                    if (!get_random_move(&state_gpu, &r, &c)) break;
                }
                gpu_moves++;
                make_move(&state_gpu, r, c);
            }

            // Count wins for each MCTS
            if (get_winner(&state_seq) == seq_player) seq_wins++;
            if (get_winner(&state_par) == par_player) par_wins++;
            if (get_winner(&state_gpu) == gpu_player) gpu_wins++;
        }

        printf("%-8d | %10.4f | %3d/%-15d | %10.4f | %3d/%-15d | %10.4f | %3d/%-15d\n",
               sims,
               seq_time / seq_moves, seq_wins, num_games,
               par_time / par_moves, par_wins, num_games,
               gpu_time / gpu_moves, gpu_wins, num_games);
    }
}


int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Othello MCTS: Sequential vs Parallel         ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("\nSystem Info:\n");
    printf("  Max OpenMP Threads: %d\n", omp_get_max_threads());
    
    if (argc > 1 && strcmp(argv[1], "quick") == 0) {
        printf("\n[QUICK MODE - Fast testing]\n");
        benchmark_seq_vs_parallel(1000, 20);
        benchmark_head_to_head(1000, 10);
        benchmark_thread_scaling(1000, 5);
        benchmark_simulation_scaling(20);
    } else if (argc > 1 && strcmp(argv[1], "full") == 0) {
        printf("\n[FULL MODE - Comprehensive testing]\n");
        benchmark_seq_vs_parallel(2000, 50);
        benchmark_head_to_head(2000, 30);
        benchmark_thread_scaling(2000, 15);
        benchmark_simulation_scaling(15);
    } else {
        printf("\n[STANDARD MODE]\n");
        benchmark_seq_vs_parallel(1000, 30);
        benchmark_head_to_head(1000, 20);
        benchmark_thread_scaling(1000, 10);
        benchmark_simulation_scaling(10);
    }
    
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║  Benchmark Complete!                           ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    
    return 0;
}