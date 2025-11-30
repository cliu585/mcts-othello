#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "othello.h"
#include "mcts.h"

// Timing utilities
typedef struct {
    clock_t start;
    clock_t end;
    double elapsed;
} Timer;

void timer_start(Timer *t) {
    t->start = clock();
}

void timer_stop(Timer *t) {
    t->end = clock();
    t->elapsed = (double)(t->end - t->start) / CLOCKS_PER_SEC;
}

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

int get_mcts_move_timed(GameState *state, int simulations, int *r, int *c, 
                        int use_gpu, double *time_taken) {
    Timer timer;
    timer_start(&timer);
    
    Node *root = create_node(state, -1, -1, NULL);
    root->player_just_moved = opponent(state->player);
    expand(root);
    
    if (root->num_children == 0) {
        free_tree(root);
        return 0;
    }
    
    if (use_gpu) {
        mcts_gpu(root, simulations);
    } else {
        mcts(root, simulations);
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
    
    timer_stop(&timer);
    *time_taken = timer.elapsed;
    
    free_tree(root);
    return best != NULL;
}

// Benchmark 1: Direct Speed Comparison
void benchmark_speed_comparison(int simulations, int num_positions) {
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║  Benchmark 1: CPU vs GPU Speed Comparison     ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("Testing %d simulations on %d different positions\n\n", 
           simulations, num_positions);
    
    double cpu_total = 0, gpu_total = 0;
    
    for (int pos = 0; pos < num_positions; pos++) {
        GameState state;
        init_board(&state);
        
        // Make some random moves to get varied positions
        for (int i = 0; i < (pos % 10); i++) {
            if (!has_valid_moves(&state)) break;
            int r, c;
            if (!get_random_move(&state, &r, &c)) break;
            make_move(&state, r, c);
        }
        
        // Test CPU
        int r, c;
        double cpu_time;
        get_mcts_move_timed(&state, simulations, &r, &c, 0, &cpu_time);
        cpu_total += cpu_time;
        
        // Test GPU
        double gpu_time;
        get_mcts_move_timed(&state, simulations, &r, &c, 1, &gpu_time);
        gpu_total += gpu_time;
        
        if ((pos + 1) % 10 == 0) {
            printf("  Progress: %d/%d positions\n", pos + 1, num_positions);
        }
    }
    
    double cpu_avg = cpu_total / num_positions;
    double gpu_avg = gpu_total / num_positions;
    double speedup = cpu_avg / gpu_avg;
    
    printf("\n┌─────────────────────────────────────────┐\n");
    printf("│ Results:                                │\n");
    printf("├─────────────────────────────────────────┤\n");
    printf("│ CPU Average Time: %8.4f seconds      │\n", cpu_avg);
    printf("│ GPU Average Time: %8.4f seconds      │\n", gpu_avg);
    printf("│ Speedup:          %8.2fx              │\n", speedup);
    printf("│ Total CPU Time:   %8.2f seconds      │\n", cpu_total);
    printf("│ Total GPU Time:   %8.2f seconds      │\n", gpu_total);
    printf("└─────────────────────────────────────────┘\n");
}

// Benchmark 2: Scaling with Simulation Count
void benchmark_simulation_scaling() {
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║  Benchmark 2: Simulation Count Scaling        ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("Testing how CPU and GPU scale with more work\n\n");
    
    int sim_counts[] = {100, 500, 1000, 2000, 5000, 10000};
    int num_configs = sizeof(sim_counts) / sizeof(sim_counts[0]);
    int trials_per_config = 20;
    
    printf("┌──────────┬────────────┬────────────┬──────────┐\n");
    printf("│   Sims   │  CPU (s)   │  GPU (s)   │  Speedup │\n");
    printf("├──────────┼────────────┼────────────┼──────────┤\n");
    
    for (int cfg = 0; cfg < num_configs; cfg++) {
        int sims = sim_counts[cfg];
        double cpu_total = 0, gpu_total = 0;
        
        for (int trial = 0; trial < trials_per_config; trial++) {
            GameState state;
            init_board(&state);
            
            // Mid-game position
            int setup[][2] = {{2,3}, {2,2}, {2,4}, {3,2}};
            for (int i = 0; i < 4; i++) {
                if (is_valid_move(&state, setup[i][0], setup[i][1])) {
                    make_move(&state, setup[i][0], setup[i][1]);
                }
            }
            
            int r, c;
            double cpu_time, gpu_time;
            get_mcts_move_timed(&state, sims, &r, &c, 0, &cpu_time);
            get_mcts_move_timed(&state, sims, &r, &c, 1, &gpu_time);
            
            cpu_total += cpu_time;
            gpu_total += gpu_time;
        }
        
        double cpu_avg = cpu_total / trials_per_config;
        double gpu_avg = gpu_total / trials_per_config;
        double speedup = cpu_avg / gpu_avg;
        
        printf("│ %8d │  %8.4f  │  %8.4f  │  %6.2fx  │\n", 
               sims, cpu_avg, gpu_avg, speedup);
    }
    
    printf("└──────────┴────────────┴────────────┴──────────┘\n");
}

// Benchmark 3: Full Game Performance
void benchmark_game_performance(int simulations, int num_games) {
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║  Benchmark 3: Full Game Performance           ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("Playing %d complete games with %d sims/move\n\n", 
           num_games, simulations);
    
    double cpu_total_time = 0, gpu_total_time = 0;
    int cpu_total_moves = 0, gpu_total_moves = 0;
    
    // CPU games
    printf("Running CPU games...\n");
    for (int game = 0; game < num_games; game++) {
        GameState state;
        init_board(&state);
        int moves = 0;
        
        while (1) {
            if (!has_valid_moves(&state)) {
                state.player = opponent(state.player);
                if (!has_valid_moves(&state)) break;
            }
            
            int r, c;
            double time_taken;
            if (!get_mcts_move_timed(&state, simulations, &r, &c, 0, &time_taken)) 
                break;
            
            cpu_total_time += time_taken;
            make_move(&state, r, c);
            moves++;
        }
        cpu_total_moves += moves;
        
        if ((game + 1) % 5 == 0) {
            printf("  CPU Progress: %d/%d games\n", game + 1, num_games);
        }
    }
    
    // GPU games
    printf("Running GPU games...\n");
    for (int game = 0; game < num_games; game++) {
        GameState state;
        init_board(&state);
        int moves = 0;
        
        while (1) {
            if (!has_valid_moves(&state)) {
                state.player = opponent(state.player);
                if (!has_valid_moves(&state)) break;
            }
            
            int r, c;
            double time_taken;
            if (!get_mcts_move_timed(&state, simulations, &r, &c, 1, &time_taken)) 
                break;
            
            gpu_total_time += time_taken;
            make_move(&state, r, c);
            moves++;
        }
        gpu_total_moves += moves;
        
        if ((game + 1) % 5 == 0) {
            printf("  GPU Progress: %d/%d games\n", game + 1, num_games);
        }
    }
    
    double cpu_avg_game = cpu_total_time / num_games;
    double gpu_avg_game = gpu_total_time / num_games;
    double cpu_avg_move = cpu_total_time / cpu_total_moves;
    double gpu_avg_move = gpu_total_time / gpu_total_moves;
    
    printf("\n┌─────────────────────────────────────────────────┐\n");
    printf("│ Results:                                        │\n");
    printf("├─────────────────────────────────────────────────┤\n");
    printf("│ CPU Total Time:      %10.2f seconds        │\n", cpu_total_time);
    printf("│ GPU Total Time:      %10.2f seconds        │\n", gpu_total_time);
    printf("│ CPU Avg/Game:        %10.4f seconds        │\n", cpu_avg_game);
    printf("│ GPU Avg/Game:        %10.4f seconds        │\n", gpu_avg_game);
    printf("│ CPU Avg/Move:        %10.4f seconds        │\n", cpu_avg_move);
    printf("│ GPU Avg/Move:        %10.4f seconds        │\n", gpu_avg_move);
    printf("│ Game Speedup:        %10.2fx               │\n", cpu_avg_game/gpu_avg_game);
    printf("│ Move Speedup:        %10.2fx               │\n", cpu_avg_move/gpu_avg_move);
    printf("└─────────────────────────────────────────────────┘\n");
}

// Benchmark 4: Quality Comparison (do both produce similar results?)
void benchmark_decision_quality(int simulations, int num_positions) {
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║  Benchmark 4: Decision Quality Comparison     ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("Comparing move quality between CPU and GPU\n");
    printf("Testing %d positions with %d simulations\n\n", 
           num_positions, simulations);
    
    int same_moves = 0;
    int total_positions = 0;
    
    for (int pos = 0; pos < num_positions; pos++) {
        GameState state;
        init_board(&state);
        
        // Create varied positions
        for (int i = 0; i < (pos % 15); i++) {
            if (!has_valid_moves(&state)) break;
            int r, c;
            if (!get_random_move(&state, &r, &c)) break;
            make_move(&state, r, c);
        }
        
        if (!has_valid_moves(&state)) continue;
        
        int cpu_r, cpu_c, gpu_r, gpu_c;
        double time_taken;
        
        if (!get_mcts_move_timed(&state, simulations, &cpu_r, &cpu_c, 0, &time_taken))
            continue;
        if (!get_mcts_move_timed(&state, simulations, &gpu_r, &gpu_c, 1, &time_taken))
            continue;
        
        total_positions++;
        if (cpu_r == gpu_r && cpu_c == gpu_c) {
            same_moves++;
        }
        
        if ((pos + 1) % 10 == 0) {
            printf("  Progress: %d/%d positions, Agreement: %.1f%%\n", 
                   pos + 1, num_positions, 
                   100.0 * same_moves / total_positions);
        }
    }
    
    double agreement = 100.0 * same_moves / total_positions;
    
    printf("\n┌─────────────────────────────────────────┐\n");
    printf("│ Results:                                │\n");
    printf("├─────────────────────────────────────────┤\n");
    printf("│ Positions Tested:  %6d              │\n", total_positions);
    printf("│ Same Moves:        %6d (%.1f%%)      │\n", 
           same_moves, agreement);
    printf("│ Different Moves:   %6d (%.1f%%)      │\n", 
           total_positions - same_moves, 100.0 - agreement);
    printf("└─────────────────────────────────────────┘\n");
    
    if (agreement > 85) {
        printf("\n✓ Excellent agreement - implementations are very consistent\n");
    } else if (agreement > 70) {
        printf("\n✓ Good agreement - minor variations expected due to randomness\n");
    } else if (agreement > 50) {
        printf("\n⚠ Moderate agreement - check for implementation differences\n");
    } else {
        printf("\n✗ Low agreement - likely bug in one implementation\n");
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║     MCTS CPU vs GPU Benchmark Suite           ║\n");
    printf("║     Leaf Parallelism Performance Analysis     ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    
    if (argc > 1 && strcmp(argv[1], "quick") == 0) {
        printf("\n[QUICK MODE - Fast testing]\n");
        benchmark_speed_comparison(1000, 20);
        benchmark_simulation_scaling();
        benchmark_game_performance(500, 5);
        benchmark_decision_quality(1000, 30);
    } else if (argc > 1 && strcmp(argv[1], "full") == 0) {
        printf("\n[FULL MODE - Comprehensive analysis]\n");
        benchmark_speed_comparison(5000, 50);
        benchmark_simulation_scaling();
        benchmark_game_performance(2000, 20);
        benchmark_decision_quality(5000, 100);
    } else {
        printf("\n[STANDARD MODE]\n");
        printf("Usage: %s [quick|full]\n\n", argv[0]);
        benchmark_speed_comparison(2000, 30);
        benchmark_simulation_scaling();
        benchmark_game_performance(1000, 10);
        benchmark_decision_quality(2000, 50);
    }
    
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║     Benchmark Complete!                        ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("\nInterpretation Guide:\n");
    printf("• Higher speedup = better GPU performance\n");
    printf("• Speedup should increase with more simulations\n");
    printf("• High decision agreement (>80%%) shows correctness\n");
    printf("• GPU excels at large batch sizes (BATCH_SIZE constant)\n");
    
    return 0;
}