# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11 -Iinclude -fopenmp   # <- add -fopenmp here
LDFLAGS = -lm -fopenmp                                     # <- and here for linking

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

# Source files
SOURCES = $(SRC_DIR)/othello.c $(SRC_DIR)/mcts.c $(SRC_DIR)/mcts_leaf.c $(SRC_DIR)/mcts_root.c benchmark.c
OBJECTS = $(OBJ_DIR)/othello.o $(OBJ_DIR)/mcts.o $(OBJ_DIR)/mcts_leaf.o $(OBJ_DIR)/mcts_root.o $(OBJ_DIR)/benchmark.o

# Headers
HEADERS = $(INC_DIR)/othello.h $(INC_DIR)/mcts.h $(INC_DIR)/mcts_leaf.h $(INC_DIR)/mcts_root.h

# Target executable
TARGET = benchmark

# Default target
all: $(TARGET)

# Create obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Build complete! Run with:"
	@echo "  ./$(TARGET)           (standard mode)"
	@echo "  ./$(TARGET) quick     (quick mode)"
	@echo "  ./$(TARGET) full      (full mode)"

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/benchmark.o: benchmark.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(TARGET)
	@echo "Clean complete!"

# Rebuild everything
rebuild: clean all

# Run benchmarks
run: $(TARGET)
	./$(TARGET)

run-quick: $(TARGET)
	./$(TARGET) quick

run-full: $(TARGET)
	./$(TARGET) full

# Debug build (with debug symbols and no optimization)
debug: CFLAGS = -Wall -Wextra -g -O0 -std=c11 -Iinclude -fopenmp
debug: clean all

# Phony targets
.PHONY: all clean rebuild run run-quick run-full debug
