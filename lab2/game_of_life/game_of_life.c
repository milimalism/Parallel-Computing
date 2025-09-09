#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10 // Grid size N x N (keep small if you want to print patterns)
#define STEPS 5 // Number of generations

int grid[N][N];
int newgrid[N][N];

// Initialize grid randomly with 0s and 1s
void initialize_grid() {
    srand(1234); // fixed seed for reproducibility
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = rand() % 2;
            }
        }
}

// Print grid to console
void print_grid() {
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        printf("%c", grid[i][j] ? 'O' : '.'); // O = alive, . = dead
        }
    printf("\n");
    }
    printf("\n");
}

// Count live neighbors of cell (x, y)
int count_neighbors(int x, int y) {
    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                count += grid[nx][ny];
            }
        }
    }
    return count;
}

// Update grid to next generation
// TODO: Students parallelize this function with OpenMP
void update() {
    //for the grid
    // if alive and count_neighbors() < 3 -> 0
    // if alive and count_neighbors() == 3 or count_neighbors() == 4 -> 1
    // if alive and count_neighbors() > 4 -> 0
    // if dead and count_neighbors() == 2 -> 1, else 0
    
    #pragma omp parallel for collapse(2) schedule(runtime)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            int cur = grid[i][j];
            if (((cur == 1) && (count_neighbors(i,j) < 3)) || ((cur == 1) && (count_neighbors(i,j) > 4)) || ((cur == 0) && (count_neighbors(i,j) != 2))) {
                newgrid[i][j] = 0;
            }
            else if (((cur == 1) && (count_neighbors(i,j) == 3)) || ((cur == 1) && (count_neighbors(i,j) == 4)) || ((cur == 0) && (count_neighbors(i,j) == 2)))
            {
                newgrid[i][j] = 1;
            }   
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            grid[i][j] = newgrid[i][j];
        }
    }
}

int main() {
    initialize_grid();
    printf("Initial Pattern:\n");
    print_grid();
    double start = omp_get_wtime();
    for (int step = 0; step < STEPS; step++) {
    update();
    }
    double end = omp_get_wtime();
    printf("Simulation finished in %f seconds\n", end - start);
    printf("Final Pattern after %d steps:\n", STEPS);
    print_grid();
    printf("\n");
    return 0;
}
