#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void check_solution(const char *xfile, const double *x, size_t n, double tol);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <A_file> <B_file> <x_ref> <n>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const char *afile = argv[1];
    const char *bfile = argv[2];
    const char *xfile = argv[3];
    const size_t n = (size_t)strtoul(argv[4], NULL, 10);

    // REQUIRE: equal row blocks to use MPI_Scatter/Gather simply
    if (n % (size_t)size != 0)
    {
        if (rank == 0)
            fprintf(stderr, "n (%zu) must be divisible by #procs (%d) for MPI_Scatter/Gather.\n", n, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t local_rows = n / (size_t)size;

    // Root reads full A,b; others keep NULL
    double *A = NULL, *b = NULL, *x = NULL;
    if (rank == 0)
    {
        A = (double *)malloc(n * n * sizeof(double));
        b = (double *)malloc(n * sizeof(double));
        x = (double *)calloc(n, sizeof(double));
        FILE *fa = fopen(afile, "r");
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                fscanf(fa, "%lf", &A[idx(i, j, n)]);
        fclose(fa);
        FILE *fb = fopen(bfile, "r");
        for (size_t i = 0; i < n; i++)
            fscanf(fb, "%lf", &b[i]);
        fclose(fb);
    }

    // Local storage for rows
    double *A_local = (double *)malloc(local_rows * n * sizeof(double));
    double *b_local = (double *)malloc(local_rows * sizeof(double));

    // STEP 1: Scatter rows of A and b
    MPI_Scatter(A, (int)(local_rows * n), MPI_DOUBLE,
                A_local, (int)(local_rows * n), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Scatter(b, (int)local_rows, MPI_DOUBLE,
                b_local, (int)local_rows, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Buffers for broadcasting pivot row and rhs
    double *pivot_row = (double *)malloc(n * sizeof(double));
    double pivot_rhs;

    // STEP 2–3: Forward elimination with broadcast of pivot row each step
    for (size_t k = 0; k < n; ++k)
    {
        int owner = (int)(k / local_rows); // which rank owns pivot row
        int local_k = (int)(k - (size_t)owner * local_rows);

        if (rank == owner)
        {
            // copy full pivot row (length n) and its RHS
            memcpy(pivot_row, &A_local[idx((size_t)local_k, 0, n)], n * sizeof(double));
            pivot_rhs = b_local[(size_t)local_k];
        }
        // STEP 2: Broadcast pivot row and pivot RHS to all
        MPI_Bcast(pivot_row, (int)n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_rhs, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // STEP 3: Each process updates its assigned rows (only rows with global_i > k)
        const double pivot = pivot_row[k];
        for (size_t li = 0; li < local_rows; ++li)
        {
            size_t global_i = (size_t)rank * local_rows + li;
            if (global_i <= k)
                continue;

            double aik = A_local[idx(li, k, n)];
            double factor = (fabs(pivot) < 1e-12) ? 0.0 : (aik / pivot);

            // Update row entries j = k..n-1 using pivot_row
            for (size_t j = k; j < n; ++j)
            {
                A_local[idx(li, j, n)] -= factor * pivot_row[j];
            }
            b_local[li] -= factor * pivot_rhs;
        }
        // (No need to zero A_local[local_k,k] explicitly; arithmetic does it.)
    }

    // STEP 4: Gather the upper-triangular matrix and modified RHS to root
    if (rank != 0)
    {
        MPI_Gather(A_local, (int)(local_rows * n), MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(b_local, (int)local_rows, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gather(A_local, (int)(local_rows * n), MPI_DOUBLE, A, (int)(local_rows * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(b_local, (int)local_rows, MPI_DOUBLE, b, (int)local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // STEP 5: Back substitution on root
    if (rank == 0)
    {
        for (long long i = (long long)n - 1; i >= 0; --i)
        {
            double sum = b[i];
            for (size_t j = (size_t)i + 1; j < n; ++j)
                sum -= A[idx((size_t)i, j, n)] * x[j];
            double aii = A[idx((size_t)i, (size_t)i, n)];
            x[i] = (fabs(aii) < 1e-12) ? 0.0 : (sum / aii);
        }
        // (Optional) verify
        check_solution(xfile, x, n, 1e-6);
    }

    // Clean up
    free(pivot_row);
    free(A_local);
    free(b_local);
    if (rank == 0)
    {
        free(A);
        free(b);
        free(x);
    }

    MPI_Finalize();
    return 0;
}
