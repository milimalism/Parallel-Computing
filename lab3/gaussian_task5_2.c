#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define IDX(i, j, n) ((size_t)(i) * (size_t)(n) + (size_t)(j))

static void check_solution(const char *xfile, const double *x, size_t n, double tol)
{
    if (!xfile)
        return;
    FILE *fx = fopen(xfile, "r");
    if (!fx)
    {
        fprintf(stderr, "Warning: could not open xfile '%s' for checking.\n", xfile);
        return;
    }
    size_t i = 0;
    double val;
    size_t bad = 0;
    while (i < n && fscanf(fx, "%lf", &val) == 1)
    {
        double diff = fabs(val - x[i]);
        if (diff > tol)
        {
            bad++;
            if (bad <= 5)
            {
                fprintf(stderr, "Mismatch at %zu: got %.10g, expected %.10g (diff=%.3g > tol=%.3g)\n",
                        i, x[i], val, diff, tol);
            }
        }
        i++;
    }
    fclose(fx);
    if (i != n)
    {
        fprintf(stderr, "Warning: xfile length (%zu) != n (%zu)\n", i, n);
    }
    if (bad == 0)
    {
        printf("Solution check: OK (all entries within tolerance %.3g)\n", tol);
    }
    else
    {
        printf("Solution check: %zu entries exceeded tolerance %.3g\n", bad, tol);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double t_start, t_end;

    if (argc < 4)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Usage: %s <n> <A.txt> <b.txt> [x_true.txt] [tol]\n", argv[0]);
            fprintf(stderr, "  Files are plain text with whitespace-separated numbers.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const char *afile = argv[1];
    const char *bfile = argv[2];
    const char *xtrue = argv[3];
    const size_t n = (size_t)strtoul(argv[4], NULL, 10);
    double tol = (argc >= 6 ? atof(argv[5]) : 1e-6);

    // Compute irregular row partitioning
    size_t q = n / (size_t)size;
    size_t r = n % (size_t)size;
    size_t local_rows = q + ((size_t)rank < r ? 1 : 0); // q + rank

    // Build row_offsets (prefix sum) on all ranks (simple and avoids an extra bcast)
    size_t *row_offsets = (size_t *)malloc((size_t)size + 1U * sizeof(size_t));
    if (!row_offsets)
    {
        fprintf(stderr, "Allocation failure for row_offsets\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    row_offsets[0] = 0;
    for (int p = 0; p < size; ++p)
    {
        size_t rp = q + ((size_t)p < r ? 1 : 0);
        row_offsets[p + 1] = row_offsets[p] + rp;
    }

    // Global arrays (root only)
    double *A = NULL;
    double *b = NULL;
    if (rank == 0)
    {
        A = (double *)malloc(n * n * sizeof(double));
        b = (double *)malloc(n * sizeof(double));
        if (!A || !b)
        {
            fprintf(stderr, "Root allocation failure for A/b\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        // Load A
        FILE *fa = fopen(afile, "r");
        if (!fa)
        {
            fprintf(stderr, "Cannot open A file: %s\n", afile);
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                if (fscanf(fa, "%lf", &A[IDX(i, j, n)]) != 1)
                {
                    fprintf(stderr, "Failed to read A[%zu,%zu]\n", i, j);
                    MPI_Abort(MPI_COMM_WORLD, 5);
                }
            }
        }
        fclose(fa);
        // Load b
        FILE *fb = fopen(bfile, "r");
        if (!fb)
        {
            fprintf(stderr, "Cannot open b file: %s\n", bfile);
            MPI_Abort(MPI_COMM_WORLD, 6);
        }
        for (size_t i = 0; i < n; ++i)
        {
            if (fscanf(fb, "%lf", &b[i]) != 1)
            {
                fprintf(stderr, "Failed to read b[%zu]\n", i);
                MPI_Abort(MPI_COMM_WORLD, 7);
            }
        }
        fclose(fb);
    }

    // Local storage
    double *A_local = (double *)malloc(local_rows * n * sizeof(double));
    double *b_local = (double *)malloc(local_rows * sizeof(double));
    if ((!A_local || !b_local))
    {
        fprintf(stderr, "Rank %d allocation failure for locals\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 8);
    }

    // Build counts/displacements for Scatterv/Gatherv (root only)
    int *cntA = NULL, *dispA = NULL, *cntb = NULL, *dispb = NULL;
    if (rank == 0)
    {
        cntA = (int *)malloc(size * sizeof(int));
        dispA = (int *)malloc(size * sizeof(int));
        cntb = (int *)malloc(size * sizeof(int));
        dispb = (int *)malloc(size * sizeof(int));
        if (!cntA || !dispA || !cntb || !dispb)
        {
            fprintf(stderr, "Root allocation failure for counts/disps\n");
            MPI_Abort(MPI_COMM_WORLD, 9);
        }
        for (int p = 0; p < size; ++p)
        {
            size_t rp = row_offsets[p + 1] - row_offsets[p];
            cntA[p] = (int)(rp * n);
            dispA[p] = (int)(row_offsets[p] * n);
            cntb[p] = (int)rp;
            dispb[p] = (int)row_offsets[p];
        }
    }

    t_start = MPI_Wtime();

    // STEP 1: Scatter rows of A and b (variable size)
    MPI_Scatterv(A, cntA, dispA, MPI_DOUBLE,
                 A_local, (int)(local_rows * n), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, cntb, dispb, MPI_DOUBLE,
                 b_local, (int)local_rows, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Buffers for pivot row and rhs (broadcast each step)
    double *pivot_row = (double *)malloc(n * sizeof(double));
    if (!pivot_row)
    {
        fprintf(stderr, "Rank %d allocation failure for pivot_row\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 10);
    }
    double pivot_rhs = 0.0;

    // STEP 2-3: Forward elimination with distributed rows
    for (size_t k = 0; k < n; ++k)
    {
        // Determine owner of pivot row k
        // owner is largest p with row_offsets[p] <= k < row_offsets[p+1]
        int owner = -1;
        // Cheap linear search (size is usually small); could do binary search if large
        for (int p = 0; p < size; ++p)
        {
            if (k >= row_offsets[p] && k < row_offsets[p + 1])
            {
                owner = p;
                break;
            }
        }
        if (owner == -1)
        {
            fprintf(stderr, "Could not find owner for k=%zu\n", k);
            MPI_Abort(MPI_COMM_WORLD, 11);
        }
        int local_k = (int)(k - row_offsets[owner]);

        if (rank == owner)
        {
            // Copy the pivot row into buffer
            memcpy(pivot_row, &A_local[IDX((size_t)local_k, 0, n)], n * sizeof(double));
            pivot_rhs = b_local[(size_t)local_k];
        }
        // Broadcast pivot row and rhs
        MPI_Bcast(pivot_row, (int)n, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_rhs, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        double pivot = pivot_row[k];
        // Optional: handle (near-)singular pivot
        if (fabs(pivot) < 1e-15)
        {
            // Skip elimination to avoid NaNs; this matrix might be singular or ill-conditioned
            continue;
        }

        // Update local rows with global index i > k
        for (size_t li = 0; li < local_rows; ++li)
        {
            size_t gi = row_offsets[rank] + li;
            if (gi <= k)
                continue;
            double aik = A_local[IDX(li, k, n)];
            double factor = aik / pivot;
            if (factor == 0.0)
                continue;
            // A[gi, j] -= factor * pivot_row[j] for j=k..n-1
            for (size_t j = k; j < n; ++j)
            {
                A_local[IDX(li, j, n)] -= factor * pivot_row[j];
            }
            b_local[li] -= factor * pivot_rhs;
        }
    }

    // STEP 4: Gather the upper-triangular matrix and the modified RHS on root
    if (rank == 0)
    {
        MPI_Gatherv(A_local, (int)(local_rows * n), MPI_DOUBLE,
                    A, cntA, dispA, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(b_local, (int)local_rows, MPI_DOUBLE,
                    b, cntb, dispb, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Gatherv(A_local, (int)(local_rows * n), MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(b_local, (int)local_rows, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // STEP 5: Back substitution on root
    double *x = NULL;
    if (rank == 0)
    {
        x = (double *)calloc(n, sizeof(double));
        if (!x)
        {
            fprintf(stderr, "Root allocation failure for x\n");
            MPI_Abort(MPI_COMM_WORLD, 12);
        }
        for (ssize_t i = (ssize_t)n - 1; i >= 0; --i)
        {
            double sum = b[(size_t)i];
            for (size_t j = (size_t)i + 1; j < n; ++j)
            {
                sum -= A[IDX((size_t)i, j, n)] * x[j];
            }
            double diag = A[IDX((size_t)i, (size_t)i, n)];
            if (fabs(diag) < 1e-15)
            {
                // Singular or nearly singular; set to zero to avoid inf
                x[(size_t)i] = 0.0;
            }
            else
            {
                x[(size_t)i] = sum / diag;
            }
        }
        t_end = MPI_Wtime();

        if (rank == 0)
        {
            printf("Compute time: %.6f seconds\n", t_end - t_start);
        }

        // Optional check against reference solution
        if (xtrue)
        {
            check_solution(xtrue, x, n, tol);
        }
        else
        {
            // Print first few entries for sanity
            size_t show = (n < 6 ? n : 6);
            printf("x[0:%zu):", show);
            for (size_t i = 0; i < show; ++i)
                printf(" %.6g", x[i]);
            printf("\n");
        }
    }

    // Cleanup
    free(pivot_row);
    free(A_local);
    free(b_local);
    free(row_offsets);
    if (rank == 0)
    {
        free(A);
        free(b);
        free(x);
        free(cntA);
        free(dispA);
        free(cntb);
        free(dispb);
    }

    MPI_Finalize();
    return 0;
}
