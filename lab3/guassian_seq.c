// gaussian_seq.c
// Input format:
//   n
//   A (n lines, n values each)
//   b (1 line, n values)
// Example (n=3):
// 3
// 2 1 -1
// -3 -1 2
// -2 1 2
// 8  -11  -3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


// Helper: count rows and cols in A.txt
void check_solution(const char *xfile, const double *x, size_t n, double tol);

static inline size_t idx(size_t i, size_t j, size_t n) { return i * n + j; }

int main(int argc, char *argv[])
{
    setvbuf(stdout, NULL, _IONBF, 0);
    fprintf(stderr, "Argument count (argc): %d\n", argc); // Print argc

    // if (argc < 3)
    // {
    //     fprintf(stderr, "Usage: %s <A_file> <B_file>\n", argv[0]);
    //     return 1;
    // }

    const char *afile = argv[1];
    const char *bfile = argv[2];
    const char *xfile = argv[3];

    printf("Hellloooooo\n");

    size_t nrows, ncols;
    nrows = ncols = (size_t)strtoul(argv[4], NULL, 10);

    size_t n = nrows;

    double *A = malloc(n * n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *x = calloc(n, sizeof(double));

    // Read A
    FILE *fa = fopen(afile, "r");
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            fscanf(fa, "%lf", &A[idx(i, j, n)]);
        }
    }
    fclose(fa);

    // Read b
    FILE *fb = fopen(bfile, "r");
    for (size_t i = 0; i < n; i++)
    {
        fscanf(fb, "%lf", &b[i]);
    }
    fclose(fb);

    // Forward Elimination (no partial pivoting; diagonal used as pivot)
    for(size_t k=0;k<n;k++){
        double pivot = A[idx(k,k,n)];
        if (fabs(pivot) < 1e-12){
            fprintf(stderr, "Near-zero pivot encountered at row %zu (value=%g)\n", k, pivot);
            // continue to avoid division by zero
            // (lab spec: partial pivoting not required)
        }
        for(size_t i=k+1;i<n;i++){
            double factor = (fabs(pivot) < 1e-12) ? 0.0 : A[idx(i,k,n)]/pivot;
            // Update row i starting from column k to keep work minimal
            for(size_t j=k;j<n;j++){
                A[idx(i,j,n)] -= factor * A[idx(k,j,n)];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back Substitution
    for(long long i=(long long)n-1; i>=0; i--){
        double sum = b[i];
        for(size_t j=i+1;j<n;j++){
            sum -= A[idx((size_t)i,j,n)] * x[j]; // subtract the values so that only the unknown and its coefficient (the diagonal element) contribute to it
        }
        double aii = A[idx((size_t)i,(size_t)i,n)]; // diagonal element
        if (fabs(aii) < 1e-12){
            fprintf(stderr, "Near-zero diagonal at row %lld; setting x[%lld]=0\n", i, i);
            x[i] = 0.0;
        } else {
            x[i] = sum / aii;
        }
    }

    // Output solution
    // for(size_t i=0;i<n;i++){
    //     printf("%.10f%c", x[i], (i+1==n)?'\n':' ');
    // }
    check_solution(xfile, x, n, 1e-6);

    free(A); free(b); free(x);
    return 0;
}

void check_solution(const char *xfile, const double *x, size_t n, double tol)
{
    FILE *fp = fopen(xfile, "r");
    if (!fp)
    {
        fprintf(stderr, "Could not open reference solution file: %s\n", xfile);
        return;
    }

    int all_ok = 1;
    for (size_t i = 0; i < n; i++)
    {
        double ref;
        if (fscanf(fp, "%lf", &ref) != 1)
        {
            fprintf(stderr, "Error: Not enough values in reference file at index %zu\n", i);
            fclose(fp);
            return;
        }
        double diff = fabs(x[i] - ref);
        if (diff > tol)
        {
            fprintf(stderr, "Mismatch at index %zu: computed=%.10f, expected=%.10f, diff=%.3e\n",
                    i, x[i], ref, diff);
            all_ok = 0;
        }
    }
    fclose(fp);

    if (all_ok)
        printf("✅ Computed solution matches reference within tolerance %.3e\n", tol);
    else
        printf("❌ Solution does not fully match reference (see mismatches above)\n");
}

void get_matrix_size(const char *filename, size_t *rows, size_t *cols)
{
    printf("Debug: Entering get_matrix_size with filename: %s\n", filename);

    FILE *fp = fopen(filename, "r");
    printf("filename : %s\n", filename);
    if (!fp)
    {
        perror("open A file");
        exit(1);
    }

    *rows = 0;
    *cols = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp))
    {
        if (*rows == 0)
        {
            // count number of columns in first line
            char *tmp = strdup(line);
            char *tok = strtok(tmp, " \t\n");
            while (tok)
            {
                (*cols)++;
                tok = strtok(NULL, " \t\n");
            }
            free(tmp);
        }
        (*rows)++;
    }
    fclose(fp);
}
