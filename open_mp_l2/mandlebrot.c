#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 10000

int mandelbrot(double complex c) {
 double complex z = 0;
 int iter = 0;
 while (cabs(z) <= 2.0 && iter < MAX_ITER) {
 z = z*z + c;
 iter++;
 }
 return iter;
}

int main() {
 int *image = malloc(WIDTH * HEIGHT * sizeof(int));
 double xmin = -2.0, xmax = 1.0;
 double ymin = -1.5, ymax = 1.5;
 double dx = (xmax - xmin) / WIDTH;
 double dy = (ymax - ymin) / HEIGHT;
 int i, j;
 printf("Starting Mandelbrot computation...\n");
 double t0 = omp_get_wtime();
 #pragma omp parallel for schedule(dynamic,1) private(j)
 for (i = 0; i < HEIGHT; i++) {
 for (j = 0; j < WIDTH; j++) {
 double x = xmin + j * dx;
 double y = ymin + i * dy;
 image[i*WIDTH + j] = mandelbrot(x + y*I);
 }
 }
 double t1 = omp_get_wtime();
 printf("Done. Time taken: %.4f seconds\n", t1 - t0);
 // Save a simple PGM image
 FILE *f = fopen("mandelbrot.pgm", "w");
 fprintf(f, "P2\n%d %d\n%d\n", WIDTH, HEIGHT, MAX_ITER);
 for (i = 0; i < HEIGHT; i++) {
 for (j = 0; j < WIDTH; j++) {
 fprintf(f, "%d ", image[i*WIDTH + j]);
 }
 fprintf(f, "\n");
}
 fclose(f);
 free(image);
 return 0;
}
