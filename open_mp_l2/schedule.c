#include <stdio.h>
#include <omp.h>
#define THREAD_NUM 4
int main() {
 int i;
 omp_set_num_threads(4);
 #pragma omp parallel for schedule(runtime) // try dynamic, guided, runtime, auto
 for (i = 0; i < 30; i++) {
 printf("Thread %d: iteration %d\n",
 omp_get_thread_num(), i);
 }
}
