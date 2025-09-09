#include <stdio.h>
#include <omp.h>
#define NTHREADS 10
int main(){
 omp_set_num_threads(NTHREADS);
 int i;
 #pragma omp parallel for ordered
 for(i=0;i<NTHREADS;i++){
 #pragma omp ordered
 printf("%d ",omp_get_thread_num());
 }
 printf("\n");
}
