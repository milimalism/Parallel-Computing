#include <stdio.h>
#include <omp.h>
#define NTHREADS 10
int main(){
 omp_set_num_threads(NTHREADS);
 int num;
 #pragma omp parallel
 {
 int id = omp_get_thread_num();
 printf("Hello from thread %d\n", id);
 if(id==0) num = omp_get_num_threads();
 }
 printf("The total number of threads is %d\n",num);
}
