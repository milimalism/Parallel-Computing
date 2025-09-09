#include <stdio.h>
#include <omp.h>
#define N_steps 1000000000
int main(){
 
 double pi=0.0;
 double dx;
 double s_time, t_time;
 dx = 1.0/N_steps;

 s_time = omp_get_wtime();
 int act_nthreads;

 printf("Default number of threads: %d\n",omp_get_max_threads());

//this works because within the parallel block, the variables are all local to each thread, threrefore, each "sum" is per thread and can be added one-by-one using critical to pi 
//this also works with atomic
#pragma omp parallel
 {
 int i;
 double sum;
 int id = omp_get_thread_num();
 int numthreads = omp_get_num_threads();
 if(id ==0) act_nthreads = numthreads;
 double x;

 for(i=id;i<N_steps;i+=numthreads){
 x = (i+0.5)*dx;
 sum += 4.0 / (1.0 + x*x);
 }

// #pragma omp critical
// {
// pi +=sum *dx;
// }

 #pragma omp atomic 
   pi+=sum;

 }
 pi*=dx;
 t_time = omp_get_wtime()-s_time;
 printf("pi = %.15lf, %ld steps, %lf secs, %d threads\n",pi, N_steps,t_time,act_nthreads);
 return 0;
}
