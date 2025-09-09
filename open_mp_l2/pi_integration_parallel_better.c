// imports
#include <stdio.h>
#include <omp.h>
#define N_steps 1000000000
#define CBLK 8

int main(){

 double pi;
 double dx;
 double s_time, t_time;
 dx = 1.0/N_steps;

 s_time = omp_get_wtime();

 int act_nthreads;
 double sum[50][CBLK] = {0.0};

 printf("Default number of threads: %d\n",omp_get_max_threads());

 #pragma omp parallel //parallel means that that block will be executed in parallel with the threads
 {
 int i;
 int id = omp_get_thread_num();
 int numthreads = omp_get_num_threads();
 if(id ==0) act_nthreads = numthreads;
 double x;
 for(i=id;i<N_steps;i+=numthreads){
 x = (i+0.5)*dx;
 sum[id][0] += 4.0 / (1.0 + x*x);
 }
 }

 pi = 0.0;
 for (int i=0; i<act_nthreads;i++){
 pi +=sum[i][0];
 }
 pi *= dx;

 t_time = omp_get_wtime()-s_time;

 printf("pi = %.15lf, %ld steps, %lf secs, %d threads\n",pi, N_steps, t_time,act_nthreads);

 return 0;
}
