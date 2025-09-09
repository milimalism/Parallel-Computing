//this program requires omp_num_threads to be set to 4 (i think the default 128 is too high and causes the fault, on right cause its 50 so use export OMP_NUM_THREADS or 

// import libs
#include <stdio.h>
#include <omp.h>
#define N_steps 1000000000

int main(){
// define vars
 double pi;
 double dx;
 double s_time, t_time;
 dx = 1.0/N_steps;

// get current time
 s_time = omp_get_wtime();

// set number of threads
 omp_set_num_threads(4);

 int act_nthreads;
 double sum[50] = {0.0};

 printf("%d\n",omp_get_max_threads());

 #pragma omp parallel num_threads(4)
 {
 int i;
 int id = omp_get_thread_num();
 int numthreads = omp_get_num_threads();

 if(id ==0) act_nthreads = numthreads;

 double x;

 for(i=id;i<N_steps;i+=numthreads){
 x = (i+0.5)*dx;
 sum[id] += 4.0 / (1.0 + x*x);
 }

 }

 pi = 0.0;
 for (int i=0; i<act_nthreads;i++){
 pi +=sum[i];
 }
 pi *= dx;

 t_time = omp_get_wtime()-s_time;

 printf("pi = %.15lf, %ld steps, %lf secs, %d threads\n",pi,N_steps, t_time,act_nthreads);
 return 0;
}
