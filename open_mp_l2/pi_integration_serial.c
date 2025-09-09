//import libraries
#include <stdio.h>
#include <omp.h>
#define N_steps 1000000000

//function declaration
int main(){

//variable declaration
 int i;
 double pi;
 double sum = 0.0;
 double x, dx;
 double s_time, t_time;

 dx = 1.0/N_steps;

// get current time
 s_time = omp_get_wtime();

// apply formula
 for(i=0;i<N_steps;i++){
 x = (i+0.5)*dx;
 sum += 4.0 / (1.0 + x*x);
 }
 
 pi = sum * dx;

//time after operation
 t_time = omp_get_wtime()-s_time;
 printf("pi = %.15lf, %ld steps, %lf secs\n",pi, N_steps, t_time);
 return 0;
}
