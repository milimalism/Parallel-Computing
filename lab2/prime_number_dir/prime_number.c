#include <omp.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#define N 40000000000

int main(){
  
  //loop through nums
  double s_time, t_time;

  s_time = omp_get_wtime();
  int num_prime = 0;
  #pragma omp parallel for reduction(+:num_prime) schedule(runtime) 
  for(int i=2; i<N; i++)
  {
    bool prime = true;
    for(int j=2; j*j<=i; j++){
	 if(i%j==0){
	    prime = false;
	    break;
	 }
    }
    if (prime){num_prime+=1;}
  }
  t_time = omp_get_wtime() - s_time;
  printf("num_prime: %d, in %f seconds with scheduling: %s\n", num_prime, t_time, getenv("OMP_SCHEDULE"));
  return 0;
}
