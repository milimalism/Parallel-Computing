#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#define N 1<<26

int main(){
  	int i=0;
	int *num_array = malloc(sizeof(int) * (size_t)N);
	double s_time, t_time;

	for(i=0; i<N; i++){
		num_array[i]=i;
	}

	long long sum = 0;
	int act_nthreads;
	#pragma omp parallel
	{
	int id = omp_get_thread_num();
	int numthreads = omp_get_num_threads();
	if (id==0) act_nthreads = numthreads;
	s_time = omp_get_wtime();

	#pragma omp for reduction(+:sum)
	for(i=0; i<N;i++){
		sum+=(long long)num_array[i];
	}
	}
	t_time = omp_get_wtime() - s_time;

	printf("Time taken for parallel with %d threads: %f\n", t_time, act_nthreads);
	printf("Sum : %lld\n", sum);
}
