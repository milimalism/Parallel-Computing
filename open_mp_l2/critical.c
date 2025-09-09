#include <omp.h>
#include <stdio.h>

int main(){
 int num;
#pragma omp parallel num_threads(10)
 {
 int id = omp_get_thread_num();
 printf("Hello from thread %d\n",id);
 if(id==0) num = omp_get_num_threads();
 int i;
 #pragma omp critical
 {
 printf("thread  %d entered the critical region", omp_get_thread_num());
 for(i=0;i<id;i++)
 printf("%d ",id);
 printf("\n");
 }
}
return 0;
}
