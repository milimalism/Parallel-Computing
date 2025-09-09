#include <omp.h>
#include <stdio.h>

int main(){
 int num;
 #pragma omp parallel
 {
 int id = omp_get_thread_num();
 printf("Hello from thread %d\n",id);
 if(id==0) num = omp_get_num_threads();
 int i;
 #pragma omp barrier
 { 
#pragma omp critical

 {
 for(i=0;i<id;i++) printf("%d ",id);
 printf("\n");
 }
 }
}
return 0;
}
