/*
 * quickSort.cu
 * Author: Marius Rejdak
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_utils.h"
#include "gpuqsortlib/gpuqsort.cu"
#include "utils.h"

// program main
int main(int argc, char** argv)
{
    void* h_mem;

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);

    srand(time(NULL));

    printf("Quick sort\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(unsigned int));

    for (int64_t size = 4000000; size <= 40000000; size+=4000000)    
    {
        int32_t N = size / sizeof(Element);
        clock_t t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i)
        {
            clock_t t;
            // N = 30;
            init_values((Element*)h_mem, N);
            // print_int_array((Element*)h_mem, N);

            gpuqsort((unsigned int*)h_mem, N, &t);
            gpuErrchk(cudaPeekAtLastError());

            // print_int_array((Element*)h_mem, N);
            assert(is_int_array_sorted((Element*)h_mem, N, false));
            t_sum += t;
        }
        t_sum /= NUM_PASSES;
        double ms = (double)t_sum/(CLOCKS_PER_SEC/1000);

        printf("%ld, %d, %fms\n", N, t_sum, ms);
        // printf("%ld,%ld\n", N, t_sum);
    }

    free(h_mem);

    return 0;
}
