/*
 * thrustSort.cu
 * Author: Marius Rejdak
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

extern "C"
{
#include "utils.h"
}

using namespace std;

int my_compare(const Element &a, const Element &b)
{
    return a.k < b.k;
}

// program main
int main(int argc, char **argv)
{
    void *h_mem;

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);

    srand(time(NULL));

    printf("CPU STL sort\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for (int64_t size = 4000000; size <= 40000000; size+=4000000)    
    {
        int32_t N = size / sizeof(Element);
        clock_t t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i)
        {
            clock_t t1;
            init_values((Element *)h_mem, N);

            t1 = clock();
            sort((Element *)h_mem, (Element *)h_mem + N, my_compare);
            t_sum += clock() - t1;

            assert(is_int_array_sorted((Element *)h_mem, N, false));
        }
        t_sum /= NUM_PASSES;
        double ms = (double)t_sum / (CLOCKS_PER_SEC / 1000);

        printf("%ld, %d, %fms\n", N, t_sum, ms);
    }

    free(h_mem);

    return 0;
}
