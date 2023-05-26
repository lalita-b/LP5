#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 10

int main()
{
    int arr[SIZE];
    int i, j, temp;
    double start_time, end_time;

    // Initialize the array with random values
    for(i = 0; i < SIZE; i++)
    {
        arr[i] = rand() % SIZE;
    }

    // Start the timer
    start_time = omp_get_wtime();

    // Perform the parallel bubble sort
    #pragma omp parallel shared(arr)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = SIZE / num_threads;
        int start_index = tid * chunk_size;
        int end_index = start_index + chunk_size;

        // Each thread performs its own bubble sort on its portion of the array
        for(i = start_index; i < end_index; i++)
        {
            for(j = start_index; j < end_index - 1; j++)
            {
                if(arr[j] > arr[j+1])
                {
                    temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
        }

        // Wait for all threads to finish their work
        #pragma omp barrier

        // Merge the sorted arrays using parallel merge algorithm
        int left_start = tid * chunk_size;
        int left_end = left_start + chunk_size - 1;
        int right_start = left_end + 1;
        int right_end = (tid == num_threads - 1) ? SIZE - 1 : right_start + chunk_size - 1;

        int merged_arr[SIZE];
        int k = 0;

        while(left_start <= left_end && right_start <= right_end)
        {
            if(arr[left_start] < arr[right_start])
            {
                merged_arr[k++] = arr[left_start++];
            }
            else
            {
                merged_arr[k++] = arr[right_start++];
            }
        }

        while(left_start <= left_end)
        {
            merged_arr[k++] = arr[left_start++];
        }

        while(right_start <= right_end)
        {
            merged_arr[k++] = arr[right_start++];
        }

        for(i = 0; i < k; i++)
        {
            arr[start_index + i] = merged_arr[i];
        }

        // Wait for all threads to finish merging their arrays
        #pragma omp barrier
    }

    // End the timer
    end_time = omp_get_wtime();

    // Print the sorted array
    printf("Sorted Array:\n");
    for(i = 0; i < SIZE; i++)
    {
        printf("%d ", arr[i]);
    }

    // Print the elapsed time
    printf("\nElapsed Time: %f seconds\n", end_time - start_time);

    return 0;
}
