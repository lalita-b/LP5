#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 1000000
#define MAX_NUM 10000

void quicksort(int* array, int left, int right) {
    if (left >= right) return;
    int pivot = array[(left + right) / 2];
    int i = left - 1, j = right + 1;
    while (i < j) {
        do i++; while (array[i] < pivot);
        do j--; while (array[j] > pivot);
        if (i < j) {
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    quicksort(array, left, j);
    quicksort(array, j + 1, right);
}

int main(int argc, char** argv) {
    int rank, size;
    int* array;
    int chunk_size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Generate random input array
        srand(time(NULL));
        array = new int[ARRAY_SIZE];
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = rand() % MAX_NUM;
        }
        start_time = MPI_Wtime();
    }

    // Broadcast input array to all processes
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int* chunk = new int[chunk_size];
    MPI_Bcast(chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort local chunk using Quicksort
    quicksort(chunk, 0, chunk_size - 1);

    // Merge sorted chunks using merge sort
    int merge_size = chunk_size * size;
    int* merge_array = NULL;
    if (rank == 0) merge_array = new int[merge_size];
    MPI_Gather(chunk, chunk_size, MPI_INT, merge_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int* temp = new int[merge_size];
        int i = 0, j = merge_size / 2, k = 0;
        while (i < merge_size / 2 && j < merge_size) {
            if (merge_array[i] <= merge_array[j]) {
                temp[k] = merge_array[i];
                i++;
            }
            else {
                temp[k] = merge_array[j];
                j++;
            }
            k++;
        }
        while (i < merge_size / 2) {
            temp[k] = merge_array[i];
            i++; k++;
        }
        while (j < merge_size) {
            temp[k] = merge_array[j];
            j++; k++;
        }
        end_time = MPI_Wtime();
        printf("Time taken = %f seconds\n", end_time - start_time);
        delete[] array;
        delete[] merge_array;
        delete[] temp;
    }

    delete[] chunk;
    MPI_Finalize();
    return 0;
}
