#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

int main() {
    int i, min_val, max_val, sum = 0;
    int arr[N];

    // Initialize array with random values
    for (i = 0; i < N; i++) {
        arr[i] = rand() % 100;
    }

    // Find minimum value using parallel reduction
    #pragma omp parallel for reduction(min:min_val)
    for (i = 0; i < N; i++) {
        if (i == 0 || arr[i] < min_val) {
            min_val = arr[i];
        }
    }
    printf("Minimum value: %d\n", min_val);

    // Find maximum value using parallel reduction
    #pragma omp parallel for reduction(max:max_val)
    for (i = 0; i < N; i++) {
        if (i == 0 || arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    printf("Maximum value: %d\n", max_val);

    // Find average value using parallel reduction
    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < N; i++) {
        sum += arr[i];
    }
    double avg = (double) sum / N;
    printf("Average value: %.2f\n", avg);

    return 0;
}
