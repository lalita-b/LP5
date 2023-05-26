#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void calculateMinMaxAvg(int arr[], int n, int* min, int* max, double* avg) {
    int localMin = arr[0];
    int localMax = arr[0];
    double localSum = 0.0;

    #pragma omp parallel for reduction(min:localMin) reduction(max:localMax) reduction(+:localSum)
    for (int i = 0; i < n; i++) {
        localMin = (arr[i] < localMin) ? arr[i] : localMin;
        localMax = (arr[i] > localMax) ? arr[i] : localMax;
        localSum += arr[i];
    }

    *min = localMin;
    *max = localMax;
    *avg = localSum / n;
}

int main() {
    int n; // Number of elements
    int arr[100]; // Array of elements

    printf("Enter the number of elements: ");
    scanf("%d", &n);

    printf("Enter the elements:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    int min, max;
    double avg;

    // Calculate min, max, and average using parallel reduction
    calculateMinMaxAvg(arr, n, &min, &max, &avg);

    printf("Minimum: %d\n", min);
    printf("Maximum: %d\n", max);
    printf("Average: %.2lf\n", avg);

    return 0;
}
