#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

void parallelBubbleSort(int arr[], int n) {
    bool swapped = true;
    int i, j;

    #pragma omp parallel num_threads(omp_get_num_threads())
    {
        while (swapped) {
            swapped = false;

            #pragma omp for
            for (i = 0; i < n - 1; i++) {
                if (arr[i] > arr[i + 1]) {
                    int temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                    swapped = true;
                }
            }
        }
    }
}

int main() {
    int n; // Number of elements
    int arr[100]; // Array to be sorted

    printf("Enter the number of elements: ");
    scanf("%d", &n);

    printf("Enter the elements:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    // Perform parallel Bubble Sort
    parallelBubbleSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
