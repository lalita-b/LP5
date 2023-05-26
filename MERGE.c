#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void merge(int arr[], int left[], int right[], int leftSize, int rightSize) {
    int i = 0, j = 0, k = 0;

    while (i < leftSize && j < rightSize) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }

    while (i < leftSize) {
        arr[k++] = left[i++];
    }

    while (j < rightSize) {
        arr[k++] = right[j++];
    }
}

void parallelMergeSort(int arr[], int n) {
    if (n <= 1) {
        return;
    }

    int mid = n / 2;
    int* left = (int*)malloc(mid * sizeof(int));
    int* right = (int*)malloc((n - mid) * sizeof(int));

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            parallelMergeSort(left, mid);
        }
        #pragma omp section
        {
            parallelMergeSort(right, n - mid);
        }
    }

    merge(arr, left, right, mid, n - mid);

    free(left);
    free(right);
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

    // Perform parallel Merge Sort
    parallelMergeSort(arr, n);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
