#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void merge(int *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    int L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++) {
        L[i] = arr[l + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[m + 1 + j];
    }

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void merge_sort(int *arr, int l, int r, int num_threads) {
    if (l < r) {
        int m = l + (r - l) / 2;

        /* Recursive call to merge_sort on left subarray */
        #pragma omp task shared(arr)
        merge_sort(arr, l, m, num_threads);

        /* Recursive call to merge_sort on right subarray */
        #pragma omp task shared(arr)
        merge_sort(arr, m + 1, r, num_threads);

        /* Merge the sorted left and right subarrays */
        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}

int main() {
    int n = 10;
    int arr[] = { 8, 7, 6, 5, 4, 3, 2, 1, 10, 9 };
    int num_threads = 2;

    /* Set the number of threads */
    omp_set_num_threads(num_threads);

    /* Sort the array using parallel merge sort */
    #pragma omp parallel shared(arr)
    #pragma omp single
    merge_sort(arr, 0, n - 1, num_threads);

    /* Print the sorted array */
    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
