#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define MAX_SIZE 100

// Function to perform parallel DFS
void parallelDFS(int adj[][MAX_SIZE], int n, int v, bool visited[]) {
    printf("%d ", v);

    // Mark the current vertex as visited
    visited[v] = true;

    // Recur for all the adjacent vertices of the current vertex
    #pragma omp parallel for num_threads(omp_get_num_threads())
    for (int i = 0; i < n; i++) {
        if (adj[v][i] && !visited[i]) {
            parallelDFS(adj, n, i, visited);
        }
    }
}

int main() {
    int n; // Number of vertices
    int adj[MAX_SIZE][MAX_SIZE];

    printf("Enter the number of vertices: ");
    scanf("%d", &n);

    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &adj[i][j]);
        }
    }

    int num_threads;
    printf("Enter the number of threads: ");
    scanf("%d", &num_threads);

    // Create a boolean array to track visited vertices
    bool visited[MAX_SIZE] = {false};

    // Perform parallel DFS for each unvisited vertex
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            parallelDFS(adj, n, i, visited);
        }
    }

    return 0;
}
