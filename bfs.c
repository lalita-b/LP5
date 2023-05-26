#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define MAX_SIZE 100

// Function to perform parallel BFS
void parallelBFS(int adj[][MAX_SIZE], int n, int s, int num_threads) {
    // Create a boolean array to track visited vertices
    bool visited[MAX_SIZE] = {false};

    // Create a queue for BFS
    int queue[MAX_SIZE];
    int front = 0, rear = 0;

    // Mark the starting vertex as visited and enqueue it
    visited[s] = true;
    queue[rear++] = s;

    while (front != rear) {
        // Dequeue a vertex from the queue
        int u = queue[front++];

        printf("%d ", u);

        // Get all adjacent vertices of the dequeued vertex u
        // If an adjacent vertex has not been visited, then mark it visited and enqueue it
        #pragma omp parallel for num_threads(num_threads)
        for (int v = 0; v < n; v++) {
            if (adj[u][v] && !visited[v]) {
                visited[v] = true;
                queue[rear++] = v;
            }
        }
    }
}

int main() {
    int n, s; // Number of vertices and starting vertex
    int adj[MAX_SIZE][MAX_SIZE];

    printf("Enter the number of vertices: ");
    scanf("%d", &n);

    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &adj[i][j]);
        }
    }

    printf("Enter the starting vertex: ");
    scanf("%d", &s);

    int num_threads;
    printf("Enter the number of threads: ");
    scanf("%d", &num_threads);

    // Perform parallel BFS
    parallelBFS(adj, n, s, num_threads);

    return 0;
}
