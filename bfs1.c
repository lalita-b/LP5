#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

// Function to add an edge to the graph
void addEdge(int** adj, int u, int v) {
    adj[u][v] = 1;
    adj[v][u] = 1;
}

// Function to perform parallel BFS
void parallelBFS(int** adj, int s, int n, int num_threads) {
    // Create a boolean array to track visited vertices
    bool* visited = (bool*) malloc(n * sizeof(bool));
    for(int i = 0; i < n; i++) {
        visited[i] = false;
    }

    // Create a queue for BFS
    int* q = (int*) malloc(n * sizeof(int));
    int front = 0;
    int rear = 0;

    // Mark the current node as visited and enqueue it
    visited[s] = true;
    q[rear++] = s;

    // Set number of threads
    omp_set_num_threads(num_threads);

    while(front != rear) {
        // Dequeue a vertex from queue and print it
        int u = q[front++];
        printf("%d ", u);

        // Get all adjacent vertices of the dequeued vertex u
        // If an adjacent vertex has not been visited, then mark it
        // visited and enqueue it
        #pragma omp parallel for
        for(int i = 0; i < n; i++) {
            if(adj[u][i] == 1 && visited[i] == false) {
                visited[i] = true;
                q[rear++] = i;
            }
        }
    }

    free(visited);
    free(q);
}

int main() {
    // Number of vertices
    int n = 6;

    // Create an adjacency matrix for the graph
    int** adj = (int**) malloc(n * sizeof(int*));
    for(int i = 0; i < n; i++) {
        adj[i] = (int*) malloc(n * sizeof(int));
        for(int j = 0; j < n; j++) {
            adj[i][j] = 0;
        }
    }
    addEdge(adj, 0, 1);
    addEdge(adj, 0, 2);
    addEdge(adj, 1, 3);
    addEdge(adj, 2, 4);
    addEdge(adj, 3, 5);
    addEdge(adj, 4, 5);

    // Starting vertex for BFS
    int s = 0;

    // Number of threads to use
    int num_threads = 4;

    // Perform parallel BFS
    parallelBFS(adj, s, n, num_threads);

    for(int i = 0; i < n; i++) {
        free(adj[i]);
    }
    free(adj);

    return 0;
}
