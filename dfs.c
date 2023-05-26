#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#define N 6

// Graph data structure
struct Graph {
    int num_vertices;
    int **adj_matrix;
};

// Function to create a new graph
struct Graph* createGraph(int num_vertices) {
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->num_vertices = num_vertices;
    graph->adj_matrix = (int**)malloc(num_vertices * sizeof(int*));
    for(int i = 0; i < num_vertices; i++) {
        graph->adj_matrix[i] = (int*)calloc(num_vertices, sizeof(int));
    }
    return graph;
}

// Function to add an edge to the graph
void addEdge(struct Graph* graph, int u, int v) {
    graph->adj_matrix[u][v] = 1;
    graph->adj_matrix[v][u] = 1;
}

// Function to perform parallel DFS
void parallelDFS(struct Graph* graph, int start_vertex, bool visited[]) {
    // Mark the current node as visited and print it
    visited[start_vertex] = true;
    printf("%d ", start_vertex);

    // Get the number of threads available
    int num_threads = omp_get_num_threads();

    // Recur for all the vertices adjacent to this vertex
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < graph->num_vertices; i++) {
        if(graph->adj_matrix[start_vertex][i] && !visited[i]) {
            parallelDFS(graph, i, visited);
        }
    }
}

int main() {
    // Create a graph
    struct Graph* graph = createGraph(N);

    // Add edges to the graph
    addEdge(graph, 0, 1);
    addEdge(graph, 0, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 2, 4);
    addEdge(graph, 3, 5);
    addEdge(graph, 4, 5);

    // Starting vertex for DFS
    int start_vertex = 0;

    // Create a boolean array to track visited vertices
    bool* visited = (bool*)calloc(N, sizeof(bool));

    // Perform parallel DFS
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelDFS(graph, start_vertex, visited);
        }
    }

    free(visited);
    for(int i = 0; i < N; i++) {
        free(graph->adj_matrix[i]);
    }
    free(graph->adj_matrix);
    free(graph);

    return 0;
}
