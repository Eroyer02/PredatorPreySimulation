/**
 * @file predprey.cu
 * @brief Predator-prey model using CUDA with timing.
 *
 * @author Ethan A. Royer(eroyer21@georgefox.edu
 */

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <string>
#include <ctime>
#include <chrono>

// define cell types in the grid
enum class CellType { EMPTY = 0, PREY, PREDATOR };

// struct representing each grid cell
struct Cell
        {
    CellType type; // Type of cell: EMPTY, PREY, or PREDATOR
    int age;       // Age of the entity (negative for PREDATOR, positive for PREY)

    // Constructor to initialize a cell with a type and age
    __device__ __host__ Cell(CellType t = CellType::EMPTY, int a = 0) : type(t), age(a) {}
};

/**
 * Kernel to initialize the grid
 *
 * @param grid the passed in grid
 * @param X the amount of columns
 * @param Y the amount of rows
 * @param PRED the percentage of predators
 * @param PREY the percentage of prey
 * @param states the current state
 */
__global__ void initKernel(Cell* grid, int X, int Y, float PRED,
                           float PREY, curandState* states)
{
    int total_cells = X * Y;
    int total_threads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // grid loop
    for (int i = idx; i < total_cells; i += total_threads)
    {
        // initialize a random state for the current cell
        curandState localState;
        curand_init(clock64() + i, i, 0, &localState);

        // generate a random number and assign cell type based on probabilities
        float rand_num = curand_uniform(&localState);
        if (rand_num < PRED)
        {
            grid[i] = Cell{CellType::PREDATOR, -1};
        }
        else if (rand_num < PRED + PREY)
        {
            grid[i] = Cell{CellType::PREY, 1};
        }
        else
        {
            grid[i] = Cell{CellType::EMPTY, 0};
        }

        // save the random state for reuse
        states[i] = localState;
    }
}

/**
 * The kernel to update the grid
 *
 * @param grid the passed in grid
 * @param new_grid the updated grid
 * @param states the current state
 * @param X the amount of columns
 * @param Y the amount of rows
 */
__global__ void updateKernel(Cell* grid, Cell* new_grid, curandState* states, int X, int Y)
{
    // initial variables
    int total_cells = X * Y;
    int total_threads = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // grid loop
    for (int i = idx; i < total_cells; i += total_threads)
    {
        int x = i % X;
        int y = i / X;

        Cell current_cell = grid[i];

        // neighbor statistics
        int pred_neighbors = 0;
        int prey_neighbors = 0;

        // loop through all neighbors
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                // skip the cell itself
                if (dx != 0 || dy != 0) {
                    int nx = (x + dx + X) % X; // Wrap around x-axis
                    int ny = (y + dy + Y) % Y; // Wrap around y-axis

                    // get the neighbor cell
                    Cell neighbor = grid[ny * X + nx];
                    if (neighbor.type == CellType::PREDATOR)
                    {
                        pred_neighbors++;
                    } else if (neighbor.type == CellType::PREY)
                    {
                        prey_neighbors++;
                    }
                }
            }
        }

        // update cell based on its type and neighbors
        if (current_cell.type == CellType::EMPTY)
        {
            // empty cell can become prey or predator if conditions are met
            if (prey_neighbors >= 4 && pred_neighbors < 3)
            {
                new_grid[i] = Cell{CellType::PREY, 1};
            }
            else if (pred_neighbors >= 4)
            {
                new_grid[i] = Cell{CellType::PREDATOR, -1};
            }
            else
            {
                new_grid[i] = Cell{CellType::EMPTY, 0};
            }
        }
        else if (current_cell.type == CellType::PREDATOR)
        {
            // predators age and die if they don't find prey
            if (prey_neighbors == 0 || current_cell.age < -20)
            {
                new_grid[i] = Cell{CellType::EMPTY, 0};
            }
            else
            {
                new_grid[i] = Cell{CellType::PREDATOR, current_cell.age - 1};
            }
        }
        else if (current_cell.type == CellType::PREY)
        {
            // prey age and can die from overcrowding
            if (prey_neighbors >= 8 || pred_neighbors >= 5)
            {
                new_grid[i] = Cell{CellType::EMPTY, 0};
            }
            else
            {
                new_grid[i] = Cell{CellType::PREY, current_cell.age + 1};
            }
        }
    }
}

// main function
int main(int argc, char* argv[])
{
    // validate command-line arguments
    if (argc < 7 || argc > 9)
    {
        std::cerr << "Usage: " << argv[0] << " X Y PRED PREY G U [T] [M]\n";
        return EXIT_FAILURE;
    }

    // parse arguments
    int X = std::atoi(argv[1]);
    int Y = std::atoi(argv[2]);
    float PRED = std::atof(argv[3]);
    float PREY = std::atof(argv[4]);
    int G = std::atoi(argv[5]);
    int U = std::atoi(argv[6]);
    int T;
    if (argc > 7)
    {
        T = std::atoi(argv[7]);
    }
    else
    {
        T = 1;
    }
    int M;
    if (argc > 8)
    {
        M = std::atoi(argv[8]);
    }
    else
    {
        M = 1;
    }

    // validate T and M
    if (T <= 0 || M <= 0)
    {
        std::cerr << "Threads per block (T) and number of blocks "
                     "(M) must be positive integers.\n";
        return EXIT_FAILURE;
    }

    // start total execution timer
    auto start_total = std::chrono::high_resolution_clock::now();

    // compute grid size and allocate memory
    size_t grid_size = X * Y * sizeof(Cell);
    Cell* h_grid = new Cell[X * Y];
    Cell *d_grid, *d_new_grid;
    curandState* d_states;

    // error checking
    cudaError_t err;
    err = cudaMalloc(&d_grid, grid_size);
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating memory for d_grid: "
        << cudaGetErrorString(err) << std::endl;
        delete[] h_grid;
        return EXIT_FAILURE;
    }
    err = cudaMalloc(&d_new_grid, grid_size);
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating memory for d_new_grid: "
        << cudaGetErrorString(err) << std::endl;
        cudaFree(d_grid);
        delete[] h_grid;
        return EXIT_FAILURE;
    }
    err = cudaMalloc(&d_states, X * Y * sizeof(curandState));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating memory for d_states: " <<
        cudaGetErrorString(err) << std::endl;
        cudaFree(d_grid);
        cudaFree(d_new_grid);
        delete[] h_grid;
        return EXIT_FAILURE;
    }

    // create CUDA events for timing kernels
    cudaEvent_t start_init, stop_init, start_update, stop_update;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventCreate(&start_update);
    cudaEventCreate(&stop_update);

    // start timing for initialization kernel
    cudaEventRecord(start_init);

    // initialize the grid on the GPU
    initKernel<<<M, T>>>(d_grid, X, Y, PRED, PREY, d_states);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error launching initKernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_grid);
        cudaFree(d_new_grid);
        cudaFree(d_states);
        delete[] h_grid;
        return EXIT_FAILURE;
    }
    cudaEventRecord(stop_init);
    cudaEventSynchronize(stop_init);

    // compute initialization kernel execution time
    float init_time = 0.0f;
    cudaEventElapsedTime(&init_time, start_init, stop_init);
    float total_update_time = 0.0f;

    // simulation loop
    for (int gen = 0; gen <= G; gen++)
    {
        // Start timing for update kernel
        cudaEventRecord(start_update);

        // Run the update kernel
        updateKernel<<<M, T>>>(d_grid, d_new_grid, d_states, X, Y);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Error launching updateKernel at generation " << gen << ": "
            << cudaGetErrorString(err) << std::endl;
            cudaFree(d_grid);
            cudaFree(d_new_grid);
            cudaFree(d_states);
            delete[] h_grid;
            return EXIT_FAILURE;
        }

        // stop timing for update kernel
        cudaEventRecord(stop_update);
        cudaEventSynchronize(stop_update);

        // accumulate update kernel execution time
        float update_time = 0.0f;
        cudaEventElapsedTime(&update_time, start_update, stop_update);
        total_update_time += update_time;

        // swap grids
        Cell* temp = d_grid;
        d_grid = d_new_grid;
        d_new_grid = temp;

        // output grid to CSV at intervals
        if (gen % U == 0 || gen == G)
        {
            err = cudaMemcpy(h_grid, d_grid, grid_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "Error copying data from device to host: "
                << cudaGetErrorString(err) << std::endl;
                cudaFree(d_grid);
                cudaFree(d_new_grid);
                cudaFree(d_states);
                delete[] h_grid;
                return EXIT_FAILURE;
            }
            std::ofstream outfile("generation_" + std::to_string(gen) + ".csv");
            if (!outfile.is_open())
            {
                std::cerr << "Error opening output file for generation " << gen << std::endl;
                cudaFree(d_grid);
                cudaFree(d_new_grid);
                cudaFree(d_states);
                delete[] h_grid;
                return EXIT_FAILURE;
            }

            // write to CSV
            for (int y = 0; y < Y; y++)
            {
                for (int x = 0; x < X; x++)
                {
                    char cell_char;
                    CellType cell_type = h_grid[y * X + x].type;
                    if (cell_type == CellType::PREDATOR)
                    {
                        cell_char = 'd'; // Predator
                    }
                    else if (cell_type == CellType::PREY)
                    {
                        cell_char = 'y'; // Prey
                    }
                    else
                    {
                        cell_char = 'e'; // Empty
                    }
                    outfile << cell_char;
                    if (x < X - 1)
                    {
                        outfile << ",";
                    }
                }
                outfile << "\n";
            }
            outfile.close();
        }
    }

    // end total execution timer
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total - start_total;

    // output timing information
    std::cout << "Initialization Kernel Execution Time: " << init_time << " ms\n";
    std::cout << "Total Update Kernels Execution Time: " << total_update_time << " ms\n";
    std::cout << "Total Execution Time: " << total_duration.count() * 1000 << " ms\n";

    // clean up CUDA events
    cudaEventDestroy(start_init);
    cudaEventDestroy(stop_init);
    cudaEventDestroy(start_update);
    cudaEventDestroy(stop_update);

    // free resources
    delete[] h_grid;
    cudaFree(d_grid);
    cudaFree(d_new_grid);
    cudaFree(d_states);

    return 0;
}
