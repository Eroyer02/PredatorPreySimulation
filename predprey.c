/**
 * @file predprey.c
 * @brief Predator-prey model using MPI.
 *
 * @author Ethan A. Royer (eroyer21@georgefox.edu)
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <stddef.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#define EMPTY 0
#define PY 1
#define PREDATOR 2

// struct for Cell
typedef struct
{
    int type;
    int age;
} Cell;

// checkpoint flag
volatile int checkpoint_flag = 0;

/**
 * The function to initialize the grid.
 *
 * @param grid the grid to be initialized
 * @param X the amount of columns
 * @param Y the amount of rows
 * @param pred_ratio the ratio of predators
 * @param prey_ratio the ratio of prey
 * @param rank the rank the machine is working on
 */
void initialize_grid(Cell** grid, int X, int Y, float pred_ratio, float prey_ratio);

/**
 * Updates the grid with the updated data.
 *
 * @param grid the original grid
 * @param new_grid the grid to be updated
 * @param start_row the row to start
 * @param end_row the row to end
 * @param X the amount of columns
 * @param Y the amount of rows
 */
void update_grid_section(Cell** grid, Cell** new_grid, int start_row, int end_row, int X, int Y);

/**
 * Exchanges the ghost rows between processes.
 *
 * @param grid the grid of the current population
 * @param X the amount of columns
 * @param Y the amount of rows
 * @param rank which rank the process is
 * @param size the size of data that process is dealing with
 * @param MPI_Cell the cell struct passed into the method
 */
void exchange_ghost_rows(Cell** grid, int X, int Y, int rank, int size, MPI_Datatype MPI_Cell);

/**
 * Writes the grid to the file.
 *
 * @param grid the grid passed in
 * @param X the amount of columns
 * @param Y the amount of rows
 * @param generation which generation is being written
 * @param rank which machine is writing the generation
 */
void write_grid_to_file(Cell** grid, int X, int Y, int generation, int rank);

/**
 * Writes the checkpoint file.
 *
 * @param filename the passed in file name
 * @param grid the passed in grid
 * @param X the amount of columns
 * @param Y the amount of rows
 * @param generation the generation to print out on
 * @param PRED the percentage of predators
 * @param PREY the percentage of prey
 */
void write_checkpoint(const char *filename, Cell **grid, int X, int Y, int generation,
                      float PRED, float PREY);

/**
 * Loads the checkpoint file.
 *
 * @param filename the passed in filename
 * @param grid the current grid
 * @param X the amount of columns
 * @param Y the amount of rows
 * @param G the amount of generations
 * @param U the generation to print out on
 * @param PRED the percentage of predators
 * @param PREY the percentage of prey
 * @return
 */
int load_checkpoint(const char *filename, Cell ***grid, int *X, int *Y,
                    int *G, float *PRED, float *PREY);

/**
 * Handles a signal interrupt.
 *
 * @param sig the signal passed in
 */
void handle_sigint(int sig);

/**
 * The main method which will execute the simulation.
 *
 * @param argc the argument count
 * @param argv the actual command line arguments passed in
 * @return 0 or 1 depending on success or failure
 */
int main(int argc, char* argv[])
{
    // initializes MPI and signal
    MPI_Init(&argc, &argv);
    signal(SIGINT, handle_sigint);

    // initializes the size and rank and starts a timer
    int size;
    int rank;
    int gen = 0;
    int X = 0;
    int Y = 0;
    float PRED = 0.0;
    float PREY = 0.0;
    int G = 0;
    int U = 0;
    Cell **grid = NULL;

    // initializes the mpi size rank and time
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double start_time = MPI_Wtime();

    // checks correct argument count
    if (rank == 0 && access("checkpoint.bin", F_OK) == 0)
    {
        if (load_checkpoint("checkpoint.bin", &grid, &X, &Y, &G, &U, &PRED, &PREY))
        {
            printf("Checkpoint loaded. Resuming from generation %d.\n", gen);
        }
        else
        {
            printf("Failed to load checkpoint. Starting from scratch.\n");
        }
    }
    else
    {
        // command-line arguments
        if (argc != 7)
        {
            if (rank == 0) printf("Incorrect argument count\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        X = atoi(argv[1]);
        Y = atoi(argv[2]);
        PRED = atof(argv[3]);
        PREY = atof(argv[4]);
        G = atoi(argv[5]);
        U = atoi(argv[6]);
    }

    // validates input parameters
    if (X <= 2 || Y <= 2 || PRED < 0 || PRED > 1 || PREY < 0 || PREY > 1
            || PRED + PREY > 1 || G < 1 || U < 0)
    {
        if (rank == 0)
        {
            printf("Invalid parameters\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // broadcasts data to ranks
    MPI_Bcast(&X, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PRED, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&PREY, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&G, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&U, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // defines the MPI datatype for the cell struct
    MPI_Datatype MPI_Cell;
    int lengths[2] = {1, 1};
    MPI_Aint offsets[2] = {offsetof(Cell, type), offsetof(Cell, age)};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Type_create_struct(2, lengths, offsets, types, &MPI_Cell);
    MPI_Type_commit(&MPI_Cell);

    // random seed initialization
    srand(time(NULL) + rank);

    // allocate the grid for the current process
    int rows_per_proc = Y / size;
    int remainder = Y % size;
    int start_row = rank * rows_per_proc;
    if (rank < remainder)
    {
        start_row += rank;
    }
    else
    {
        start_row += remainder;
    }

    // establishes the last row
    int end_row = start_row + rows_per_proc;
    if (rank < remainder)
    {
        end_row += 1;
    }

    // creates space in memory for the grids
    grid = (Cell **) malloc((end_row - start_row + 2) * sizeof(Cell *));
    Cell **new_grid = (Cell **) malloc((end_row - start_row + 2) * sizeof(Cell *));
    for (int i = 0; i < (end_row - start_row + 2); i++)
    {
        grid[i] = (Cell *) malloc(X * sizeof(Cell));
        new_grid[i] = (Cell *) malloc(X * sizeof(Cell));
    }

    // initialize the grid with predators and prey
    initialize_grid(grid, X, end_row - start_row, PRED, PREY);

    // start simulation
    for (int generation = 0; generation <= G; generation++)
    {
        if (checkpoint_flag)
        {
            write_checkpoint("checkpoint.bin", grid, X, end_row - start_row, gen, PRED, PREY);
            checkpoint_flag = 0; // Reset the flag
            if (rank == 0) {
                printf("Checkpoint written at generation %d.\n", gen);
            }
        }
        // exchange ghost rows (neighboring rows) between processes
        exchange_ghost_rows(grid, X, end_row - start_row, rank, size, MPI_Cell);
        // update the grid section handled by this process
        update_grid_section(grid, new_grid, 1,
                            end_row - start_row, X, end_row - start_row);

        // swap grids for next generation
        Cell **temp = grid;
        grid = new_grid;
        new_grid = temp;

        // output the grid at specified generations
        if (generation % U == 0 || generation == G)
        {
            write_grid_to_file(grid, X, end_row - start_row, generation, rank);
        }
        if (generation == G && rank == 0)
        {
            remove("checkpoint.bin"); // Clean up checkpoint file on completion
        }
    }
    // end timing
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    if (rank == 0)
    {
        printf("Simulation took %f seconds\n", elapsed_time);
    }

    // free memory
    for (int i = 0; i < (end_row - start_row + 2); i++)
    {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);

    // free datatype and finalize MPI
    MPI_Type_free(&MPI_Cell);
    MPI_Finalize();
    return 0;
}

// signal handler for checkpointing
void handle_sigint(int sig)
{
        checkpoint_flag = 1; // Set the checkpoint flag
}
// function to initialize the grid with predators, prey, and empty cells
void initialize_grid(Cell** grid, int X, int Y, float pred_ratio, float prey_ratio)
{
    // initial variables
    int total_cells = X * Y;
    int initial_predators = (int)(total_cells * pred_ratio);
    int initial_prey = (int)(total_cells * prey_ratio);

    // initialize the grid with empty cells
    for (int i = 0; i < Y; i++)
    {
        for (int j = 0; j < X; j++)
        {
            grid[i + 1][j].type = EMPTY;
            grid[i + 1][j].age = 0;
        }
    }

    // randomly place predators
    while (initial_predators > 0)
    {
        int x = rand() % X;
        int y = rand() % Y;
        if (grid[y + 1][x].type == EMPTY)
        {
            grid[y + 1][x].type = PREDATOR;
            grid[y + 1][x].age = -1;
            initial_predators--;
        }
    }

    // randomly place prey
    while (initial_prey > 0)
    {
        int x = rand() % X;
        int y = rand() % Y;
        if (grid[y + 1][x].type == EMPTY)
        {
            grid[y + 1][x].type = PY;
            grid[y + 1][x].age = 1;
            initial_prey--;
        }
    }
}

// function to update a section of the grid
void update_grid_section(Cell** grid, Cell** new_grid, int start_row, int end_row, int X, int Y)
{
    for (int i = start_row; i <= end_row; i++)
    {
        for (int j = 0; j < X; j++)
        {
            int pred_neighbors = 0;
            int prey_neighbors = 0;
            int pred_breeding_neighbors = 0;
            int prey_breeding_neighbors = 0;

            // count neighbors
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    if (dx != 0 || dy != 0)
                    {
                        int nx = j + dx;
                        int ny = i + dy;

                        if (nx >= 0 && nx < X && ny >= 0 && ny < Y)
                        {
                            // checks predator neighbor
                            if (grid[ny][nx].type == PREDATOR)
                            {
                                pred_neighbors++;
                                if (grid[ny][nx].age <= -3)
                                {
                                    pred_breeding_neighbors++;
                                }
                            }
                            // checks prey neighbor
                            else if (grid[ny][nx].type == PY)
                            {
                                prey_neighbors++;
                                if (grid[ny][nx].age >= 2)
                                {
                                    prey_breeding_neighbors++;
                                }
                            }
                        }
                    }
                }
            }
            // empty logic
            if (grid[i][j].type == EMPTY)
            {
                if (pred_neighbors >= 4 && pred_breeding_neighbors >= 3 && prey_neighbors < 4)
                {
                    new_grid[i][j].type = PREDATOR;
                    new_grid[i][j].age = -1;
                }
                else if (prey_neighbors >= 4 && prey_breeding_neighbors >= 3
                && pred_neighbors < 4)
                {
                    new_grid[i][j].type = PY;
                    new_grid[i][j].age = 1;
                }
                else
                {
                    new_grid[i][j].type = EMPTY;
                }
            }
            // predator logic
            else if (grid[i][j].type == PREDATOR)
            {
                if (grid[i][j].age < -20 || (pred_neighbors >= 6 && prey_neighbors == 0)
                || rand() % 64 == 0)
                {
                    new_grid[i][j].type = EMPTY;
                    new_grid[i][j].age = 0;
                }
                else
                {
                    new_grid[i][j].type = PREDATOR;
                    new_grid[i][j].age--;
                }
            }
            // prey logic
            else if (grid[i][j].type == PY)
            {
                if (grid[i][j].age > 10 || pred_neighbors >= 5 || prey_neighbors == 8)
                {
                    new_grid[i][j].type = EMPTY;
                    new_grid[i][j].age = 0;
                }
                else
                {
                    new_grid[i][j].type = PY;
                    new_grid[i][j].age++;
                }
            }
        }
    }
}

// function to exchange ghost rows (neighbors) between adjacent processes
void exchange_ghost_rows(Cell** grid, int X, int Y, int rank, int size, MPI_Datatype MPI_Cell)
{
    // stores status of MPI communication
    MPI_Status status;
    // if not the main rank send and receive the top ghost row
    // for the neighboring processes above
    if (rank != 0)
    {
        // send the second row
        MPI_Send(grid[1], X, MPI_Cell, rank - 1, 0, MPI_COMM_WORLD);
        // receive the first row
        MPI_Recv(grid[0], X, MPI_Cell, rank - 1, 0, MPI_COMM_WORLD, &status);
    }
    // send and receive the bottom ghost row for neighboring processes below
    if (rank != size - 1)
    {
        // send last row to process below
        MPI_Send(grid[Y], X, MPI_Cell, rank + 1, 0, MPI_COMM_WORLD);
        // receive process below
        MPI_Recv(grid[Y + 1], X, MPI_Cell, rank + 1, 0, MPI_COMM_WORLD, &status);
    }
}

// function to write the grid to a file for a specific generation
void write_grid_to_file(Cell** grid, int X, int Y, int generation, int rank)
{
    // sets up file in write mode
    char filename[256];
    snprintf(filename, sizeof(filename),
             "grid_gen_%d_rank_%d.txt", generation, rank);
    FILE* file = fopen(filename, "w");

    for (int i = 1; i <= Y; i++)
    {
        for (int j = 0; j < X; j++)
        {
            // writes predators
            if (grid[i][j].type == PREDATOR)
            {
                fprintf(file, "P ");
            }
            // writes prey
            else if (grid[i][j].type == PY)
            {
                fprintf(file, "Y ");
            }
            // writes empty
            else
            {
                fprintf(file, "E ");
            }
        }
        fprintf(file, "\n");
    }
    // closes file
    fclose(file);
}

// function to write the checkpoint to a file
void write_checkpoint(const char *filename, Cell **grid, int X, int Y, int generation, float PRED, float PREY)
{
    // creates a file and writes it
    FILE *file = fopen(filename, "wb");
    if (file)
    {
        fwrite(&X, sizeof(int), 1, file);
        fwrite(&Y, sizeof(int), 1, file);
        fwrite(&generation, sizeof(int), 1, file);
        fwrite(&PRED, sizeof(float), 1, file);
        fwrite(&PREY, sizeof(float), 1, file);
        fwrite(grid[0], sizeof(Cell), X * Y, file);
        fclose(file);
    }
}

// loads the checkpoint if checkpoint.bin is present
int load_checkpoint(const char *filename, Cell ***grid, int *X, int *Y,
                    int *G,float *PRED, float *PREY)
{
    // writes the checkpoint to the file to start again
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        return 0;
    }
    fread(X, sizeof(int), 1, file);
    fread(Y, sizeof(int), 1, file);
    fread(G, sizeof(int), 1, file);
    fread(PRED, sizeof(float), 1, file);
    fread(PREY, sizeof(float), 1, file);
    *grid = (Cell **)malloc((*Y) * sizeof(Cell *));
    for (int i = 0; i < *Y; i++)
    {
        (*grid)[i] = (Cell *)malloc(*X * sizeof(Cell));
    }
    fread((*grid)[0], sizeof(Cell), (*X) * (*Y), file);
    fclose(file);
    return 1;
}
