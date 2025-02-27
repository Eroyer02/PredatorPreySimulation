/**
 * @file predprey.cpp
 * @brief Predator-prey model with memory leak fixes and threading improvements.
 *
 * @author Ethan A. Royer (eroyer21@georgefox.edu)
 */

#include "predprey.h"

/**
 * @brief The predator prey simulation.
 *
 * @return The exit status of the program.
 */

int main(int argc, char* argv[])
{
    // argument count check
    if (argc != 7 && argc != 8)
    {
        std::cerr << "Incorrect argument count\n";
        return EXIT_FAILURE;
    }

    // population variables
    int X = std::atoi(argv[1]);
    int Y = std::atoi(argv[2]);
    float PRED = std::atof(argv[3]);
    float PREY = std::atof(argv[4]);
    int G = std::atoi(argv[5]);
    int U = std::atoi(argv[6]);
    int K = 0;

    // random generation set up
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distX(0, X - 1);
    std::uniform_int_distribution<> distY(0, Y - 1);
    std::uniform_int_distribution<> rand64(0, 63);

    // if no input 1 thread, if input get from command line
    if (argc == 7)
    {
        K = 1;
    }
    else
    {
        K = std::atoi(argv[7]);
    }

    // validate input parameters
    if (X <= 2 || Y <= 2 || PRED < 0 || PRED > 1 || PREY < 0 || PREY > 1
        || PRED + PREY > 1 || G < 1 || U < 0 || K > 16 || K < 0)
    {
        std::cerr << "Invalid parameters\n";
        return EXIT_FAILURE;
    }

// start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

// initialize the grid
    Cell** grid = new Cell*[Y];
    Cell** new_grid = new Cell*[Y];
    for (int i = 0; i < Y; i++)
    {
        grid[i] = new Cell[X];
        new_grid[i] = new Cell[X];
    }

// populate the grid with predators and prey
    int initialPredators = static_cast<int>((X * Y) * PRED);
    int initialPrey = static_cast<int>((X * Y) * PREY);

    // ensure that population stays within the grid
    initialPredators = std::min(initialPredators, X * Y);
    initialPrey = std::min(initialPrey, (X * Y) - initialPredators);

    // randomly place prey
    for (int i = 0; i < initialPrey; i++)
    {
        int x = distX(gen);
        int y = distY(gen);
        while (grid[y][x].type != Cell::EMPTY)
        {
            x = distX(gen);
            y = distY(gen);
        }
        grid[y][x] = Cell(Cell::PREY, 1);
    }

    // randomly place predators
    for (int i = 0; i < initialPredators; i++)
    {
        int x = distX(gen);
        int y = distY(gen);
        while (grid[y][x].type != Cell::EMPTY)
        {
            x = distX(gen);
            y = distY(gen);
        }
        grid[y][x] = Cell(Cell::PREDATOR, -1);
    }

    // simulate generations
    for (int gen_num = 0; gen_num <= G; gen_num++)
    {
        // output the grid at specified generations
        if (gen_num % U == 0 || gen_num == G)
        {
            std::ofstream outfile("generation_" + std::to_string(gen_num) + ".csv");
            if (!outfile)
            {
                std::cerr << "Error opening output file for generation " << gen_num << std::endl;
                return EXIT_FAILURE;
            }
            for (int i = 0; i < Y; i++)
            {
                for (int j = 0; j < X; j++)
                {
                    // predator ends in d
                    if (grid[i][j].type == Cell::PREDATOR)
                    {
                        outfile << "d";
                    }
                        // prey ends in y
                    else if (grid[i][j].type == Cell::PREY)
                    {
                        outfile << "y";
                    }
                        // e for empty
                    else
                    {
                        outfile << "e";
                    }
                    // set up commas
                    if (j < X - 1)
                    {
                        outfile << ",";
                    }
                }
                // newline
                outfile << '\n';
            }
            // close file
            outfile.close();
        }

        // sets up variables for the threaded simulation
        std::thread threads[K];
        int rows_per_thread = Y / K;

        // splits up the grid into sections and kicks of the threads to run it
        for (int t = 0; t < K; t++)
        {
            int start_row = t * rows_per_thread;
            int end_row = 0;
            // finds the end row
            if (t == K - 1)
            {
                end_row = Y;
            }
            else
            {
                end_row = start_row + rows_per_thread;
            }
            std::mt19937 thread_gen(rd());
            threads[t] = std::thread(update_grid_section, std::ref(grid),
                                     std::ref(new_grid), start_row, end_row,
                                     X, Y, std::ref(thread_gen));
        }

        // joins threads
        for (int t = 0; t < K; t++)
        {
            if (threads[t].joinable())
            {
                threads[t].join();
            }
        }

        // swap grids for the next generation
        std::swap(grid, new_grid);
    }

// clean up memory
    for (int i = 0; i < Y; i++)
    {
        delete[] grid[i];
        delete[] new_grid[i];
    }
    delete[] grid;
    delete[] new_grid;

// output how long it took to run
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}

void update_grid_section(Cell** grid, Cell** new_grid, int start_row, int end_row, int X, int Y,
                         std::mt19937& gen)
{
    std::uniform_int_distribution<> rand64(0, 63);
    // update the grid for the next generation
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < X; j++)
        {
            // set up variables for update checks
            Cell& current_cell = grid[i][j];
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
                            // check predator neighbors
                            if (grid[ny][nx].type == Cell::PREDATOR)
                            {
                                pred_neighbors++;
                                if (grid[ny][nx].age <= -3)
                                {
                                    pred_breeding_neighbors++;
                                }
                            }
                            // check prey neighbors
                            else if (grid[ny][nx].type == Cell::PREY)
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

            // cell update logic for empty
            if (current_cell.type == Cell::EMPTY)
            {
                // cell becomes predator
                if (pred_neighbors >= 4 && pred_breeding_neighbors >= 3
                    && prey_neighbors < 4)
                {
                    new_grid[i][j] = Cell(Cell::PREDATOR, -1);
                }
                // cell becomes prey
                else if (prey_neighbors >= 4 && prey_breeding_neighbors >= 3
                         && pred_neighbors < 4)
                {
                    new_grid[i][j] = Cell(Cell::PREY, 1);
                }
                // stay empty
                else
                {
                    new_grid[i][j] = Cell(Cell::EMPTY, 0);
                }
            }
            // cell update logic for predator
            else if (current_cell.type == Cell::PREDATOR)
            {
                int random_num = rand64(gen);
                // cell dies of old age
                if (current_cell.age < -20)
                {
                    new_grid[i][j] = Cell(Cell::EMPTY, 0);
                }
                // cell dies of starvation
                else if (pred_neighbors >= 6 && prey_neighbors == 0)
                {
                    new_grid[i][j] = Cell(Cell::EMPTY, 0);
                }
                // cell dies randomly
                else if (random_num == 0)
                {
                    new_grid[i][j] = Cell(Cell::EMPTY, 0);
                }
                // survives
                else
                {
                    new_grid[i][j] = Cell(Cell::PREDATOR, current_cell.age - 1);
                }
            }
            // prey logic
            else if (current_cell.type == Cell::PREY)
            {
                // cells dies of old age
                if (current_cell.age > 10)
                {
                    new_grid[i][j] = Cell(Cell::EMPTY, 0);
                }
                // cell killed by predators
                else if (pred_neighbors >= 5)
                {
                    new_grid[i][j] = Cell(Cell::EMPTY, 0);
                }
                // cells dies of over population
                else if (prey_neighbors == 8)
                {
                    new_grid[i][j] = Cell(Cell::EMPTY, 0);
                }
                // cell survives
                else
                {
                    new_grid[i][j] = Cell(Cell::PREY, current_cell.age + 1);
                }
            }
        }
    }
}
