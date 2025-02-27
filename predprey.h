#ifndef PREDATORPREY_PREDPREY_H
#define PREDATORPREY_PREDPREY_H
/**
 * @file predprey.h
 * @brief Method definitions for predator prey model
 *
 * @author Ethan A. Royer (eroyer21@georgefox.edu)
 */

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <random>
#include <thread>
#include <chrono>
#include <limits>
#include <random>

// the struct for the cell
struct Cell
        {
    enum Type
            {
        EMPTY,
        PREDATOR,
        PREY
            };

    Type type;
    int age;

    Cell(Type t = EMPTY, int a = 0) : type(t), age(a) {}
};
/**
 * The function which updates each grid section from the thread.
 *
 * @param grid the un-updated gird
 * @param new_grid the grid to be updated
 * @param start_row the first row of the section
 * @param end_row the last low of the section
 * @param X the width of the grid
 * @param Y the height of the grid
 * @param gen the random number for random death
 */
void update_grid_section(Cell** grid, Cell** new_grid, int start_row, int end_row,
                         int X, int Y, std::mt19937& gen);

#endif //PREDATORPREY_PREDPREY_H
