# MOIQL - Multi-Objective Inverse Q-Learning

## Project Overview
This is a Multi-Objective Inverse Q-Learning (MOIQL) repository targeting two benchmark environments for multi-objective reinforcement learning.

## Target Environments

### Deep Sea Treasure
A classic multi-objective reinforcement learning benchmark environment where a submarine agent navigates a 2D grid to collect treasures on the ocean floor.

**Objectives:**
- **Treasure reward**: Treasures have values from 1 to 124, with higher-value treasures located deeper in the ocean
- **Time penalty**: Each movement step incurs a cost of -1, creating a trade-off between treasure value and time spent

**Key characteristics:**
- Discrete action space (up, down, left, right)
- Deterministic transitions
- Creates a well-defined Pareto frontier with 10 optimal solutions
- Used to evaluate multi-objective RL algorithms' ability to find diverse optimal trade-offs

### Highway Environment
A collection of configurable driving simulation environments built on top of Pygame, designed for testing autonomous driving and decision-making algorithms.

**Common scenarios:**
- **highway-v0**: Lane keeping, velocity control, and safe overtaking on a multi-lane highway
- **merge-v0**: Merging onto a highway while avoiding collisions
- **roundabout-v0**: Navigating roundabouts with multiple vehicles
- **parking-v0**: Parallel parking maneuvers

**Multi-objective settings:**
- **Speed vs. Safety**: Maximize velocity while minimizing collision risk
- **Comfort vs. Efficiency**: Smooth driving vs. reaching destination quickly
- **Lane discipline vs. Speed**: Staying in lanes vs. aggressive overtaking

**Features:**
- Continuous or discrete action spaces
- Configurable traffic density, speeds, and road layouts
- Built-in reward shaping for multi-objective scenarios

## Key Concepts
- **Multi-Objective RL**: Learning policies that balance multiple conflicting objectives
- **Inverse Q-Learning**: Learning reward functions from expert demonstrations
- **Pareto Optimality**: Set of solutions where no objective can be improved without degrading another
