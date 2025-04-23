# 🧠 Sudoku Solver

A Python-based Sudoku game and solver that compares three AI-driven techniques to solve puzzles of varying sizes and difficulties. Built with a user-friendly GUI, real-time validation, and performance benchmarking tools.

---

## 👥 Submitted By

- **Zunaira Amjad** (23K-0013)  
- **Emaan Arshad** (23I-2560)  
- **Dania Khan** (23K-0072)  
- **Tanisha Kataria** (23K-0067)

**Course:** Artificial Intelligence  
**Instructor:** Ramsha Jat  
**Submission Date:** 11-05-25

---
## VIDEO DEMO
https://drive.google.com/file/d/1PYkPA734cHGijGcl7xPVI_YxxObizI-i/view?usp=sharing

## 📌 Executive Summary

### Project Overview:
This project is a Python-based Sudoku game and solver that implements and compares three different backtracking-based algorithms for solving Sudoku puzzles:

- **Basic Backtracking** (Brute-force approach)  
- **Backtracking with Minimum Remaining Values (MRV)** (Heuristic-based optimization)  
- **Backtracking with Forward Checking** (Constraint propagation optimization)

🔹 **The tool allows users to:**
- Generate Sudoku puzzles of different sizes (4×4, 9×9, 16×16, etc.) and difficulty levels (Easy, Medium, Hard)
- Solve puzzles using any of the three algorithms
- Get hints during gameplay
- Check if their solution is correct
- Reset the puzzle (either to the original state or generate a new one)
- Compare algorithm performance based on:
  - Time taken (Execution speed)
  - Nodes visited (Computational effort)
  - Recursion depth (Memory usage approximation)

---

## 📖 Introduction

### Background:
Sudoku is a classic logic-based number placement puzzle, traditionally played on a 9×9 grid divided into smaller 3×3 subgrids (boxes). The objective is to fill the grid so that:

- Each row contains all digits from 1 to 9 without repetition.
- Each column contains all digits from 1 to 9 without repetition.
- Each 3×3 box contains all digits from 1 to 9 without repetition.

Sudoku was chosen for this project because:

- It’s a well-defined constraint satisfaction problem (CSP), making it ideal for comparing algorithmic approaches.
- It has varying difficulty levels, allowing performance analysis across different complexities.
- It’s widely recognized, making it easy to understand and validate results.

### Objectives of the Project:

1. **Develop and Compare AI Solvers** – Implement and evaluate three distinct Sudoku-solving algorithms to analyze their efficiency.
2. **Enhance Gameplay Features** – Expand traditional Sudoku with adjustable grid sizes and real-time validation.
3. **Enable Human-AI Interaction** – Provide hints, solution checking, and reset features to assist players.
4. **Benchmark Performance** – Compare solvers through time, steps, and memory usage across multiple puzzle configurations.

---

## 🎮 Game Description

### Original Game Rules:
Sudoku is a number-placement puzzle played on a 9×9 grid (or other perfect-square sizes like 4×4 or 16×16). The goal is to fill empty cells with digits from 1 to 9 (or 1 to n for n×n grids) such that each row, column, and subgrid contains every digit exactly once. The puzzle starts with some cells pre-filled ("clues").

### Innovations and Modifications:
- Variable grid sizes (4×4, 9×9, 16×16, etc.)
- Three AI-solving algorithms (backtracking, MRV heuristic, forward checking)
- Adjustable difficulty levels (easy, medium, hard)
- GUI with real-time validation
- Hint generation, solution checking, and performance comparison

---

## 🧠 AI Approach and Methodology

### AI Techniques Used:
This project employs classical search and constraint satisfaction techniques:

1. **Backtracking (Brute-Force Search)** – Explores possible placements recursively.
2. **Minimum Remaining Values (MRV) Heuristic** – Prioritizes cells with the fewest legal options.
3. **Forward Checking** – Eliminates invalid options early, preventing wasted paths.

These methods demonstrate core AI principles such as heuristic-guided search, state-space reduction, and constraint satisfaction.

---

### Algorithm and Heuristic Design

- **Backtracking:** Explores number placements, reverting upon invalid moves.
- **MRV Heuristic:** Reduces search space by targeting the most constrained cells first.
- **Forward Checking:** Dynamically updates possible values to avoid futile paths.

---

### AI Performance Evaluation

Performance measured by **time complexity, nodes visited, and recursion depth** across puzzle sizes (4×4 to 16×16).  
- MRV and Forward Checking solved 9×9 puzzles in under **0.1 seconds** with ~**500 nodes**, while basic backtracking visited **10,000+ nodes**.
- On 16×16 puzzles, forward checking performed best overall.

---

### Modified Game Rules

- Classic rules preserved: each row, column, and subgrid must contain unique digits.
- Added:
  - **Custom grid sizes**
  - **Adjustable difficulties** (easy = 40% empty, hard = 60%)
  - **Real-time validation** and **AI hints**

---

### Turn-based Mechanics

- Players fill cells through GUI input.
- Instant feedback on move validity.
- Option to solve puzzle or receive hints.
- Game ends when correctly filled or AI is asked to solve.

---

### Winning Conditions

- Grid must be completely and correctly filled.
- The system checks for row, column, and subgrid consistency.
- Players can check progress mid-game.

---

## 🛠 Development Process

- Built in **Python**
- GUI with **Tkinter**
- Grid manipulation using **NumPy**
- Visual performance metrics using **Matplotlib**
- Version control with **GitHub**

---

### Programming Languages and Tools

- **Language:** Python  
- **Libraries:** Tkinter, NumPy, Matplotlib  
- **Tools:** GitHub, Visual Studio Code

---

### Challenges Encountered

- **Scalability issues** on 16×16+ grids → resolved with optimization techniques
- **Recursion depth** exceeded limits → addressed with iterative and pruning methods
- **GUI synchronization** → resolved via structured state management

---

### Team Contributions

- **Zunaira Amjad:** Implemented core backtracking/MRV algorithms  
- **Emaan Arshad:** Designed forward checking and solver evaluation  
- **Dania Khan:** Developed Tkinter GUI and real-time validation  
- **Tanisha Kataria:** Conducted performance benchmarking and metrics visualization  

---

### AI Performance

| Metric         | Backtracking | MRV         | Forward Checking |
|----------------|--------------|-------------|------------------|
| Time (9×9)     | ~0.5 sec     | ~0.05 sec   | ~0.03 sec        |
| Nodes Visited  | 10,000+      | ~500        | ~300             |
| Depth (16×16)  | High         | Medium      | Low              |

Players using AI hints solved puzzles **2–3x faster** than unaided attempts.

---

## 📈 References

1. Russell & Norvig, *Artificial Intelligence: A Modern Approach*  
2. Knuth, *Dancing Links*  
3. [Python.org](https://www.python.org) – Tkinter and NumPy Docs  
4. Academic literature on Sudoku solver heuristics


