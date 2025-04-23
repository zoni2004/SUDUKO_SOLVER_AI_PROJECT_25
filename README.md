# SUDUKO_SOLVER_AI_PROJECT_25

A Python-based Sudoku game and solver that compares three AI-driven techniques to solve puzzles of varying sizes and difficulties. Built with a user-friendly GUI, real-time validation, and performance benchmarking tools.

---

## 👥 Team Members

- **Zunaira Amjad** (23K-0013) – Backtracking & MRV Algorithms  
- **Emaan Arshad** (23I-2560) – Forward Checking & Solver Evaluation  
- **Dania Khan** (23K-0072) – GUI Development & Real-time Validation  
- **Tanisha Kataria** (23K-0067) – Performance Metrics & Visualization

---

## 📌 Project Overview

This project implements and compares three classic CSP-based AI algorithms to solve Sudoku puzzles:

1. **Basic Backtracking** (Brute-force)
2. **Minimum Remaining Values (MRV)** Heuristic
3. **Forward Checking** (Constraint Propagation)

🔹 **Key Features:**
- Solve puzzles using different algorithms  
- Variable grid sizes: `4×4`, `9×9`, `16×16`, and beyond  
- Adjustable difficulty levels: Easy, Medium, Hard  
- Hint generation & solution checker  
- Performance comparison via time, recursion depth, and nodes visited  
- GUI interface with real-time validation and player-AI interaction  

---

## 🧩 Game Rules & Innovations

### ✅ Classic Rules:
- Every row, column, and subgrid must contain all digits exactly once.
- Puzzles start with given clues and must be completed logically.

### 🚀 Innovations:
- Custom grid sizes (4×4 to 25×25)  
- Real-time error highlighting  
- Hint-based gameplay with AI assistance  
- Difficulty control through clue density  
- Performance metrics visualization (Matplotlib)

---

## 🧠 AI Techniques Used

- **Backtracking** – Standard DFS, checks validity, backtracks on failure.
- **MRV Heuristic** – Prioritizes cells with the fewest legal values.
- **Forward Checking** – Dynamically prunes illegal options during search.

---

## 📊 Performance Evaluation

| Metric             | Backtracking | MRV Heuristic | Forward Checking |
|--------------------|--------------|---------------|------------------|
| Time (9×9 Easy)    | ~0.5 sec     | ~0.05 sec     | ~0.03 sec        |
| Nodes Visited      | 10,000+      | ~500          | ~300             |
| Recursion Depth    | High         | Low           | Low              |

*Performance tested on puzzles with varying difficulties and sizes (4×4 to 16×16).*

---

## 💻 Tools & Technologies

- **Language:** Python  
- **GUI:** Tkinter  
- **Logic:** NumPy  
- **Plots:** Matplotlib  
- **Version Control:** GitHub  
- **IDE:** VS Code  

---

## 🛠 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/sudoku-solver.git
cd sudoku-solver

# Install dependencies (if any)
pip install numpy matplotlib

# Run the main file
python main.py
