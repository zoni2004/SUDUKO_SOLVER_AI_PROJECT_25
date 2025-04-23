import tkinter as tk #UI
from tkinter import simpledialog #for user input
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

#method to get the size of suduko
def get_sudoku_size():
    #Prompt user for Sudoku size (must be perfect square)
    root = tk.Tk()
    root.withdraw()  # Hiding the main window
    
    while True:
        n = simpledialog.askinteger("Sudoku Size", "Enter Sudoku size (perfect square like 4, 9, 16):", minvalue=4, maxvalue=25)
        if n is None:  # User cancelled
            return None
        if math.isqrt(n) ** 2 == n:
            return n
        tk.messagebox.showerror("Invalid Size", "Size must be a perfect square (4, 9, 16, etc.)")

#making sure no same number in same col, row or box
def is_valid(board, row, col, num, n, box_size):
    # Check row and column
    if num in board[row] or num in board[:, col]:
        return False
    # Check box
    start_row, start_col = row - row % box_size, col - col % box_size #identifying the start of the box(row, col)
    if num in board[start_row:start_row + box_size, start_col:start_col + box_size]:
        return False
    return True

def fill_board(board, n, box_size):
    for row in range(n):
        for col in range(n):
            if board[row][col] == 0:
                nums = list(range(1, n + 1)) #creating a list of numbers that are possible for the nxn board (i.e. 1,n)
                random.shuffle(nums) #shuffling the numbers for filling in the board randomly
                for num in nums:
                    #checking validity of the num from the shuffled num list
                    if is_valid(board, row, col, num, n, box_size):
                        board[row][col] = num #putting the number in the board
                        if fill_board(board, n, box_size): #recursively calls the function to fill the next empty cell
                            return True #when all the boxes are filled
                        board[row][col] = 0 #backtracks if the recursive call doesnt lead to a solution 
                return False #If no number from 1 to n works in this cell, returns False to trigger backtracking in the previous call.
    return True #If all cells are filled, returns True indicating a complete solution.

#removing numbers from the filled suduko table to create the game board
def remove_numbers(board, difficulty):
    # Difficulty = "easy", "medium", "hard"
    n = len(board)
    total_cells = n * n
    removals = {
        #deciding on how many cells to remove depending on teh difficulty level
        "easy": int(0.4 * total_cells), #removing 40% values
        "medium": int(0.5 * total_cells), #removing 50% values
        "hard": int(0.6 * total_cells) #removing 60% values
    }
    cells_to_remove = removals.get(difficulty, 40) #Retrieves the number of cells to remove based on the difficulty. Defaults = 40
    puzzle = board.copy() #copying the filled board to remove values

    #looping to remove values
    while cells_to_remove > 0:
        row, col = random.randint(0, n - 1), random.randint(0, n - 1) #randomly selecting rows and cols to remove value
        #removing the value from the puzzle[row][col]
        if puzzle[row][col] != 0:
            puzzle[row][col] = 0 #0 value represents an empty cell to be filled
            cells_to_remove -= 1 #decreasing the numbers of cells left to remove for loop condition

    return puzzle #returning the suduko board with removed values from cells

#creating the suduko board
def create_sudoku_puzzle(n=9, difficulty="medium"):
    #making sure 9 is a perfect square 
    if math.isqrt(n) ** 2 != n:
       print("n must be a perfect square (e.g., 4, 9, 16).")
       return

    board = np.zeros((n, n), dtype=int) #creating the board
    box_size = int(math.sqrt(n)) #deciding on the box size 
    fill_board(board, n, box_size) #filling the board with random numbers
    puzzle = remove_numbers(board, difficulty) #creating the puzzle by removing values from random cells
    return puzzle

#class to solve the suduko using mrv
class SudokuSolverMRV:
    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.box_size = int(math.sqrt(self.n))
        self.nodes_visited = 0
        self.max_recursion_depth = 0

    def is_valid(self, row, col, num):
        # using the global is_valid function
        return is_valid(self.board, row, col, num, self.n, self.box_size)

    #method to implemenet the minimum remaining values heuritstic (finds the empty cell with the fewest valid number options)
    def find_mrv_cell(self):
        min_options = self.n + 1 #setting to an option larger to the maximum possible options to have as few options during comparison as possible
        best_cell = None
        for row in range(self.n):
            for col in range(self.n):
                #finding the first empty cell
                if self.board[row][col] == 0:
                    options = [num for num in range(1, self.n + 1) if self.is_valid(row, col, num)] #Generating a list of valid numbers that can legally be placed in (row, col) using self.is_valid()
                    #finding the better candidate by checking if the current cell has fewer options (len(options)) than the previous minimum (min_options)
                    if len(options) < min_options:
                        min_options = len(options)
                        best_cell = (row, col) #finding the best cell (cell with minimum options)
                        if min_options == 1: #if a cell has minimum options = 1 immediatly return the row and col for that (best cell)
                            return best_cell 
        return best_cell

    #solving the suduko
    def solve(self, depth=0):
        self.nodes_visited += 1
        if depth > self.max_recursion_depth:
            self.max_recursion_depth = depth
            
        cell = self.find_mrv_cell() #finding best cell to put value in using mrv
        #if no empty cells left then leave the game as puzzle solved
        if not cell:
            return True
        #otherwise
        row, col = cell
        for num in range(1, self.n + 1):
            if self.is_valid(row, col, num): #checking if number is valid to be put in the cell
                self.board[row][col] = num
                #recursive calling to check if all cells filled
                if self.solve(depth + 1):
                    return True
                self.board[row][col] = 0 #backtracking if recursion fails
        return False

#class to solve the suduko using backtracking
class SudokuSolverBacktracking:
    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.box_size = int(math.sqrt(self.n))
        self.nodes_visited = 0
        self.max_recursion_depth = 0

    def is_valid(self, row, col, num):
        # using the global is_valid function
        return is_valid(self.board, row, col, num, self.n, self.box_size)

    def solve(self, depth=0):
        self.nodes_visited += 1
        if depth > self.max_recursion_depth:
            self.max_recursion_depth = depth
            
        for row in range(self.n):
            for col in range(self.n):
                if self.board[row][col] == 0: #checking if cell empty
                     for num in range(1, self.n + 1):
                        if self.is_valid(row, col, num): #checking validity of the number
                            self.board[row][col] = num
                            if self.solve(depth + 1): #recursive call returning true if no empty cell
                                return True
                            self.board[row][col] = 0 #backtracking if recursive call not true
                     return False
        return True

#class to solve the suduko using forward checking
class ForwardCheckingSolver:
    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.box_size = int(math.sqrt(self.n))
        self.nodes_visited = 0
        self.max_recursion_depth = 0
        # Initializing domains for all empty cells (1-9 for standard Sudoku)
        self.domains = {}
        for row in range(self.n):
            for col in range(self.n):
                if board[row][col] == 0:
                    self.domains[(row, col)] = set(range(1, self.n + 1))
    
    def is_valid(self, row, col, num):
        # using the global is_valid function
        return is_valid(self.board, row, col, num, self.n, self.box_size)
    
    #applying forward checking
    def forward_check(self, row, col, num):
        # Remove 'num' from domains of constrained cells
        affected = []
        
        # Check row and column
        for i in range(self.n):
            if i != col and (row, i) in self.domains and num in self.domains[(row, i)]:
                self.domains[(row, i)].remove(num)
                affected.append((row, i))
            
            if i != row and (i, col) in self.domains and num in self.domains[(i, col)]:
                self.domains[(i, col)].remove(num)
                affected.append((i, col))
        
        # Check box
        start_row, start_col = row - row % self.box_size, col - col % self.box_size
        for i in range(self.box_size):
            for j in range(self.box_size):
                r, c = start_row + i, start_col + j
                if (r, c) in self.domains and num in self.domains[(r, c)] and (r, c) != (row, col):
                    self.domains[(r, c)].remove(num)
                    affected.append((r, c))
        
        return affected
    
    def restore_domains(self, affected, num):
        # Restore 'num' to domains of affected cells
        for (row, col) in affected:
            self.domains[(row, col)].add(num)
    
    def solve(self, depth=0):
        self.nodes_visited += 1
        if depth > self.max_recursion_depth:
            self.max_recursion_depth = depth
        # Find the cell with minimum remaining values (MRV)
        if not self.domains:
            return True  # All cells filled
        
        # Get cell with smallest domain
        row, col = min(self.domains.keys(), key=lambda k: len(self.domains[k]))
        
        for num in list(self.domains[(row, col)]):
            if self.is_valid(row, col, num):
                self.board[row][col] = num
                del self.domains[(row, col)]
                
                # Forward checking
                affected = self.forward_check(row, col, num)
                
                if self.solve(depth + 1):
                    return True
                
                # Backtracking otherwise
                self.board[row][col] = 0
                self.domains[(row, col)] = set(range(1, self.n + 1))
                self.restore_domains(affected, num)
        
        return False

class SudokuSolverEvaluator:
    def __init__(self, puzzle):
        self.puzzle = puzzle.copy()
        self.n = len(puzzle)
        self.box_size = int(math.sqrt(self.n))
        self.metrics = {
            'MRV': {'time': 0, 'nodes': 0, 'depth': 0},
            'Backtracking': {'time': 0, 'nodes': 0, 'depth': 0},
            'ForwardChecking': {'time': 0, 'nodes': 0, 'depth': 0}
        }
    
    def evaluate_all(self):
        self.evaluate_mrv()
        self.evaluate_backtracking()
        self.evaluate_forward_checking()
        self.show_results()
    
    def evaluate_mrv(self):
        board = self.puzzle.copy()
        solver = SudokuSolverMRV(board)
        
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        
        self.metrics['MRV']['time'] = end_time - start_time
        self.metrics['MRV']['nodes'] = solver.nodes_visited
        self.metrics['MRV']['depth'] = solver.max_recursion_depth
    
    def evaluate_backtracking(self):
        board = self.puzzle.copy()
        solver = SudokuSolverBacktracking(board)
        
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        
        self.metrics['Backtracking']['time'] = end_time - start_time
        self.metrics['Backtracking']['nodes'] = solver.nodes_visited
        self.metrics['Backtracking']['depth'] = solver.max_recursion_depth
    
    def evaluate_forward_checking(self):
        board = self.puzzle.copy()
        solver = ForwardCheckingSolver(board)
        
        start_time = time.time()
        solver.solve()
        end_time = time.time()
        
        self.metrics['ForwardChecking']['time'] = end_time - start_time
        self.metrics['ForwardChecking']['nodes'] = solver.nodes_visited
        self.metrics['ForwardChecking']['depth'] = solver.max_recursion_depth
    
    def show_results(self):
        # Create a summary string
        result_text = "Algorithm Performance Comparison:\n\n"
        for algo, metrics in self.metrics.items():
            result_text += f"{algo}:\n"
            result_text += f"  Time: {metrics['time']:.6f} seconds\n"
            result_text += f"  Nodes visited: {metrics['nodes']}\n"
            result_text += f"  Max recursion depth: {metrics['depth']}\n\n"
        
        # Show the text results
        messagebox.showinfo("Algorithm Comparison", result_text)
        
        # Generate and show plots
        self.generate_plots()
    
    def generate_plots(self):
        algorithms = list(self.metrics.keys())
        times = [self.metrics[algo]['time'] for algo in algorithms]
        nodes = [self.metrics[algo]['nodes'] for algo in algorithms]
        depths = [self.metrics[algo]['depth'] for algo in algorithms]
        
        plt.figure(figsize=(15, 5))
        
        # Time comparison
        plt.subplot(1, 3, 1)
        plt.bar(algorithms, times, color=['blue', 'green', 'red'])
        plt.title('Time Comparison (seconds)')
        plt.yscale('log')  # Log scale for better visualization
        
        # Nodes visited comparison
        plt.subplot(1, 3, 2)
        plt.bar(algorithms, nodes, color=['blue', 'green', 'red'])
        plt.title('Nodes Visited')
        plt.yscale('log')
        
        # Recursion depth comparison
        plt.subplot(1, 3, 3)
        plt.bar(algorithms, depths, color=['blue', 'green', 'red'])
        plt.title('Max Recursion Depth')
        
        plt.tight_layout()
        plt.show()
        
#creating the board using tinker
class SudokuGUI:
    def __init__(self, root):
        self.root = root
        self.n = None
        self.box_size = None
        self.difficulty = None
        self.puzzle = None
        self.initial_puzzle = None #to store the initial state of the puzzle
        self.entries = None #creating a 2D list to store the Tkinter Entry widgets that represent each cell in the Sudoku grid.
        self.get_puzzle_parameters()
        #Only proceeding if parameters were provided
        if self.n and self.difficulty:
            self.initialize_puzzle()
            self.draw_grid()
            self.add_buttons()
    
    def get_puzzle_parameters(self):
        # Get puzzle size and difficulty from user
        self.n = get_sudoku_size()
        if not self.n:
            return
            
        self.difficulty = simpledialog.askstring("Difficulty", "Choose difficulty (easy, medium, hard):", initialvalue="medium")
        if not self.difficulty:
            return
            
        self.box_size = int(math.sqrt(self.n))

    def initialize_puzzle(self):
        #Create the puzzle based on user parameters
        self.puzzle = create_sudoku_puzzle(self.n, self.difficulty)
        self.initial_puzzle = self.puzzle.copy()
        self.entries = [[None for _ in range(self.n)] for _ in range(self.n)]
        
    #method to visually create and lay out the Sudoku grid using Tkinter Entry widgets.
    def draw_grid(self):
        for row in range(self.n):
            for col in range(self.n):
                entry = tk.Entry(self.root, width=2, font=('Arial', 18), justify='center')
                entry.grid(row=row, column=col, padx=1, pady=1)
                if self.puzzle[row][col] != 0:
                    entry.insert(0, str(self.puzzle[row][col]))
                    entry.config(state='disabled', disabledforeground='red')
                self.entries[row][col] = entry
    
    #method to validate inputs to get real life output if input is invalid    
    def validate_input(self, event, row, col):
        #Validate user input in real-time
        entry = self.entries[row][col]
        value = entry.get()
        
        # Skip validation for empty cells or original clues
        if not value or self.initial_puzzle[row][col] != 0:
            return
            
        # Check if input is a single digit and within valid range
        if not value.isdigit() or len(value) > 1 or int(value) > self.n or int(value) < 1:
            entry.delete(0, tk.END)
            entry.config(bg='#FFCCCC')  # Light red for invalid
            return
            
        num = int(value)
        
        # Check validity against current board state
        temp_board = self.get_current_board()
        temp_board[row][col] = 0  # Temporarily clear current cell
        
        if not is_valid(temp_board, row, col, num, self.n, self.box_size):
            entry.config(bg='#FFCCCC')  # Light red for conflict
        else:
            entry.config(bg='green')   # green for valid
    
    #method to get hint from AI at a current state    
    def get_hint(self):
        #Provide hint by filling a random valid number for an empty cell
        # Get current board state
        board = self.get_current_board()
        
        # Find all empty cells with their valid options
        candidates = []
        for row in range(self.n):
            for col in range(self.n):
                if board[row][col] == 0:
                    options = [num for num in range(1, self.n+1) 
                              if is_valid(board, row, col, num, self.n, self.box_size)]
                    if options:
                        candidates.append((row, col, options))
        
        if not candidates:
            tk.messagebox.showinfo("Hint", "Puzzle already solved!")
            return
            
        # Choose a random empty cell
        row, col, options = random.choice(candidates)
        # Choose a random valid number for that cell
        chosen_num = random.choice(options)
        
        # Update the board and GUI
        self.entries[row][col].delete(0, tk.END)
        self.entries[row][col].insert(0, str(chosen_num))
        self.entries[row][col].config(bg='#CCFFCC')  # Light green to indicate hint
        
        # Show which cell was filled
        hint_window = tk.Toplevel(self.root)
        hint_window.title("Hint Applied")
        tk.Label(hint_window, text=f"Filled {chosen_num} at row {row+1}, column {col+1}", 
                font=('Arial', 12)).pack(padx=20, pady=10)
        
        # Auto-close after 3 seconds
        self.root.after(3000, hint_window.destroy)
    
    # Add this method to the SudokuGUI class
    def check_solution(self):
        # Get current board state from GUI
        current_board = self.get_current_board()
        # Check if any cell is empty
        for row in range(self.n):
            for col in range(self.n):
                if current_board[row][col] == 0:
                    tk.messagebox.showinfo("Incomplete", "There are still empty cells to fill!")
                    return
                
        # Validate all numbers
        for row in range(self.n):
            for col in range(self.n):
                num = current_board[row][col]
                # Temporarily clear the cell to check validity
                current_board[row][col] = 0
                if not is_valid(current_board, row, col, num, self.n, self.box_size):
                    tk.messagebox.showinfo("Incorrect", f"Conflict found at row {row+1}, column {col+1}!")
                    # Highlighting the incorrect cells
                    self.entries[row][col].config(bg='#FFCCCC')
                    return
                current_board[row][col] = num
                
        #the solution is correct
        self.show_congratulations()
    
    def show_congratulations(self):
        # Create a new window for the congratulations message
        win_window = tk.Toplevel(self.root)
        win_window.title("Congratulations!")
        win_window.geometry("400x200")
        
        # Add congratulatory message
        tk.Label(win_window, text="Congratulations!", font=('Arial', 24, 'bold'), fg='green').pack(pady=20)
        tk.Label(win_window, text="You've solved the Sudoku puzzle correctly!", font=('Arial', 14)).pack()
    
        # Add a close button
        tk.Button(win_window, text="Close", command=win_window.destroy, font=('Arial', 12)).pack(pady=20)
        
    #method to reset the code
    def reset_puzzle(self):
        # Ask user what type of reset they want
        choice = tk.messagebox.askquestion("Reset Puzzle", "Would you like to:\n\n""Yes - Reset to original puzzle\n" "No - Generate a new puzzle\n""Cancel - Do nothing", icon='question', type='yesnocancel')
        
        if choice == 'yes':
            # Reset to original puzzle
            for row in range(self.n):
                for col in range(self.n):
                    self.entries[row][col].config(state='normal', bg='white')
                    self.entries[row][col].delete(0, tk.END)
                    if self.initial_puzzle[row][col] != 0:
                        self.entries[row][col].insert(0, str(self.initial_puzzle[row][col]))
                        self.entries[row][col].config(state='disabled', disabledforeground='red')
        elif choice == 'no':
            # Generate a completely new puzzle
            self.puzzle = create_sudoku_puzzle(self.n, self.difficulty)
            self.initial_puzzle = self.puzzle.copy()
            # Clear and redraw the grid
            for row in range(self.n):
                for col in range(self.n):
                    self.entries[row][col].config(state='normal', bg='white')
                    self.entries[row][col].delete(0, tk.END)
                    if self.puzzle[row][col] != 0:
                        self.entries[row][col].insert(0, str(self.puzzle[row][col]))
                        self.entries[row][col].config(state='disabled', disabledforeground='red')
    
    def compare_algorithms(self):
        evaluator = SudokuSolverEvaluator(self.puzzle)
        evaluator.evaluate_all()
                        
    def draw_grid(self):
        for row in range(self.n):
            for col in range(self.n):
                entry = tk.Entry(self.root, width=2, font=('Arial', 18), justify='center')
                entry.grid(row=row, column=col, padx=1, pady=1)
                
                # Bind validation to each entry
                entry.bind('<KeyRelease>', lambda e, r=row, c=col: self.validate_input(e, r, c))
                
                if self.puzzle[row][col] != 0:
                    entry.insert(0, str(self.puzzle[row][col]))
                    entry.config(state='disabled', disabledforeground='red')
                self.entries[row][col] = entry
            
    #method to add a buttons that needs to be clicked to solve the puzzle
    def add_buttons(self):
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=self.n + 1, column=0, columnspan=self.n, pady=10)
        
        mrv_btn = tk.Button(button_frame, text="Solve with MRV", 
                           command=self.solve_puzzle_mrv, font=('Arial', 12))
        mrv_btn.pack(side=tk.LEFT, padx=5) #button for solving using MRV
        
        backtrack_btn = tk.Button(button_frame, text="Solve with Backtracking", 
                                command=self.solve_puzzle_backtracking, font=('Arial', 12))
        backtrack_btn.pack(side=tk.LEFT, padx=5) #button for solving using Backtrack
        
        fc_btn = tk.Button(button_frame, text="Solve with Forward Checking", 
                          command=self.solve_puzzle_forward_checking, font=('Arial', 12))
        fc_btn.pack(side=tk.LEFT, padx=5) #button for solving using Forward Checking
        
        hint_btn = tk.Button(button_frame, text="Get Hint", command=self.get_hint, font=('Arial', 12))
        hint_btn.pack(side=tk.LEFT, padx=5) #button for solving getting hint
        
        check_btn = tk.Button(button_frame, text="Check Solution", command=self.check_solution, font=('Arial', 12))
        check_btn.pack(side=tk.LEFT, padx=5) #button for checking user solution
        
        reset_btn = tk.Button(button_frame, text="Reset", command=self.reset_puzzle, font=('Arial', 12))
        reset_btn.pack(side=tk.LEFT, padx=5) #button for resetting
        
        compare_btn = tk.Button(button_frame, text="Compare Algorithms", command=self.compare_algorithms, font=('Arial', 12))
        compare_btn.pack(side=tk.LEFT, padx=5) #button for comparing algos

    def get_current_board(self):
        board = np.zeros((self.n, self.n), dtype=int)
        for row in range(self.n):
            for col in range(self.n):
                val = self.entries[row][col].get()
                board[row][col] = int(val) if val.isdigit() else 0
        return board
    
    #updating gui depending on what button is pressed
    def update_gui(self, solved_board):
        for r in range(self.n):
            for c in range(self.n):
                self.entries[r][c].config(state='normal') #temporarily sets the state of the Tkinter Entry widget at position (r,c) to 'normal'
                self.entries[r][c].delete(0, tk.END) #Ensures we start with a blank cell before inserting the solved value.
                self.entries[r][c].insert(0, str(solved_board[r][c])) #Inserts the solved number from solved_board[r][c] at position 0 (beginning of entry).
                #if solved value then display green
                if self.initial_puzzle[r][c] == 0:
                    self.entries[r][c].config(disabledforeground='green', state='disabled')
                else: #else show red (i.e. original value)
                    self.entries[r][c].config(disabledforeground='red', state='disabled')
    
    #using method to call the original solve method for mrv and updating the gui as is
    def solve_puzzle_mrv(self):
        board = self.get_current_board()
        solver = SudokuSolverMRV(board)
        if solver.solve():
            self.update_gui(solver.board)
        else:
            print("No solution found with MRV.")
            
    #using method to call the original solve method for backtracking and updating the gui as is
    def solve_puzzle_backtracking(self):
        board = self.get_current_board()
        solver = SudokuSolverBacktracking(board)
        if solver.solve():
             self.update_gui(solver.board)
        else:
            print("No solution found with Backtracking.")
    
     #using method to call the original solve method for forward checking and updating the gui as is
    def solve_puzzle_forward_checking(self):
        board = self.get_current_board()
        solver = ForwardCheckingSolver(board)
        if solver.solve():
            self.update_gui(solver.board)
        else:
            print("No solution found with Forward Checking.")

# Launching GUI
root = tk.Tk()
root.title("Sudoku Puzzle")
# Only create GUI if user provided valid parameters
gui = SudokuGUI(root)
# If user cancelled parameter selection, close the window
if not gui.n:
    root.destroy()
else:
    root.mainloop()