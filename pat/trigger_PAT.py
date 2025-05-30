#import required libraries
from pathlib import Path
import subprocess
import textwrap
import itertools
import sys
#define global variables
TEMPLATE_PATH = Path("template.csp") 
PAT_EXE = Path(r"MONO-PAT-v3.6.0/PAT3.Console.exe") 
OUTPUT_FILE = Path("output.txt")
RUN_CSP = Path("run.csp")

#define function to build CSP from template CSP 
def build_csp(grid, start_rc, goal_rc, M, out_path="run.csp"):
    #get positions
    start_r, start_c = start_rc
    goal_r,  goal_c  = goal_rc
    #get template text
    template = TEMPLATE_PATH.read_text()
    #replace place holders with updated parameters
    filled = (template
              .replace("{{M}}", str(M))
              .replace("{{GRID}}", str(grid))
              .replace("{{START_R}}", str(start_r))
              .replace("{{START_C}}", str(start_c))
              .replace("{{GOAL_R}}", str(goal_r))
              .replace("{{GOAL_C}}", str(goal_c)))
    #write to output file
    out_file = Path(out_path)
    out_file.write_text(filled)
    return out_file

#run command line call to PAT
def run_pat(csp_path):
    #run subprocess
    print(str(PAT_EXE), str(csp_path), str(OUTPUT_FILE))
    result = subprocess.run(['mono', str(PAT_EXE), str(csp_path), str(OUTPUT_FILE)],text=True, capture_output=True)
    print(f"result: {result}") 
    if result.returncode != 0:
        raise RuntimeError(f"PAT exited with {result.returncode}")

#main code
if __name__ == "__main__":
    # --- example data ----------------------------------------------------
    grid = [
            1, 1, 1,-1, 1, 1, 1, 1,
            1, 1, 1,-1,-1, 1, 1, 1,
            1,-1,-1, 1, 1, 1, 1, 1,
            1,-1, 1, 1, 1, 1, 1,-1,
            1, 1, 1, 1, 1, 1,-1,-1,
            1, 1, 1, 1, 1,-1, 1,-1,
            1, 1, 1, 1, 1,-1, 1, 1,
            1, 1, 1, 1, 1,-1, 1, 1,
    ]
    start_pos = (4, 1)
    goal_pos  = (0, 4)
    grid_size = 8
    # ---------------------------------------------------------------------

    #get CSP file and run PAT 
    csp_file = build_csp(grid, start_pos, goal_pos, grid_size) 
    run_pat(csp_file)