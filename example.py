import DSL
from prog_utils import ProgUtils
import program_interpreter as pi
import numpy as np
from ARC_gym.grid_sampling.grid_sampler import GridSampler
import ARC_gym.utils.visualization as viz


# IMPORTANT: example.py expects ARC_gym V2 to be installed (https://github.com/SimonOuellette35/ARC_gym/tree/refactor/ARC_gymV2). 
# Also, you will need ARC-AGI-2 (https://github.com/arcprize/ARC-AGI-2) cloned into your current folder, since it samples grids and sub-grids from training examples.

# ================================================================ Various program examples ============================================================================

def generateHFlip(N):
    # This program horizontally flips a grid.
    program, ref = DSL.get_subroutine_hmirror(N+0, N+0)
    program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateVFlip(N):
    # This program vertically flips a grid
    program, ref = DSL.get_subroutine_vmirror(N+0, N+0)
    program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateSetColor(N):
    # This program sets all non-zero pixels of a grid to color 2
    program = [
        ('equal', [(N+0, '.c'), 0]),
        ('switch', [N+1, 0, 2]),
        ('del', [N+1]),
        ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
        ('del', [N+0]),
        ('del', [N+0])
    ]
    return program

def generateTask1(N):
    # This program horizontally flips a grid, then sets all non-zero pixels of a grid to color 2
    hmirror_program, ref = DSL.get_subroutine_hmirror(N+0, N+0)

    hmirror_program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    hmirror_program.append(('del', [N+0]))
    hmirror_program.append(('del', [N+0]))

    set_fg_col_program = [
        ('equal', [(N+0, '.c'), 0]),
        ('switch', [N+1, 0, 2]),
        ('del', [N+1]),
        ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
        ('del', [N+0]),
        ('del', [N+0])
    ]

    program = hmirror_program
    program.extend(set_fg_col_program)

    return program

def generateTask2(N):
    # vertical flip, set non-zero pixels to color 2
    vmirror_program, ref = DSL.get_subroutine_vmirror(N+0, N+0)

    vmirror_program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    vmirror_program.append(('del', [N+0]))
    vmirror_program.append(('del', [N+0]))

    set_fg_col_program = [
        ('equal', [(N+0, '.c'), 0]),
        ('switch', [N+1, 0, 2]),
        ('del', [N+1]),
        ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
        ('del', [N+0]),
        ('del', [N+0])
    ]

    program = vmirror_program
    program.extend(set_fg_col_program)

    return program

def generateTask3(N):
    # shift pixels to the right (no wrapping around, preserve original grid dimensions)
    program = [
        ('add', [(N+0, '.x'), 1]),
        ('set_pixels', [N+0, N+1, (N+0, '.y'), (N+0, '.c')]),
        ('del', [N+1]),
        ('set_pixels', [N+1, 0, (N+1, '.y'), 0]),
        ('del', [N+1]),
        ('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]),
        ('del', [N+0]),
        ('del', [N+0])
    ]

    return program

def generateTask4(N):
    # right shift, then hflip
    program = [
        ('add', [(N+0, '.x'), 1]),
        ('set_pixels', [N+0, N+1, (N+0, '.y'), (N+0, '.c')]),
        ('del', [N+1]),
        ('set_pixels', [N+1, 0, (N+1, '.y'), 0]),
        ('del', [N+1]),
        ('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]),
        ('del', [N+0]),
        ('del', [N+0])
    ]

    hmirror_program, ref = DSL.get_subroutine_hmirror(N+0, N+0)
    hmirror_program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    hmirror_program.append(('del', [N+0]))
    hmirror_program.append(('del', [N+0]))

    program.extend(hmirror_program)
    return program

def generateTask5(N):
    # vflip, right shift
    program, ref = DSL.get_subroutine_vmirror(N+0, N+0)
    program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    # right shift:
    program.append(('add', [(N+0, '.x'), 1]))
    program.append(('set_pixels', [N+0, N+1, (N+0, '.y'), (N+0, '.c')]))
    program.append(('del', [N+1]))
    program.append(('set_pixels', [N+1, 0, (N+1, '.y'), 0]))
    program.append(('del', [N+1]))
    program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateTask6(N):
    # hflip, shift up
    program, ref = DSL.get_subroutine_hmirror(N+0, N+0)
    program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    # shift up:
    program.append(('sub', [(N+0, '.y'), 1]))
    program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
    program.append(('del', [N+1]))
    program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
    program.append(('del', [N+1]))
    program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateTask7(N):
    # shift down, hflip
    program = []
    program.append(('add', [(N+0, '.y'), 1]))
    program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
    program.append(('del', [N+1]))
    program.append(('set_pixels', [N+1, (N+1, '.x'), 0, 0]))
    program.append(('del', [N+1]))
    program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    tmp_hflip, ref = DSL.get_subroutine_hmirror(N+0, N+0)
    program.extend(tmp_hflip)
    program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateTask8(N):
    # shift up, vflip
    program = []
    program.append(('sub', [(N+0, '.y'), 1]))
    program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
    program.append(('del', [N+1]))
    program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
    program.append(('del', [N+1]))
    program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    # vflip , shift up
    tmp_vflip, ref = DSL.get_subroutine_vmirror(N+0, N+0)
    program.extend(tmp_vflip)
    program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateTask9(N):
    # vflip , shift up
    program, ref = DSL.get_subroutine_vmirror(N+0, N+0)
    program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    program.append(('sub', [(N+0, '.y'), 1]))
    program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
    program.append(('del', [N+1]))
    program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
    program.append(('del', [N+1]))
    program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateShiftUp(N):
    # Shift pixels upward (no wrapping around, preserve original grid dimensions)
    program = []
    program.append(('sub', [(N+0, '.y'), 1]))
    program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
    program.append(('del', [N+1]))
    program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
    program.append(('del', [N+1]))
    program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))

    return program

def generateShiftDown(N):
    # Shift pixels downward (no wrapping around, preserve original grid dimensions)
    program = []
    program.append(('add', [(N+0, '.y'), 1]))
    program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
    program.append(('del', [N+1]))
    program.append(('set_pixels', [N+1, (N+1, '.x'), 0, 0]))
    program.append(('del', [N+1]))
    program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
    program.append(('del', [N+0]))
    program.append(('del', [N+0]))
    
    return program

# =========================================================================================================================================================================

DSL_size = len(DSL.semantics)
print(f"The DSL contains {DSL_size} primitives, and uses {ProgUtils.NUM_SPECIAL_TOKENS} special tokens.")

# Step 1 - pick a random program example.
program_func_list = [generateHFlip, generateVFlip, generateSetColor, generateTask1, generateTask2, generateTask3, generateTask4, generateTask5, generateTask6,
                     generateTask7, generateTask8, generateTask9, generateShiftUp, generateShiftDown]

program_func = np.random.choice(program_func_list)

# here we generate the actual program sequence, in "hand-written" representation.
program = program_func(DSL_size)

# Step 2 - generate a random input grid, using ARC_gym.
sampler = GridSampler()
input_grid = sampler.sample()

# Step 3 - execute the program on the input grid.
input_grid_DSL = DSL.Grid(input_grid)

# state is a list of state variables. Here this list contains only the input grid as a DSL.Grid instance.
initial_state = [input_grid_DSL]

token_seq_list = ProgUtils.convert_prog_to_token_seq(program, DSL)
output_grid_DSL = pi.execute(token_seq_list, initial_state, DSL)

output_grid = output_grid_DSL.cells

# Step 4 - display the input and related output for the selected program, using ARC_gym.
print("==> Executed program:")
for step_idx, instr_step in enumerate(program):
    print(f"Step #{step_idx}: {instr_step}")
    
viz.draw_grid_pair(input_grid, output_grid)