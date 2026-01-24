import AmotizedDSL.program_interpreter as pi
import numpy as np
import AmotizedDSL.DSL as DSL
from AmotizedDSL.prog_utils import ProgUtils
from ARC_gym.grid_sampling.grid_sampler import GridSampler
import ARC_gym.utils.visualization as viz
import copy

def execute_prog(prog, grid_categories, K=3):

    # Step 1: generate K input grids using ARC_gym
    sampler = GridSampler()

    k_input_grids_dsl = []
    k_object_masks = []
    for k in range(K):
        input_grid, object_mask = sampler.sample_by_category(grid_categories)

        input_grid_dsl = DSL.GridObject.from_grid(input_grid)

        k_input_grids_dsl.append([input_grid_dsl])
        k_object_masks.append(np.array(object_mask))
    
    # Step 2: compile and execute the program example
    instructions = ProgUtils.convert_user_format_to_token_seq(prog)
    initial_state = copy.deepcopy(k_input_grids_dsl)

    output_grids_dsl = pi.execute(instructions, initial_state, DSL, k_object_masks)

    # Step 3: visualize the results
    for k in range(len(output_grids_dsl)):
        print(f"Example #{k+1}")
        viz.draw_grid_pair(k_input_grids_dsl[k][0].cells_as_numpy(), output_grids_dsl[k].cells_as_numpy())

# This program example is "Shear Left" in Task DB
# It shears the input grids towards the left from the bottom up
shear_left_prog = [
    'add(N+0.x, N+0.y)',
    'new_grid(N+0.width, N+0.height, 0)',
    'set_pixels(N+2, N+1, N+0.y, N+0.c)',
    'del(N+0)',
    'del(N+0)',
    'del(N+0)'
]

print("Executing and visualizing the \"Shear Left\" program.")
execute_prog(shear_left_prog, ['shearable_grids'])

# This program example is "Object Move Down by Height" in Task DB
# It independently translates downward each object by a distance equal
# to their respective heights.
obj_move_down_prog = [
    'get_objects(N+0)',
    'get_bg(N+0)',
    'del(N+0)',
    'add(N+0.y, N+0.height)',
    'set_y(N+0, N+2)',
    'del(N+0)',
    'del(N+1)',
    'rebuild_grid(N+0, N+1)',
    'del(N+0)',
    'del(N+0)'
]

print("Executing and visualizing the \"Object Move Down by Height\" program.")
execute_prog(obj_move_down_prog, ['distinct_colors_adjacent', 'simple_filled_rectangles'])
