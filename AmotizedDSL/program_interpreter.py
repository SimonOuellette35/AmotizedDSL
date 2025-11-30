from typing import List, Tuple
from AmotizedDSL.prog_utils import ProgUtils
import torch
from AmotizedDSL.delete_action import DeleteAction


def resolve_arg(arg, states, primitives, verbose=True):
    if isinstance(arg, Tuple):
        object_arg = arg[0]
        attr_arg = arg[1]

        # resolve the object reference
        ref_idx = object_arg - len(primitives.semantics)
        parent_obj = states[ref_idx]

        # resolve the attribute
        # handle pixel attributes differently than grid-level attributes
        attr_name = primitives.inverse_lookup(attr_arg)
        attr_func = primitives.semantics[attr_name]

        if attr_name in primitives.pixel_attributes:
            if isinstance(parent_obj, List):
                pixels = []
                for obj in parent_obj:
                    if isinstance(obj, primitives.GridObject):
                        tmp_pixels = obj.pixels
                    else:
                        tmp_pixels = obj

                    pixels.append(tmp_pixels)
            else:
                pixels = parent_obj.pixels
                
            output = attr_func(pixels)
        else:
            output = attr_func(parent_obj)

        return output

    else:
        if arg < 10:
            return arg
        elif arg >= len(primitives.semantics):
            # resolve the reference
            ref_idx = arg - len(primitives.semantics)
            return states[ref_idx]
        else:
            print("==> BUG: this case shouldn't happen in resolve_arg.")
            exit(-1)

def get_num_args(prim: int, DSL):
    return DSL.arg_counts[prim]

def execute_step(step_token_seq, states, primitives, object_mask=None, verbose=True):
    '''
    @param step_token_seq: a tuple of tokens (integers) representing the step to execute.
    @param state: [k, num states, variable]
    @param object_mask: Optional object mask (2D numpy array or list) for get_objects/get_bg

    @return Returns the new intermediate state after executing the step.
    '''

    # Step 1: resolve the main primitive
    prim_idx = step_token_seq[0]
    prim_name = primitives.inverse_lookup(prim_idx)
    prim_func = primitives.semantics[prim_name]
    
    # Step 2: parse the arguments (if any)
    token_args = step_token_seq[1]

    result = []
    is_del = False
    for example_idx, example in enumerate(states):
        resolved_args = []
        example_result = None

        if prim_name == 'switch':
            # 'switch' is a special statement, in that the number of arguments is dynamic, and
            # some logic must be used to determine which are the conditions and which are the operations.
            otherwise = resolve_arg(token_args[-1], example, primitives, verbose)

            conditions_range = (len(token_args) - 1) // 2

            conditions = []
            for arg_idx in range(conditions_range):
                tmp_arg = resolve_arg(token_args[arg_idx], example, primitives, verbose)
                conditions.append(tmp_arg)

            operations = []
            for arg_idx in range(conditions_range, len(token_args) - 1):
                tmp_arg = resolve_arg(token_args[arg_idx], example, primitives, verbose)
                operations.append(tmp_arg)

            resolved_args.append(conditions)
            resolved_args.append(operations)
            resolved_args.append(otherwise)
            example_result = prim_func(*resolved_args)
        elif prim_name == 'del':
            result.append(DeleteAction(token_args[-1] - len(primitives.semantics)))
            is_del = True
        elif prim_name == 'get_objects' or prim_name == 'get_bg':
            # Special handling for get_objects and get_bg: they need the grid and object_mask
            # The first argument should be the grid (N+0, which is example[0])
            if len(token_args) == 0:
                # No arguments means use N+0 (the first state variable, which is the input grid)
                grid = example[0]
            else:
                # Resolve the grid argument
                grid = resolve_arg(token_args[0], example, primitives, verbose)
            
            # Convert object_mask to the right format
            # Treat empty list as None (no mask provided)
            if object_mask is not None:
                import numpy as np
                if isinstance(object_mask, np.ndarray):
                    if object_mask.size == 0:
                        obj_mask = None
                    else:
                        obj_mask = object_mask.tolist()
                elif isinstance(object_mask, list):
                    if len(object_mask) == 0:
                        obj_mask = None
                    else:
                        obj_mask = object_mask
                else:
                    obj_mask = None
            else:
                obj_mask = None
            
            if obj_mask is None:
                # If no object_mask provided, create an empty one (all zeros)
                if isinstance(grid, primitives.GridObject):
                    grid_array = grid.to_grid()
                    obj_mask = [[0] * grid_array.shape[1] for _ in range(grid_array.shape[0])]
                else:
                    obj_mask = [[0] * len(grid[0]) for _ in range(len(grid))]
            
            # Execute get_objects or get_bg with grid and object_mask
            example_result = prim_func(grid, obj_mask)
        else:
            for arg in token_args:
                tmp_arg = resolve_arg(arg, example, primitives, verbose)
                resolved_args.append(tmp_arg)
            example_result = prim_func(*resolved_args)

        if not is_del:
            result.append(example_result)

    return result

def is_neural_primitive(token_seq, primitives):
    '''
    Check if the first instruction in the program is a neural primitive (get_objects or get_bg).
    
    @param token_seq: a token sequence representing an instruction step
    @param primitives: the DSL
    
    @return: (is_neural, prim_name) tuple
    '''
    token_tuple = ProgUtils.convert_token_seq_to_token_tuple(token_seq, primitives)
    prim_idx = token_tuple[0]
    prim_name = primitives.inverse_lookup(prim_idx)
    
    if prim_name in ['get_objects', 'get_bg']:
        return True, prim_name
    else:
        return False, prim_name

def kickstart_neural_primitive_program(state, primitives, obj_masks):
    # Add the output to the state
    for k_idx in range(len(state)):

        state_id1 = primitives.get_objects(state[k_idx][0], obj_masks[k_idx])
        state_id2 = primitives.get_bg(state[k_idx][0], obj_masks[k_idx])

        state[k_idx].append(state_id1)
        state[k_idx].append(state_id2)
    
    # Execute the rest of the program starting from step 2 (index 1)
    return state

def execute(token_seq_list, state, primitives, object_mask=None):
    '''
    This function executes a whole program in token sequence format.

    @param token_seq_list: a list of token sequences representing the whole program (decoder output format).
    @param primitives: the DSL
    @param object_mask: Optional object mask (2D numpy array or list) for get_objects/get_bg

    @return the output of the program, necessarily a Grid.
    '''
    for step_idx, _ in enumerate(token_seq_list):
        token_tuple = ProgUtils.convert_token_seq_to_token_tuple(token_seq_list[step_idx], primitives)

        # If end of program instruction step, return previous output.
        if token_tuple[0] == -1:
            return state[-1]

        output = execute_step(token_tuple, state, primitives, object_mask)
        
        if isinstance(output[0], DeleteAction):
            #print("==> Output: DeleteAction")
            idx_to_remove = output[0].state_idx

            # Delete the element at idx_to_remove from the state
            for k_idx in range(len(state)):
                state[k_idx] = [s for i, s in enumerate(state[k_idx]) if i != idx_to_remove]
        else:
            #print("==> Output: ")
            for k_idx in range(len(state)):
                #print(output[k_idx])
                state[k_idx].append(output[k_idx])

    if len(state[0]) > 1:
        print("==> WARNING: final state contains more than one values! Suggests missing memory management primitives!")

    last_states = []
    for k_idx in range(len(state)):
        last_states.append(state[k_idx][-1])
    return last_states

def execute_instruction_step_batch(instr_step_batch, intermediate_state_batch, primitives):
    batch_outputs = []

    batch_size = len(instr_step_batch)
    for idx in range(batch_size):
        instr_step = instr_step_batch[idx]
        intermediate_state = intermediate_state_batch[idx]

        if instr_step[0] == ProgUtils.EOS_TOKEN:
            outputs = []
            for k in range(len(intermediate_state_batch)):
                outputs.append(None)

            batch_outputs.append(outputs)
        else:
            tmp_output = execute_instruction_step(instr_step, intermediate_state, primitives)
            batch_outputs.append(tmp_output)

    return batch_outputs


def execute_instruction_step(instr_step, intermediate_state, primitives, verbose=False):

    if ProgUtils.validate_instr_step(instr_step):

        try:
            # Convert tensor to list if needed
            if torch.is_tensor(instr_step):
                instr_step = instr_step.tolist()
            
            if verbose:
                print("Instruction step: ", instr_step)
            token_tuple = ProgUtils.convert_token_seq_to_token_tuple(instr_step, primitives)

            if verbose:
                print("Token tuple representation: ", token_tuple)

            str_token_tuple = ProgUtils.convert_token_tuple_to_str(token_tuple, primitives)

            if verbose:
                print("String representation of token tuple: ", str_token_tuple)

            if verbose:
                print("intermediate_state = ", intermediate_state)

            prog_output = execute_step(token_tuple, intermediate_state, primitives)
            if verbose:
                print("Instruction step output = ", prog_output)

            if prog_output is None:
                print("ERROR: execute_step output is None!")
                exit(-1)
                
            return prog_output
        except:
            # Capture and display the traceback of the exception
            if verbose:
                import traceback
                print("Exception occurred during instruction step execution:")
                traceback.print_exc()

            return None
    
    else:
        if verbose:
            print("==> ERROR: program is invalid!")
        # INSERT_YOUR_CODE
        import traceback
        print("Traceback (most recent call last):")
        traceback.print_stack()
        return None


