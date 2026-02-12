from typing import List, Tuple
from AmotizedDSL.prog_utils import ProgUtils
import torch
from AmotizedDSL.delete_action import DeleteAction
import numpy as np


# Cache for pre-compiled programs (token_seq_list -> list of token tuples)
_program_cache = {}


def _compile_program(token_seq_list, primitives):
    """Pre-compile a program from token sequences to token tuples."""
    return [ProgUtils.convert_token_seq_to_token_tuple(seq, primitives) for seq in token_seq_list]


def resolve_arg(arg, states, primitives, semantics_len, attr_cache=None, verbose=True):
    if isinstance(arg, tuple):
        object_arg = arg[0]
        attr_arg = arg[1]

        # resolve the object reference
        ref_idx = object_arg - semantics_len
        parent_obj = states[ref_idx]

        # resolve the attribute
        # handle pixel attributes differently than grid-level attributes
        if attr_cache is not None and attr_arg in attr_cache:
            attr_name, attr_func = attr_cache[attr_arg]
        else:
            attr_name = primitives.inverse_lookup(attr_arg)
            attr_func = primitives.semantics[attr_name]
            if attr_cache is not None:
                attr_cache[attr_arg] = (attr_name, attr_func)

        if attr_name in primitives.pixel_attributes:
            if isinstance(parent_obj, list):
                pixels = []
                for obj in parent_obj:
                    if isinstance(obj, primitives.GridObject):
                        pixels.append(obj.pixels)
                    else:
                        pixels.append(obj)
            else:
                pixels = parent_obj.pixels
                
            output = attr_func(pixels)
        else:
            output = attr_func(parent_obj)

        return output

    else:
        if arg < 10:
            return arg
        elif arg >= semantics_len:
            # resolve the reference
            ref_idx = arg - semantics_len
            return states[ref_idx]
        else:
            print("==> BUG: this case shouldn't happen in resolve_arg.")
            exit(-1)

def get_num_args(prim: int, DSL):
    return DSL.arg_counts[prim]

def execute_step(step_token_seq, states, primitives, object_mask=None, prim_cache=None, attr_cache=None, verbose=True, object_mask_override=None):
    '''
    @param step_token_seq: a tuple of tokens (integers) representing the step to execute.
    @param state: [k, num states, variable]
    @param object_mask: Optional list of k object masks (each is a 2D numpy array or list) for get_objects/get_bg
    @param object_mask_override: Optional list of k masks to use instead of object_mask for this step (e.g. sub_obj_masks for second get_objects)
    @param prim_cache: Cache dict for primitive index -> (name, func) lookups
    @param attr_cache: Cache dict for attribute index -> (name, func) lookups

    @return Returns the new intermediate state after executing the step.
    '''
    # Use override for this step when provided (e.g. sub_object_mask_list for second get_objects)
    mask_for_step = object_mask_override if object_mask_override is not None else object_mask

    # Step 1: resolve the main primitive (with caching)
    prim_idx = step_token_seq[0]
    if prim_cache is not None and prim_idx in prim_cache:
        prim_name, prim_func = prim_cache[prim_idx]
    else:
        prim_name = primitives.inverse_lookup(prim_idx)
        prim_func = primitives.semantics[prim_name]
        if prim_cache is not None:
            prim_cache[prim_idx] = (prim_name, prim_func)
    
    # Step 2: parse the arguments (if any)
    token_args = step_token_seq[1]
    
    # Cache semantics length to avoid repeated lookups
    semantics_len = len(primitives.semantics)
    
    needs_obj_mask = (prim_name == 'get_objects' or prim_name == 'get_bg')

    result = []
    is_del = False
    for k_idx, example in enumerate(states):
        resolved_args = []
        example_result = None

        if prim_name == 'switch':
            # 'switch' is a special statement, in that the number of arguments is dynamic, and
            # some logic must be used to determine which are the conditions and which are the operations.
            conditions_range = (len(token_args) - 1) // 2
            
            conditions = [resolve_arg(token_args[arg_idx], example, primitives, semantics_len, attr_cache, verbose) 
                         for arg_idx in range(conditions_range)]
            
            operations = [resolve_arg(token_args[arg_idx], example, primitives, semantics_len, attr_cache, verbose) 
                        for arg_idx in range(conditions_range, len(token_args) - 1)]
            
            otherwise = resolve_arg(token_args[-1], example, primitives, semantics_len, attr_cache, verbose)
            
            resolved_args.append(conditions)
            resolved_args.append(operations)
            resolved_args.append(otherwise)
            example_result = prim_func(*resolved_args)
        elif prim_name == 'del':
            result.append(DeleteAction(token_args[-1] - semantics_len))
            is_del = True
        elif needs_obj_mask:
            # Special handling for get_objects and get_bg: they need the grid and object_mask
            # The first argument should be the grid (N+0, which is example[0])
            if len(token_args) == 0:
                # No arguments means use N+0 (the first state variable, which is the input grid)
                grid = example[0]
            else:
                # Resolve the grid argument
                grid = resolve_arg(token_args[0], example, primitives, semantics_len, attr_cache, verbose)
            
            if isinstance(object_mask_override, list):
                # Sub object masks are being passed as parameter, loop over object/sub-obj-mask and collect results
                obj_mask = mask_for_step[k_idx]
                print(f"==> obj_mask[{k_idx}]: {obj_mask}")
                example_result = []
                for sub_obj_idx in range(len(object_mask_override[k_idx])):
                    obj_mask = obj_mask[sub_obj_idx]

                    print(f"==> sub-object #{sub_obj_idx}: {obj_mask}")

                    # List of sub_obj_masks (second get_objects)
                    current_obj_mask = [m.tolist() if isinstance(m, np.ndarray) else m for m in obj_mask]

                    print(f"==> current_obj_mask: {current_obj_mask}")

                    # Execute get_objects or get_bg with grid and object_mask
                    tmp_result = prim_func(grid[sub_obj_idx], current_obj_mask)
                    
                    print(f"==> tmp_result: {tmp_result}")
                    example_result.append(tmp_result)

                print(f"==> example_result: {example_result}")

            else:
                # Get the object mask for this example (k_idx); use mask_for_step (object_mask or override)
                if mask_for_step is not None and k_idx < len(mask_for_step):
                    obj_mask = mask_for_step[k_idx]
                    # Convert numpy array to list if needed; for sub_obj_masks (list of masks), convert each element too
                    if isinstance(obj_mask, np.ndarray):
                        current_obj_mask = obj_mask.tolist()
                    else:
                        current_obj_mask = obj_mask
                else:
                    # If no object_mask provided, create an empty one (all zeros)
                    if isinstance(grid, primitives.GridObject):
                        grid_array = grid.to_grid()
                        current_obj_mask = [[0] * grid_array.shape[1] for _ in range(grid_array.shape[0])]
                    else:
                        current_obj_mask = [[0] * len(grid[0]) for _ in range(len(grid))]
            
                # Execute get_objects or get_bg with grid and object_mask
                example_result = prim_func(grid, current_obj_mask)
        else:
            resolved_args = [resolve_arg(arg, example, primitives, semantics_len, attr_cache, verbose) 
                           for arg in token_args]
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

def execute(token_seq_list, state, primitives, object_mask=None, debug_info=None, sub_object_mask_list=None):
    '''
    This function executes a whole program in token sequence format.

    @param token_seq_list: a list of token sequences representing the whole program (decoder output format).
    @param primitives: the DSL
    @param object_mask: Optional list of k object masks (each is a 2D numpy array or list) for get_objects/get_bg
    @param sub_object_mask_list: Optional list of k sub-object mask lists (for second get_objects: one list of masks per object per example)

    @return the output of the program, necessarily a Grid.
    '''
    # Pre-compile program: convert all token sequences to token tuples once
    # Use cache key based on program content (tuple of token sequence tuples for hashing)
    cache_key = tuple(tuple(seq) for seq in token_seq_list)
    if cache_key in _program_cache:
        compiled_program = _program_cache[cache_key]
    else:
        compiled_program = _compile_program(token_seq_list, primitives)
        _program_cache[cache_key] = compiled_program
    
    # For each step, determine if it is get_objects and which occurrence (1st, 2nd, ...). Use sub_object_mask_list for 2nd+ get_objects.
    step_mask_override = []
    get_objects_count = 0
    for token_tuple in compiled_program:
        if token_tuple[0] == -1:
            step_mask_override.append(None)
            continue
        prim_idx = token_tuple[0]
        prim_name = primitives.inverse_lookup(prim_idx)
        if prim_name == 'get_objects':
            get_objects_count += 1
            step_mask_override.append(sub_object_mask_list if get_objects_count >= 2 and sub_object_mask_list else None)
        else:
            step_mask_override.append(None)
    
    # Build lookup caches for primitives and attributes (reused across all steps)
    prim_cache = {}
    attr_cache = {}
    
    for step_i, token_tuple in enumerate(compiled_program):
        # If end of program instruction step, return previous output.
        if token_tuple[0] == -1:
            return state[-1]

        override = step_mask_override[step_i] if step_i < len(step_mask_override) else None
        print(f"==> token_tuple = {token_tuple}")
        print(f"\tOverride: {override}")
        
        output = execute_step(token_tuple, state, primitives, object_mask, prim_cache, attr_cache, object_mask_override=override)
        
        if isinstance(output[0], DeleteAction):
            #print("==> Output: DeleteAction")
            idx_to_remove = output[0].state_idx

            # Delete the element at idx_to_remove from the state
            for k_idx in range(len(state)):
                del state[k_idx][idx_to_remove]
        else:
            #print("==> Output: ")
            for k_idx in range(len(state)):
                #print(output[k_idx])
                state[k_idx].append(output[k_idx])

    if len(state[0]) > 1:
        if debug_info is not None:
            print(f"==> WARNING [Task: {debug_info['task_name']}]: final state contains more than one values! Suggests missing memory management primitives!")
        else:
            print("==> WARNING: final state contains more than one values! Suggests missing memory management primitives!")

    return [state[k_idx][-1] for k_idx in range(len(state))]

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

            # Build caches for this single-step execution
            prim_cache = {}
            attr_cache = {}
            prog_output = execute_step(token_tuple, intermediate_state, primitives, None, prim_cache, attr_cache)
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


