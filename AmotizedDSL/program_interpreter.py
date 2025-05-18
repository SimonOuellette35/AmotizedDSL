from typing import List, Tuple
from AmotizedDSL.prog_utils import ProgUtils
from AmotizedDSL.DSL import Pixel
import torch


class DeleteAction:

    def __init__(self, state_idx):
        self.state_idx = state_idx


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
            # Convert list of (x,y,c) tuples into list of Pixel instances
            if isinstance(parent_obj, List):
                pixels = []
                for obj in parent_obj:
                    tmp_pixels = [Pixel(x, y, c) for x, y, c in obj.pixels]    
                    pixels.append(tmp_pixels)
            else:
                pixels = [Pixel(x, y, c) for x, y, c in parent_obj.pixels]
                
            output = attr_func(pixels)
        else:
            output = attr_func(parent_obj)

        return output

    else:
        if arg < 10:
            # integer constant
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

def execute_step(step_token_seq, states, primitives, verbose=True):
    '''
    @param step_token_seq: a tuple of tokens (integers) representing the step to execute.
    @param state: List of states, each representing the output of the previous tokens and whether it was resolved or not.

    @return Returns the new intermediate state after executing the step.
    '''

    # Step 1: resolve the main primitive
    prim_idx = step_token_seq[0]
    prim_name = primitives.inverse_lookup(prim_idx)
    prim_func = primitives.semantics[prim_name]
    
    # Step 2: parse the arguments (if any)
    token_args = step_token_seq[1]
    
    resolved_args = []

    if prim_name == 'switch':
        # 'switch' is a special statement, in that the number of arguments is dynamic, and
        # some logic must be used to determine which are the conditions and which are the operations.
        otherwise = resolve_arg(token_args[-1], states, primitives, verbose)

        conditions_range = (len(token_args) - 1) // 2

        conditions = []
        for arg_idx in range(conditions_range):
            tmp_arg = resolve_arg(token_args[arg_idx], states, primitives, verbose)
            conditions.append(tmp_arg)

        operations = []
        for arg_idx in range(conditions_range, len(token_args) - 1):
            tmp_arg = resolve_arg(token_args[arg_idx], states, primitives, verbose)
            operations.append(tmp_arg)

        resolved_args.append(conditions)
        resolved_args.append(operations)
        resolved_args.append(otherwise)
    elif prim_name == 'del':
        return DeleteAction(token_args[-1] - len(primitives.semantics))
    else:
        for arg in token_args:
            tmp_arg = resolve_arg(arg, states, primitives, verbose)
            resolved_args.append(tmp_arg)

    # Step 3: Execute the instruction step
    result = prim_func(*resolved_args)
    
    return result

def execute(token_seq_list, state, primitives):
    '''
    This function executes a whole program in token sequence format.

    @param token_seq_list: a list of token sequences representing the whole program (decoder output format).
    @param primitives: the DSL

    @return the output of the program, necessarily a Grid.
    '''
    for step_idx, _ in enumerate(token_seq_list):
        token_tuple = ProgUtils.convert_token_seq_to_token_tuple(token_seq_list[step_idx], primitives)

        # If end of program instruction step, return previous output.
        if token_tuple[0] == -1:
            return state[-1]
        
        output = execute_step(token_tuple, state, primitives)

        if isinstance(output, DeleteAction):
            idx_to_remove = output.state_idx

            # Delete the element at idx_to_remove from the state
            state = [s for i, s in enumerate(state) if i != idx_to_remove]
        else:
            state.append(output)

    return state[-1]


def execute_instruction_step_batch(instr_step_batch, intermediate_state_batch, primitives):
    batch_outputs = []

    batch_size = len(instr_step_batch)
    for idx in range(batch_size):
        instr_step = instr_step_batch[idx]
        intermediate_state = intermediate_state_batch[idx]

        if instr_step[0] == ProgUtils.EOS_TOKEN:
            batch_outputs.append(None)
        else:
            tmp_output = execute_instruction_step(instr_step, intermediate_state, primitives)
            batch_outputs.append(tmp_output)

    return batch_outputs


def execute_instruction_step(instr_step, intermediate_state, primitives, verbose=False, max_int=55):

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

            # validate output. For example, if it's a list of integers and it contains negative
            # integers, it's certainly an incorrect program (at least right now in the current
            # scope of problems)
            # Validate output for lists of integers
            if isinstance(prog_output, list) and all(isinstance(x, int) for x in prog_output):
                for val in prog_output:
                    if val < 0 or val > max_int:
                        if verbose:
                            print(f"==> ERROR: Integer value {val} out of valid range [0, {max_int}]")
                        return None

            return prog_output
        except:
            # Capture and display the traceback of the exception
            return None
    
    else:
        print("==> ERROR: program is invalid!")
        return None


