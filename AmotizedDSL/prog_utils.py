from typing import List, Tuple, Union
import re
import AmotizedDSL.DSL as DSL
import inspect
import traceback
import numpy as np


class ProgUtils:

    '''
    This class implements utility functions that manipulate program representations. In particular, it offers conversion methods between the
    different program representations. They are:

    1 - hand-written representation: this format exists to facilitate manually describing programs. It is a list of instruction steps forming a
    whole program. Each instruction step is a tuple of (primitive, arguments). In hand-written representation, each token is a text string,
    with the exception of reference IDs that are integers.

    2 - intermediate representation: in this format, we still have tuples as in hand-written representation, but the text strings have been resolved
    to the indices in the DSL for each primitive. This format is used as input to the execute_step function in the program interpreter, because it's 
    easier to process.

    3 - token sequence: this format is directly what is outputted by the decoding process. Here, the token IDs (integers) are offset relative to the
    primitive indices in the DSL, because there are special tokens to help structure the output. The token sequence format is, for each instruction
    step (decoding phase):

        [SOS, primitive, SOP, arg1(, attr), ARG_SEP, arg2(, attr), ARG_SEP, ..., EOS]

    4 - LLM format: this format is used to interface with an LLM.
    '''

    SOS_TOKEN = 0           # Start of sentence
    SOP_TOKEN = 1           # Start of parameters
    ARG_SEP_TOKEN = 2       # Argument separator
    EOS_TOKEN = 3           # End of sentence

    NUM_SPECIAL_TOKENS = 4

    TYPE_GRIDOBJECT = 0
    TYPE_INT = 1
    TYPE_BOOL = 2
    TYPE_PIXEL = 3
    TYPE_LIST_GRIDOBJECT = 4
    TYPE_LIST_INT = 5
    TYPE_LIST_BOOL = 6
    TYPE_LIST_PIXEL = 7
    NUM_TYPE_TOKENS = 8  # Number of distinct type tokens

    @staticmethod
    def get_variable_type_code(var):
        """
        Convert a variable from intermediate state to an integer type code.
        
        Args:
            var: A variable from intermediate state (Grid, GridObject, int, bool, Pixel, or list)
            
        Returns:
            Integer code representing the variable type
        """
        if var is None:
            return ProgUtils.TYPE_GRIDOBJECT  # Default to int for None
        
        type_str = str(type(var))
        
        if 'list' in type_str.lower() or isinstance(var, list) or isinstance(var, np.ndarray):
            if len(var) == 0:
                return ProgUtils.TYPE_LIST_GRIDOBJECT  # Default for empty list

            # Check element type - handle nested lists by checking first element recursively
            first_elem = var[0]

            # If first element is also a list/array, check its first element
            first_elem_type_str = str(type(first_elem))
            if 'list' in first_elem_type_str.lower() or isinstance(first_elem, list) or isinstance(first_elem, np.ndarray):
                if len(first_elem) == 0:
                    return ProgUtils.TYPE_LIST_GRIDOBJECT
                # Check the nested element type
                nested_elem = first_elem[0]
                nested_elem_type = str(type(nested_elem))
                # Check if nested element is also a list (third level of nesting)
                if 'list' in nested_elem_type.lower() or isinstance(nested_elem, list) or isinstance(nested_elem, np.ndarray):
                    if len(nested_elem) == 0:
                        return ProgUtils.TYPE_LIST_PIXEL  # Default for triple nested empty list
                    # Check the triple nested element type
                    triple_nested_elem = nested_elem[0]
                    triple_nested_elem_type = str(type(triple_nested_elem))
                    if 'GridObject' in triple_nested_elem_type:
                        return ProgUtils.TYPE_LIST_GRIDOBJECT
                    elif 'Pixel' in triple_nested_elem_type:
                        return ProgUtils.TYPE_LIST_PIXEL
                    elif 'int' in triple_nested_elem_type:
                        return ProgUtils.TYPE_LIST_INT
                    elif 'bool' in triple_nested_elem_type:
                        return ProgUtils.TYPE_LIST_BOOL
                    else:
                        print(f"==> ERROR: unknown variable type {triple_nested_elem_type} (nested in list of list)!")
                        exit(-1)
                elif 'GridObject' in nested_elem_type:
                    return ProgUtils.TYPE_LIST_GRIDOBJECT
                elif 'Pixel' in nested_elem_type:
                    return ProgUtils.TYPE_LIST_PIXEL
                elif 'int' in nested_elem_type:
                    return ProgUtils.TYPE_LIST_INT
                elif 'bool' in nested_elem_type:
                    return ProgUtils.TYPE_LIST_BOOL
                else:
                    print(f"==> ERROR: unknown variable type {nested_elem_type} (nested in list)!")
                    exit(-1)
            else:
                # First element is not a list, check its type directly
                elem_type = str(type(first_elem))
                if 'GridObject' in elem_type:
                    return ProgUtils.TYPE_LIST_GRIDOBJECT
                elif 'Pixel' in elem_type:
                    return ProgUtils.TYPE_LIST_PIXEL
                elif 'int' in elem_type:
                    return ProgUtils.TYPE_LIST_INT
                elif 'bool' in elem_type:
                    return ProgUtils.TYPE_LIST_BOOL
                else:
                    print(f"==> ERROR: unknown variable type {elem_type} (element in list)!")
                    exit(-1)
        elif 'GridObject' in type_str:
            return ProgUtils.TYPE_GRIDOBJECT
        elif 'Pixel' in type_str:
            return ProgUtils.TYPE_PIXEL
        elif 'int' in type_str:
            return ProgUtils.TYPE_INT
        elif 'bool' in type_str:
            return ProgUtils.TYPE_BOOL
        else:
            print(f"==> ERROR: unknown variable type {type_str}!")
            exit(-1)

    @staticmethod
    def validate_instruction(instr, intermediate_state):
        '''
        This validates instruction 'instr' based on the current program state contained in 'intermediate_state'.

        Intermediate state is expected to be of shape [k, number of variables]

        We assume that all instances of the k demonstration sets have the same number of variables, since it's the
        same program applied to all k instances (and thus the same program depth == same variable stack size).
        '''
        num_vars = len(intermediate_state[0])

        # This is the delete instruction
        if len(instr) >= 3 and instr[0] == 0 and instr[1] == 51 and instr[2] == 1:
            if num_vars == 1:
                return False, "cannot have a del instruction when there is only 1 state variable."
            else:
                ref_idx = instr[3] - ProgUtils.NUM_SPECIAL_TOKENS - len(DSL.semantics)
                if ref_idx == len(intermediate_state) - 1:
                    return False, "cannot delete the output of the last instruction (because it annuls it)."

        # Make it impossible for reference IDs to refer to values > num_vars, so eliminate some possibilities here as well
        # Remove from valid_instructions all instruction sequences that contain 61+num_vars or higher
        threshold_val = 61 + num_vars
        if any(isinstance(token, int) and token >= threshold_val for token in instr):
            return False, f"impossible for reference IDs to refer to ids > {num_vars}"

        # Extract the instruction sequence's arguments
        arguments = ProgUtils.parse_arguments(instr)

        # Some primitives have special validation rules.
        primitive_idx = instr[1] - ProgUtils.NUM_SPECIAL_TOKENS
        prim_name = DSL.inverse_lookup(primitive_idx)

        if prim_name == 'new_grid':
            if not ProgUtils.validate_new_grid_statement(arguments):
                return False, 'new_grid cannot have 0 as its first or second argument.'

        if prim_name == 'crop':
            if not ProgUtils.validate_crop_statement(arguments):
                return False, 'crops arguments ref.x, ref.y or ref.c are invalid.'

        if prim_name == 'div' or prim_name == 'mod':
            if not ProgUtils.validate_div_mod_statement(arguments):
                return False, 'division or modulo by zero not allowed.'

        # Extract the data types
        arg_types, is_list = ProgUtils.extract_data_types(arguments, intermediate_state[0])
        
        if not ProgUtils.validate_attr_usage(arguments, intermediate_state[0]):
            return False, "cannot use attributes on non-GridObject references."

        is_valid, error_msg = ProgUtils.check_state_variable_types(arg_types, is_list, arguments, instr[1])
        
        return is_valid, error_msg

    @staticmethod
    def validate_crop_statement(arguments):

        x_attr = DSL.prim_indices['.x']
        y_attr = DSL.prim_indices['.y']
        c_attr = DSL.prim_indices['.c']

        for arg in arguments:
            if len(arg) == 2:
                attr_idx = arg[1] - ProgUtils.NUM_SPECIAL_TOKENS
                if attr_idx == x_attr or attr_idx == y_attr or attr_idx == c_attr:
                    return False

        return True

    @staticmethod
    def validate_div_mod_statement(arguments):

        for arg in arguments:
            if len(arg) == 1:
                arg_val = arg[0] - ProgUtils.NUM_SPECIAL_TOKENS
                if arg_val == 0:
                    return False

        return True

    @staticmethod
    def validate_attr_usage(arguments, intermediate_state):

        for arg in arguments:
            if len(arg) > 1:
                obj_id = arg[0]
                ref_idx = obj_id - len(DSL.semantics) - ProgUtils.NUM_SPECIAL_TOKENS
                ref_obj = intermediate_state[ref_idx]
                type_str = str(type(ref_obj))
                if 'GridObject' not in type_str:
                    return False

        return True


    @staticmethod
    def parse_arguments(instr: List[int]) -> List[List[int]]:
        """
        Parse arguments from a program instruction sequence.
        - 1 marks the start of arguments list
        - 2 separates arguments
        - 3 marks end of sequence (or padding)
        Returns a list of argument token lists.
        """
        try:
            start_idx = instr.index(1)
        except ValueError:
            return []

        args: List[List[int]] = []
        current: List[int] = []
        i = start_idx + 1
        while i < len(instr):
            token = instr[i]
            if token == 3:
                if current:
                    args.append(current)
                break
            if token == 2:
                args.append(current)
                current = []
            elif token == 1:
                # Unexpected nested start marker; end current parsing
                if current:
                    args.append(current)
                break
            else:
                current.append(token)
            i += 1
        return args

    @staticmethod
    def extract_data_types(arguments, intermediate_state):
        f'''
        The list of arguments is actually a list of lists, because it's possible to have an argument "pair" (Object reference + attribute).

        Goes through the arguments, resolves references to variables in intermediate_state, and returns the
        argument type for each argument. There are 5 possible argument types (it gets simplified from GridObject or List[List[Pixels]], etc.
        because of the implicit type overloading in most primitives):
            - GridObject
            - Pixel
            - int
            - bool
        
        We also return whether the argument is a list or not, because in some cases a List must be passed (e.g. get_index, count_items, etc.)
        '''
        arg_types = []
        is_list = []

        for arg in arguments:
            arg_val = arg[0] - ProgUtils.NUM_SPECIAL_TOKENS
            if arg_val < 10:
                # Constant
                # Add int type to arg_types
                arg_types.append(int)
                is_list.append(False)
            else:
                if len(arg) > 1:
                    # Attribute reference, this is always an int
                    arg_types.append(List[int])
                    
                    attr_idx = arg[1] - ProgUtils.NUM_SPECIAL_TOKENS
                    attr_name = DSL.inverse_lookup(attr_idx)

                    if attr_name == '.x' or attr_name == '.y' or attr_name == '.c':
                        is_list.append(True)
                    else:
                        ref_idx = arg[0] - len(DSL.semantics) - ProgUtils.NUM_SPECIAL_TOKENS    
                        obj = intermediate_state[ref_idx]

                        type_str = str(type(obj))
                        if 'list' in type_str.lower():
                            is_list.append(True)
                        else:
                            is_list.append(False)
                else:
                    # direct object reference, just find the type of the referred variable
                    ref_idx = arg_val - len(DSL.semantics)
                    obj = intermediate_state[ref_idx]

                    # parse and simplify the possible types
                    type_str = str(type(obj))
                    if 'list' in type_str.lower():
                        is_list.append(True)
                    else:
                        is_list.append(False)

                    if 'GridObject' in type_str:
                        arg_types.append(List[DSL.GridObject])
                    elif 'int' in type_str:
                        arg_types.append(List[int])
                    elif 'bool' in type_str:
                        arg_types.append(List[bool])
                    elif 'Pixel'in type_str:
                        arg_types.append(List[DSL.Pixel])
                    elif 'list' in type_str:
                        elem_type = str(type(obj[0]))
                        if 'int' in elem_type:
                            arg_types.append(List[int])
                        elif 'bool' in elem_type:
                            arg_types.append(List[bool])
                        elif 'Pixel' in elem_type:
                            arg_types.append(List[DSL.Pixel])
                        elif 'GridObject' in elem_type:
                            arg_types.append(List[DSL.GridObject])
                        elif 'list' in elem_type:
                            elem_type = str(type(obj[0][0]))
                            if 'int' in elem_type:
                                arg_types.append(List[int])
                            elif 'bool' in elem_type:
                                arg_types.append(List[bool])
                            elif 'Pixel' in elem_type:
                                arg_types.append(List[DSL.Pixel])
                            elif 'GridObject' in elem_type:
                                arg_types.append(List[DSL.GridObject])
                            else:
                                print(f"==> Error: unknown list of list element type: {elem_type}")
                        else:
                            print(f"==> Error: unknown list element type: {elem_type}")
                    else:
                        print(f"==> Error: unknown data type: {type_str}")

        return arg_types, is_list

    @staticmethod
    def extract_var_types(arguments, var_types):
        """
        For each argument in 'arguments', determines the type directly from the corresponding 'var_types' entry.
        Also returns whether the argument is a list or not, based on the resolved type.
        If an argument is a constant (index < 10 after offset), it is treated as an int (not a list).

        var_types: list of types (e.g., int, List[int], DSL.GridObject, etc.) corresponding to variables in intermediate_state.
        """

        arg_types = []

        for arg in arguments:
            arg_val = arg[0] - ProgUtils.NUM_SPECIAL_TOKENS
            if arg_val < 10:
                # Constant: Always int (not a list)
                arg_types.append(ProgUtils.TYPE_INT)
            else:
                if len(arg) > 1:
                    # Attribute reference: always List[int]
                    attr_idx = arg[1] - ProgUtils.NUM_SPECIAL_TOKENS
                    attr_name = DSL.inverse_lookup(attr_idx)
                    if attr_name in ('.x', '.y', '.c'):
                        arg_types.append(ProgUtils.TYPE_LIST_INT)
                    else:
                        arg_types.append(ProgUtils.TYPE_INT)
                else:
                    # Direct variable reference, lookup from var_types
                    # Variables in var_types are indexed after DSL.semantics
                    var_idx = arg_val - len(DSL.semantics)
                    typ = var_types[var_idx]
                    arg_types.append(typ)

        return arg_types

    # @staticmethod
    # def infer_from_parameter_lists():
    #     # TODO: 5) there are list and non-list examples? If so, must distinguish somehow which is being returned!
    #     # TODO: 6) to do this, check the parameter types used. If they are all List, easy: it's the list type.
    #     # TODO: 7) if, none of them is list, easy: it's the non-list type.
    #     # TODO: 8) if it's a mix, must find the parameters of the same type as the return type. Are they list? If so, it's the list type,
    #     #  if not, it's not the list type.
    #     #   Or rather: it any is List, the output is List? Actually depends on the primitive's logic? Examples:
    #     #   - less_than(a: Union[int, List[int], List[List[int]]], b: Union[int, List[int], List[List[int]]]) -> Union[bool, List[bool], List[List[bool]]]:
    #     #
    #     #       ==> if any is list, use list, else use non-list.
    #     #
    #     #   - get_index(list: List[T], i: int) -> Union[T, List[T]]:
    #     #
    #     #       ==> T if list is List[T], List[T] is list is List[List[T]]!
    #     #
    #     #   - addition(a: Union[int, List[int], List[List[int]]], b: Union[int, List[int], List[List[int]]]) -> Union[int, List[int]]:
    #     #
    #     #       ==> if any is list, use list, else use non-list.
    #     #
    #     #   - equal(a: Union[int, List[int]], b: Union[int, List[int]]) -> Union[bool, List[bool]]:
    #     #
    #     #       ==> if any is list, use list, else use non-list.
    #     #
    #     #   - set_pixels(target_grid: Union[GridObject, List[GridObject]], set_x: Union[DIM, List[DIM], List[List[DIM]]],  set_y: Union[DIM, List[DIM], List[List[DIM]]],
    #     #       colors: Union[COLOR, List[COLOR], List[List[COLOR]]]) -> Union[GridObject, List[GridObject]]:
    #     #
    #     #       ==> if target_grid is GridObject, use GridObject, else use List
    #     #
    #     #   - crop(g: Union[GridObject, List[GridObject]], x1: Union[int, List[int]], y1: Union[int, List[int]], x2: Union[int, List[int]], 
    #     #       y2: Union[int, List[int]]) -> Union[GridObject, List[GridObject]]:
    #     #
    #     #       ==> if g is GridObject, use GridObject, else use List
    #     #
    #     #   Heuristic:
    #     #       --> actually construct an example argument of the same type as in the instruction sequence, execute the command, and get the resulting output?


    # @staticmethod
    # def infer_result_type(instruction_seq, state_var_types):
    #     # 1) get the primitive function
    #     primitive_idx = instruction_seq[1] - ProgUtils.NUM_SPECIAL_TOKENS

    #     # TODO: special case: switch

    #     prim_name = DSL.inverse_lookup(primitive_idx)
    #     prim_func = DSL.semantics[prim_name]

    #     # 2) inspect the primitive function's signature, especially its return type
    #     annotations = prim_func.__annotations__
    #     result_ann = annotations["return"]

    #     # TODO: if return type is T, it must take the actual type of the T parameters!

    #     # 3) if it's not a Union, this directly gives us the answer.
    #     if "Union" not in result_ann:
    #         # TODO: re-use such a method? Do I already have one?
    #         return convert_to_type_int(result_ann)
    #     else:
    #     # 4) if it's a Union:
    #         # List all of the types inside the Union brackets in result_ann
    #         start = result_ann.find('[') + 1
    #         end = result_ann.rfind(']')
    #         union_types = [t.strip() for t in result_ann[start:end].split(',')]

    #         # For each type in the union, check if "list" is present (case-insensitive)
    #         all_contain_list = all('list' in t.lower() for t in union_types)
            
    #         # If ALL union types contain 'list', simply return that list type
    #         if all_contain_list:
    #             return convert_to_type_int(union_types[0])
    #         else:
    #             return infer_from_parameter_lists()

    @staticmethod
    def generate_var_examples(data_types):
        var_examples = []
        for arg_idx, data_type in enumerate(data_types):
            if data_type == ProgUtils.TYPE_BOOL:
                var_examples.append(True)
            elif data_type == ProgUtils.TYPE_INT:
                if arg_idx < 2:
                    var_examples.append(1)
                else:
                    var_examples.append(2)
            elif data_type == ProgUtils.TYPE_GRIDOBJECT:
                var_examples.append(DSL.GridObject.from_grid([
                    [0, 9, 9],
                    [0, 6, 9],
                    [6, 6, 0]
                ]))

            elif data_type == ProgUtils.TYPE_PIXEL:
                var_examples.append(DSL.Pixel(1, 1, 5))

            elif data_type == ProgUtils.TYPE_LIST_BOOL:
                if arg_idx == 0:
                    var_examples.append([True, False, True])
                elif arg_idx == 1:
                    var_examples.append([True, True, True])
                else:
                    var_examples.append([True, False, False])

            elif data_type == ProgUtils.TYPE_LIST_INT:
                var_examples.append([3 + arg_idx, 4 + arg_idx, 5 + arg_idx])

            elif data_type == ProgUtils.TYPE_LIST_GRIDOBJECT:
                var_examples.append([DSL.GridObject.from_grid([
                    [9, 9],
                    [0, 9]
                ]),
                DSL.GridObject.from_grid([
                    [0, 6],
                    [6, 6]
                ]),
                DSL.GridObject.from_grid([
                    [3, 3],
                    [3, 3]
                ])])

            elif data_type == ProgUtils.TYPE_LIST_PIXEL:
                var_examples.append([DSL.Pixel(1+arg_idx, 1+arg_idx, 5), DSL.Pixel(2, 2, 9), DSL.Pixel(1, arg_idx, 4)])
            else:
                print(f"==> ERROR: unknown data_type {data_type}")
                exit(-1)

        return var_examples

    @staticmethod
    def infer_switch_type(instruction_seq, state_var_types):
        arguments = ProgUtils.parse_arguments(instruction_seq)
        data_types = ProgUtils.extract_var_types(arguments, state_var_types)

        n_args = len(arguments)
        group_n = (n_args - 1) // 2

        ops_type = data_types[group_n]

        if ops_type == ProgUtils.TYPE_LIST_BOOL:
            return ProgUtils.TYPE_LIST_BOOL
        elif ops_type == ProgUtils.TYPE_LIST_INT:
            return ProgUtils.TYPE_LIST_INT
        elif ops_type == ProgUtils.TYPE_LIST_PIXEL:
            return ProgUtils.TYPE_LIST_PIXEL
        elif ops_type == ProgUtils.TYPE_LIST_GRIDOBJECT:
            return ProgUtils.TYPE_LIST_GRIDOBJECT

        is_list_cond = False
        if data_types[0] == ProgUtils.TYPE_LIST_BOOL:
            is_list_cond = True

        if ops_type == ProgUtils.TYPE_BOOL:
            if is_list_cond:
                return ProgUtils.TYPE_LIST_BOOL
            else:
                return ProgUtils.TYPE_BOOL
        elif ops_type == ProgUtils.TYPE_INT:
            if is_list_cond:
                return ProgUtils.TYPE_LIST_INT
            else:
                return ProgUtils.TYPE_INT
        elif ops_type == ProgUtils.TYPE_PIXEL:
            if is_list_cond:
                return ProgUtils.TYPE_LIST_PIXEL
            else:
                return ProgUtils.TYPE_PIXEL
        elif ops_type == ProgUtils.TYPE_GRIDOBJECT:
            if is_list_cond:
                return ProgUtils.TYPE_LIST_GRIDOBJECT
            else:
                return ProgUtils.TYPE_GRIDOBJECT

        print("==> ERROR: should reach here in infer_switch_type...")
        exit(-1)

    @staticmethod
    def infer_result_type(instruction_seq, state_var_types):
        primitive_idx = instruction_seq[1] - ProgUtils.NUM_SPECIAL_TOKENS

        prim_name = DSL.inverse_lookup(primitive_idx)

        if prim_name == 'get_objects':
            return ProgUtils.TYPE_LIST_GRIDOBJECT

        if prim_name == 'get_bg':
            return ProgUtils.TYPE_GRIDOBJECT

        if prim_name == 'switch':
            # Special case
            return ProgUtils.infer_switch_type(instruction_seq, state_var_types)

        prim_func = DSL.semantics[prim_name]

        arguments = ProgUtils.parse_arguments(instruction_seq)
        data_types = ProgUtils.extract_var_types(arguments, state_var_types)

        # generate variable examples and execute the instruction to get a result example
        var_examples = ProgUtils.generate_var_examples(data_types)

        try:
            result = prim_func(*var_examples)

            if result is None:
                print("ERROR: result is None!")
                exit(-1)

        except Exception as e:
            print("Exception while executing primitive function:")
            traceback.print_exc()
            print(f"Error message: {str(e)}")
            exit(-1)

        # get result type as an integer
        return ProgUtils.get_variable_type_code(result)


    @staticmethod
    def get_prim_func_arg_types(primitive_idx, nargs):
        primitive_idx -= ProgUtils.NUM_SPECIAL_TOKENS

        prim_name = DSL.inverse_lookup(primitive_idx)
        prim_func = DSL.semantics[prim_name]

        if prim_name == 'switch':
            arg_types = ['int'] * nargs
            conditions_len = int((nargs - 1) // 2)
            arg_types[:conditions_len] = ['bool'] * conditions_len

            return arg_types

        annotations = prim_func.__annotations__
        if len(annotations) == 0:
            # it's a lambda expression, in which case we return True automatically
            return ['GridObject']
        
        param_names = list(inspect.signature(prim_func).parameters.keys())

        nargs = DSL.arg_counts[primitive_idx]
        arg_types = []
        for arg_idx in range(nargs):
            arg_name = param_names[arg_idx]

            arg_type_hint = f'{annotations[arg_name]}'

            # simplify into the 4 main types: GridObject, Pixel, Int, Bool
            if 'GridObject' in arg_type_hint:
                arg_types.append('GridObject')
            elif 'int' in arg_type_hint or 'COLOR' in arg_type_hint or 'DIM' in arg_type_hint:
                arg_types.append('int')
            elif 'bool' in arg_type_hint:
                arg_types.append('bool')
            elif 'Pixel' in arg_type_hint:
                arg_types.append('Pixel')
            else:
                arg_types.append('T')

        return arg_types


    @staticmethod
    def validate_new_grid_statement(arguments):
        if arguments[0] == 4 or arguments[1] == 4:
            return False

        return True

    @staticmethod
    def validate_switch_statement(arg_types):
        n = len(arg_types)
        group_n = int((n - 1) / 2)

        conditions = arg_types[:group_n]
        operations = arg_types[group_n: group_n*2]
        otherwise = arg_types[-1]

        # Check conditions: all need to be 'bool'
        for cond_type in conditions:
            if 'bool' not in str(cond_type):
                return False, "switch statement needs bool in conditions arguments."

        # Check operations: all need to be 'int'
        for op_type in operations:
            if 'int' not in str(op_type):
                return False, "switch statement needs int in operations arguments."

        # Check otherwise: needs to be 'int'
        if 'int' not in str(otherwise):
            return False, "switch statement needs int in otherwise argument."

        return True, "switch statement valid"


    @staticmethod
    def check_state_variable_types(arg_types, is_list, arguments, primitive_idx):
        '''
        arg_types are the types of the actual arguments passed to the function. They can be:
        - GridObject
        - Pixel
        - int
        - bool
        
        and the primitive_idx is the index into the DSL for the primitive function being called.
        
        Returns True is the argument types are all valid, False if there is a mismatch.
        '''
        
        primitive_idx -= ProgUtils.NUM_SPECIAL_TOKENS

        prim_name = DSL.inverse_lookup(primitive_idx)
        prim_func = DSL.semantics[prim_name]

        if prim_name == 'switch':
            return ProgUtils.validate_switch_statement(arg_types)

        nargs = DSL.arg_counts[primitive_idx]
        if len(arg_types) != nargs:
            return False, f"ERROR: {len(arg_types)} arguments given, but the primitive has {nargs} arguments!"

        annotations = prim_func.__annotations__
        if len(annotations) == 0:
            # it's a lambda expression, in which case we return True automatically
            return True, "lambda expression, automatically valid."
        
        param_names = list(inspect.signature(prim_func).parameters.keys())

        for arg_idx, arg_type in enumerate(arg_types):
            # Attempt to extract type annotation of the argument for prim_func
            arg_name = param_names[arg_idx]
            arg_type_hint = f'{annotations[arg_name]}'
            arg_val = arguments[arg_idx]

            if 'Union' in arg_type_hint:
                inner = arg_type_hint[arg_type_hint.find('[')+1:arg_type_hint.find(']')]
                union_types = [x.strip() for x in inner.split(',')]

                # For each type in the union, check if "list" is present (case-insensitive)
                all_contain_list = all('list' in t.lower() for t in union_types)
                
                # If ALL union types contain 'list', but our argument is not a list, return type error
                if all_contain_list and not is_list[arg_idx]:
                    return False, f"ERROR: argument {arg_idx} MUST be a list, but a non-list was passed. (Union types: {union_types}, arg type: {arg_type})"
            else:
                if 'list' in arg_type_hint.lower():
                    if not is_list[arg_idx]:
                        return False, f"ERROR: argument {arg_idx} MUST be a list, but a non-list was passed."

            if 'DSL.GridObject' in arg_type_hint:
                if 'DSL.GridObject' not in f'{arg_type}':
                    return False, f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})"

            elif '~COLOR' in arg_type_hint or 'int' in arg_type_hint or '~DIM' in arg_type_hint:
                if 'int' not in f'{arg_type}':
                    return False, f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})"

            elif 'bool' in arg_type_hint:
                if 'bool' not in f'{arg_type}':
                    return False, f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})"

            elif 'Pixel' in arg_type_hint:
                if 'Pixel' not in f'{arg_type}':
                    return False, f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})"

        return True, "Data types valid."

    @staticmethod
    def validate_instr_step(instr_step):
        # Validate instruction step format
        if len(instr_step) < 4:  # Must have at least SOS, primitive, SOP, EOS
            print("==> ERROR: len(instr_step) > 4")
            return False
            
        if instr_step[0] != ProgUtils.SOS_TOKEN:  # Must start with SOS token (0)
            print("==> ERROR: instr_step[0] != SOS_TOKEN]")
            print("==> instr_step = ", instr_step)
            return False
            
        if instr_step[2] != ProgUtils.SOP_TOKEN:  # Third element must be SOP token (1) 
            print("==> ERROR: instr_step[2] != SOP_TOKEN]")
            return False
            
        if instr_step[-1] != ProgUtils.EOS_TOKEN:  # Must end with EOS token (3)
            print("==> ERROR: instr_step[-1] != EOS_TOKEN]")
            return False

        return True

    @staticmethod
    def resolve_token_str_to_token(token_str, primitives):
        '''
        This resolves a text string to a token ID for token sequence representation, meaning that the integer
        will be offset by the number of special tokens.
        '''
        if token_str is None:
            return None
        
        prim_idx = ProgUtils.resolve_token_str_to_idx(token_str, primitives)
        return prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    @staticmethod
    def resolve_token_str_to_idx(token_str, primitives):
        '''
        This resolves a text string to a DSL index for intermediate representation.
        '''
        if token_str is None:
            return None
        
        if isinstance(token_str, str):
            prim_idx = primitives.prim_indices[token_str]
            return prim_idx
        else:
            # This is a reference idx (or an integer constant)
            return token_str

    @staticmethod
    def convert_token_subseq(instr_step, primitives):
        '''
        This function converts an instruction step from hand-written format to token sequence format

        @param instr_step: a tuple of (context, primitive, arguments) in hand-written format (using text strings)
        @param primitives: the DSL
        
        @return A sequence of tokens (integers or strings for placeholders) as directly outputted by the decoder in one iteration of program generation.
        '''
        prim_name = instr_step[0]
        args = instr_step[1]

        label_seq = [ProgUtils.SOS_TOKEN]

        token_id = ProgUtils.resolve_token_str_to_token(prim_name, primitives)
        label_seq.append(token_id)
        label_seq.append(ProgUtils.SOP_TOKEN)

        def resolve_arg_to_token(arg_val):
            """Resolve an argument to a token, preserving placeholder strings."""
            # Check if this is a placeholder string (starts with "param")
            if isinstance(arg_val, str) and arg_val.startswith("param"):
                return arg_val  # Preserve placeholder strings as-is
            else:
                return ProgUtils.resolve_token_str_to_token(arg_val, primitives)

        if args is not None:
            for arg_idx, arg in enumerate(args):
                # handle object-attribute pairs
                if isinstance(arg, Tuple):
                    tok_obj_id = resolve_arg_to_token(arg[0])
                    tok_attr_id = resolve_arg_to_token(arg[1])
                    label_seq.append(tok_obj_id)
                    label_seq.append(tok_attr_id)
                else:
                    if isinstance(arg, List):
                        # Here we can assume this is a switch statement, in which lists of conditions are possible.
                        for tmp_idx, arg_elem in enumerate(arg):
                            token_id = resolve_arg_to_token(arg_elem)
                            label_seq.append(token_id)

                            if tmp_idx < len(arg) - 1:
                                label_seq.append(ProgUtils.ARG_SEP_TOKEN)
                    else:
                        token_id = resolve_arg_to_token(arg)
                        label_seq.append(token_id)

                if arg_idx < len(args) - 1:
                    label_seq.append(ProgUtils.ARG_SEP_TOKEN)

        label_seq.append(ProgUtils.EOS_TOKEN)

        return label_seq

    @staticmethod
    def split_instr_comment(total_sequence):
        '''
        Split a sequence into instructions and comments parts.
        
        @param total_sequence: string in one of three formats:
            1. "instructions" - no separator, comments will be empty string
            2. "instructions//comments" - separator with comments
            3. "instructions//" - separator with empty comments
            
        @return: tuple of (instructions, comments) where comments can be empty string
        '''
        if '//' not in total_sequence:
            # Format 1: "instructions" - no separator
            return total_sequence, ""
        
        # Split on the first occurrence of '//'
        parts = total_sequence.split('//', 1)
        instructions = parts[0]
        comments = parts[1] if len(parts) > 1 else ""
        
        return instructions, comments


    @staticmethod
    def convert_token_sub_seq_to_llm(step_token_seq, primitives):
        '''
        This function converts an instruction step in a program in token sequence format into an intermediate representation.
        '''
        primitive = step_token_seq[1] - ProgUtils.NUM_SPECIAL_TOKENS
        prim_code = prim_text = primitives.inverse_lookup(primitive)

        if prim_text in primitives.text_to_code:
            prim_code = primitives.text_to_code[prim_text]

        arg_tokens = []
        llm_txt = f"{prim_code}"
        
        for arg_idx in range(3, len(step_token_seq)):
            arg = step_token_seq[arg_idx]

            if arg == ProgUtils.ARG_SEP_TOKEN or arg == ProgUtils.EOS_TOKEN:
                if len(arg_tokens) == 1:
                    arg_id = arg_tokens[0] - ProgUtils.NUM_SPECIAL_TOKENS
                    if arg_id < 10:
                        llm_txt += f' {arg_id}'
                    else:
                        ref_id = arg_id - len(primitives.semantics)
                        llm_txt += f' id{ref_id}'
                    arg_tokens = []
                elif len(arg_tokens) == 2:
                    obj = arg_tokens[0] - ProgUtils.NUM_SPECIAL_TOKENS
                    attr = arg_tokens[1] - ProgUtils.NUM_SPECIAL_TOKENS

                    attr_code = primitives.inverse_lookup(attr)

                    if attr_code in primitives.text_to_code:
                        attr_code = primitives.text_to_code[attr_code]

                    ref_id = obj - len(primitives.semantics)                    
                    llm_txt += f' id{ref_id}{attr_code}'
                    arg_tokens = []
                else:
                    break
            else:
                arg_tokens.append(arg)

        return llm_txt

    @staticmethod
    def convert_llm_instr_to_token_subseq(llm_instr, primitives):
        try:
            txt_tokens = re.split(r'\s+', llm_instr.strip())

            token_subseq = [ProgUtils.SOS_TOKEN]
            token_id = 0
            for tok_idx, tok in enumerate(txt_tokens):
                if tok is None:
                    return None
                
                if tok_idx == 0:
                    token_id = primitives.code_to_token_id(tok)

                    # Validate that the first token is a valid primitive
                    if token_id <= 10 or tok.startswith('.'):
                        return None
                    
                    prim_id = token_id + ProgUtils.NUM_SPECIAL_TOKENS
                    token_subseq.append(prim_id)
                    token_subseq.append(ProgUtils.SOP_TOKEN)
                else:
                    if 'id' in tok:
                        # Handle id or id.attr
                        if '.' in tok:
                            # e.g., id3.w or id3.attr
                            id_part, attr_part = tok.split('.', 1)
                            attr_part = f".{attr_part}"

                            obj_id = int(id_part.replace('id', ''))
                            # Find the token id for the attribute
                            attr_token = primitives.code_to_token_id(attr_part)
                            token_subseq.append(obj_id + len(primitives.semantics) + ProgUtils.NUM_SPECIAL_TOKENS)
                            token_subseq.append(attr_token + ProgUtils.NUM_SPECIAL_TOKENS)
                        else:
                            # Just idN
                            obj_id = int(tok.replace('id', ''))
                            token_subseq.append(obj_id + len(primitives.semantics) + ProgUtils.NUM_SPECIAL_TOKENS)
                    else:
                        # Validate integer token range
                        int_tok = int(tok)
                        if int_tok < 0 or int_tok > 9:
                            return None
                        token_subseq.append(int_tok + ProgUtils.NUM_SPECIAL_TOKENS)
                    
                    if tok_idx < len(txt_tokens) - 1:
                        token_subseq.append(ProgUtils.ARG_SEP_TOKEN)

            token_subseq.append(ProgUtils.EOS_TOKEN)
            return token_subseq
        except (ValueError, KeyError, AttributeError, IndexError):
            # Return None if the LLM instruction doesn't correspond to a valid DSL instruction
            # This can happen when:
            # - code_to_token_id fails (KeyError/AttributeError)
            # - int() conversion fails (ValueError) 
            # - String parsing operations fail (ValueError/IndexError)
            return None

    @staticmethod
    def convert_prog_to_token_seq(program, primitives):
        '''
        This function converts a whole program from hand-written format to token sequence format

        @param program: a list of instructions steps (i.e. a whole program) in hand-written format (i.e. using text strings and tuples)
        @param primitives: the DSL

        @return A list of sequences of tokens (integers) as outputted by iterative decoding phases.
        '''
        label_seq = []
        for token_subseq in program:
            tmp_seq = ProgUtils.convert_token_subseq(token_subseq, primitives)
            label_seq.append(tmp_seq)

        return label_seq

    @staticmethod
    def convert_token_seq_to_token_tuple(step_token_seq, primitives):
        '''
        This function converts an instruction step in a program in token sequence format into an intermediate representation.
        '''
        primitive = None
        args_seq = []

        primitive = step_token_seq[1] - ProgUtils.NUM_SPECIAL_TOKENS

        arg_tokens = []
        for arg_idx in range(3, len(step_token_seq)):
            arg = step_token_seq[arg_idx]

            if arg == ProgUtils.ARG_SEP_TOKEN or arg == ProgUtils.EOS_TOKEN:
                if len(arg_tokens) == 1:
                    args_seq.append(arg_tokens[0] - ProgUtils.NUM_SPECIAL_TOKENS)
                    arg_tokens = []
                elif len(arg_tokens) == 2:
                    obj = arg_tokens[0] - ProgUtils.NUM_SPECIAL_TOKENS
                    attr = arg_tokens[1] - ProgUtils.NUM_SPECIAL_TOKENS
                    args_seq.append((obj, attr))
                    arg_tokens = []
                else:
                    break
            else:
                arg_tokens.append(arg)

        return (primitive, args_seq)

    @staticmethod
    def convert_token_tuple_to_str(token_tuple, primitives):

        N = len(primitives.semantics)
        def arg_lookup(arg_idx):
            if arg_idx >= N:
                ref_id = arg_idx - N
                return 'N+%i' % ref_id
            
            else:
                arg_name = primitives.inverse_lookup(arg_idx)
                return arg_name

        prim_idx = token_tuple[0]

        prim_name = primitives.inverse_lookup(prim_idx)

        arg_list = token_tuple[1]

        arg_strs = []
        for arg in arg_list:
            if isinstance(arg, Tuple):
                str1 = arg_lookup(arg[0])
                str2 = arg_lookup(arg[1])

                arg_strs.append((str1, str2))
            else:
                str_arg = arg_lookup(arg)
                arg_strs.append(str_arg)

        return (prim_name, arg_strs)
