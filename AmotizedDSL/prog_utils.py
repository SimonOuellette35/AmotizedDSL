from typing import List, Tuple
import re
import AmotizedDSL.DSL as DSL
import inspect


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

    @staticmethod
    def validate_instruction(instr, intermediate_state):
        num_vars = len(intermediate_state)

        # If num_vars == 1, remove all instructions 51 (those starting with [0, 51, 1]) from valid_instructions
        # This is the delete instruction
        if num_vars == 1:
            if len(instr) >= 3 and instr[0] == 0 and instr[1] == 51 and instr[2] == 1:
                return False, "cannot have a del instruction when there is only 1 state variable."

        # Make it impossible for reference IDs to refer to values > num_vars, so eliminate some possibilities here as well
        # Remove from valid_instructions all instruction sequences that contain 61+num_vars or higher
        threshold_val = 61 + num_vars
        if any(isinstance(token, int) and token >= threshold_val for token in instr):
            return False, f"impossible for reference IDs to refer to ids > {num_vars}"

        # Check if the instruction sequence has compatible argument types
        arguments = ProgUtils.parse_arguments(instr)

        # Special case: new_grid cannot have 0 as its first or second argument
        primitive_idx -= ProgUtils.NUM_SPECIAL_TOKENS
        prim_name = DSL.inverse_lookup(primitive_idx)

        if prim_name == 'new_grid':
            if not ProgUtils.validate_new_grid_statement(arguments):
                return False, 'new_grid cannot have 0 as its first or second argument.'

        arg_types = ProgUtils.extract_data_types(arguments, intermediate_state)
        
        if not ProgUtils.validate_attr_usage(arguments, intermediate_state):
            return False, "cannot use attributes on non-GridObject references."

        is_valid, error_msg = ProgUtils.check_state_variable_types(arg_types, arguments, instr[1])
        
        return is_valid, error_msg

    @staticmethod
    def validate_attr_usage(arguments, intermediate_state):

        for arg in arguments:
            if len(arg) > 1:
                obj_id = arg[0]
                ref_idx = obj_id - len(DSL.semantics)
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
            - List[GridObject]
            - List[Pixel]
            - List[int]
            - List[bool]
            - int (constant)
        '''
        arg_types = []

        for arg in arguments:
            arg_val = arg[0] - ProgUtils.NUM_SPECIAL_TOKENS
            if arg_val < 10:
                # Constant
                # Add int type to arg_types
                arg_types.append(int)
            else:
                if len(arg) > 1:
                    # Attribute reference, this is always a List[int]
                    arg_types.append(List[int])
                else:
                    # direct object reference, just find the type of the referred variable
                    ref_idx = arg_val - len(DSL.semantics)
                    obj = intermediate_state[ref_idx]

                    # parse and simplify the possible types
                    type_str = str(type(obj))
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
                        else:
                            print(f"==> Error: unknown list element type: {elem_type}")
                    else:
                        print(f"==> Error: unknown data type: {type_str}")

        return arg_types


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

        print("group_n = ", group_n)

        conditions = arg_types[:group_n]
        operations = arg_types[group_n: group_n*2]
        otherwise = arg_types[-1]

        # Check conditions: all need to be 'bool'
        for cond_type in conditions:
            if 'bool' not in str(cond_type):
                return False

        # Check operations: all need to be 'int'
        for op_type in operations:
            if 'int' not in str(op_type):
                return False

        # Check otherwise: needs to be 'int'
        if 'int' not in str(otherwise):
            return False

        return True


    @staticmethod
    def check_state_variable_types(arg_types, arguments, primitive_idx):
        '''
        arg_types are the types of the actual arguments passed to the function. They can be:
        - List[GridObject]
        - List[Pixel]
        - List[Int] or constant (which is also int)
        - List[bool]
        
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
            print(f"ERROR: {len(arg_types)} arguments given, but the primitive has {nargs} arguments!")
            return False

        annotations = prim_func.__annotations__
        if len(annotations) == 0:
            # it's a lambda expression, in which case we return True automatically
            return True
        
        param_names = list(inspect.signature(prim_func).parameters.keys())

        for arg_idx, arg_type in enumerate(arg_types):
            print(f"========================================== Validating argument {arg_idx} ==========================================")
            # Attempt to extract type annotation of the argument for prim_func
            arg_name = param_names[arg_idx]

            arg_type_hint = annotations[arg_name]
            print(f"Function {prim_name} argument {arg_idx} has type hint: {arg_type_hint}")
            
            arg_val = arguments[arg_idx]
            print("arg_type = ", arg_type)
            if 'DSL.GridObject' in f'{arg_type_hint}':
                if 'DSL.GridObject' in f'{arg_type}':
                    print("OK, type matches.")
                else:
                    print(f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})")
                    return False
            elif '~COLOR' in f'{arg_type_hint}' or 'int' in f'{arg_type_hint}' or '~DIM' in f'{arg_type_hint}':
                if 'int' in f'{arg_type}':
                    print("OK, type matches.")
                else:
                    print(f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})")
                    return False
            elif 'bool' in f'{arg_type_hint}':
                if 'bool' in f'{arg_type}':
                    print("OK, type matches.")
                else:
                    print(f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})")
                    return False

            elif 'Pixel' in f'{arg_type_hint}':
                if 'Pixel' in f'{arg_type}':
                    print("OK, type matches.")
                else:
                    print(f"ERROR: type mismatch on argument {arg_idx} (arg type: {arg_type}, arg val: {arg_val})")
                    return False

        return True

    @staticmethod
    def filter_invalid(instr, instr_list, error_msg):
        if "cannot have a del instruction when there is only 1 state variable" in error_msg:
            # Remove from instr_list all lists whose second element is 51.
            prim_idx = DSL.prim_indices["del"] + ProgUtils.NUM_SPECIAL_TOKENS
            instr_list = [item for item in instr_list if not item[1] == prim_idx]
            return instr_list
        elif "impossible for reference IDs to refer to ids > " in error_msg:
            # delete all occurences of N + num_vars or higher
            # Remove from instr_list all lists whose sequence contains any integer >= max_id
            match = re.search(r'ids > (\d+)', error_msg)
            if match:
                num_args = int(match.group(1))
            else:
                num_args = 8    # arbitrary, but should not happen anyway

            max_id = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + num_args
            instr_list = [item for item in instr_list if all((isinstance(token, int) and token < max_id) or not isinstance(token, int) for token in item)]
            return instr_list
        elif "new_grid cannot have 0 as its first or second argument." in error_msg:
            # delete all occurrences of 0 in first or second argument of new_grid
            prim_idx = DSL.prim_indices["new_grid"] + ProgUtils.NUM_SPECIAL_TOKENS
            instr_list = [
                item for item in instr_list 
                if not (
                    item[1] == prim_idx and 
                    (
                        ProgUtils.parse_arguments(item)[0][0] == 4 or
                        ProgUtils.parse_arguments(item)[1][0] == 4
                    )
                )
            ]
            return instr_list
        elif "ERROR: type mismatch on argument" in error_msg:
            match = re.search(r'argument\s+(\d+)', error_msg)
            if match:
                arg_idx = int(match.group(1))
            else:
                arg_idx = 0

            # arg_val is a list of integers, e.g., [61, 54], [10], etc.
            arg_val_match = re.search(r'arg val:\s*(\[[^\]]*\])', error_msg)
            if arg_val_match:
                arg_val_str = arg_val_match.group(1)
                try:
                    arg_val = list(map(int, re.findall(r'-?\d+', arg_val_str)))
                except Exception:
                    arg_val = [0]
            else:
                arg_val = [0]

            # Remove all occurrences of this primitive where:
            print("arg_val = ", arg_val)
            # if arg is a constant, any constant at arg_idx
            if arg_val[0] < 13:
                print("is constant")
                instr_list = [
                    item for item in instr_list
                    if not (
                        len(ProgUtils.parse_arguments(item)) > arg_idx and 
                        isinstance(ProgUtils.parse_arguments(item)[arg_idx][0], int) and
                        ProgUtils.parse_arguments(item)[arg_idx][0] < 13 and
                        item[1] == instr[1]
                    )
                ]
                return instr_list
            elif len(arg_val) == 1 and arg_val[0] >= (len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS):
                # if arg is a reference, that specific reference ID at that arg_idx
                instr_list = [
                    item for item in instr_list
                    if not (
                        len(ProgUtils.parse_arguments(item)) > arg_idx and
                        item[1] == instr[1] and
                        ProgUtils.parse_arguments(item)[arg_idx][0] == arg_val[0]
                    )
                ]
                return instr_list
            elif len(arg_val) == 2:
                # if arg is an attribute, any attribute at that arg_idx
                instr_list = [
                    item for item in instr_list
                    if not (
                        item[1] == instr[1] and
                        len(ProgUtils.parse_arguments(item)) > arg_idx and
                        len(ProgUtils.parse_arguments(item)[arg_idx]) == 2
                    )
                ]
                return instr_list
        return instr_list


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
        
        @return A sequence of tokens (integers) as directly outputted by the decoder in one iteration of program generation.
        '''
        prim_name = instr_step[0]
        args = instr_step[1]

        label_seq = [ProgUtils.SOS_TOKEN]

        token_id = ProgUtils.resolve_token_str_to_token(prim_name, primitives)
        label_seq.append(token_id)
        label_seq.append(ProgUtils.SOP_TOKEN)

        if args is not None:
            for arg_idx, arg in enumerate(args):
                # handle object-attribute pairs
                if isinstance(arg, Tuple):
                    tok_obj_id = ProgUtils.resolve_token_str_to_token(arg[0], primitives)
                    tok_attr_id = ProgUtils.resolve_token_str_to_token(arg[1], primitives)
                    label_seq.append(tok_obj_id)
                    label_seq.append(tok_attr_id)
                else:
                    if isinstance(arg, List):
                        # Here we can assume this is a switch statement, in which lists of conditions are possible.
                        for tmp_idx, arg_elem in enumerate(arg):
                            token_id = ProgUtils.resolve_token_str_to_token(arg_elem, primitives)
                            label_seq.append(token_id)

                            if tmp_idx < len(arg) - 1:
                                label_seq.append(ProgUtils.ARG_SEP_TOKEN)
                    else:
                        token_id = ProgUtils.resolve_token_str_to_token(arg, primitives)
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
