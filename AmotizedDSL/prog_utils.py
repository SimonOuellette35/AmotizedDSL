from typing import List, Tuple
import re
import AmotizedDSL.DSL as DSL


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
    def extract_arguments(instr):
        # TODO: parse the instruction sequence and get the argument list for input to extract_data_types
        

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
                    else:
                        print(f"==> Error: unknown data type: {type_str}")

        return arg_types


    @staticmethod
    def check_state_variable_types(arg_types, primitive_idx):
        '''
        arg_types are the types of the actual arguments passed to the function. They can be:
        - List[GridObject]
        - List[Pixels]
        - List[Int]
        - List[bool]
        - constant

        and the primitive_idx is the index into the DSL for the primitive function being called.
        
        Returns True is the argument types are all valid, False if there is a mismatch.
        '''
        
        primitive_idx -= ProgUtils.NUM_SPECIAL_TOKENS

        nargs = DSL.arg_counts[primitive_idx]
        if len(arg_types) != nargs:
            print(f"ERROR: {len(arg_types)} arguments given, but the primitive has {nargs} arguments!")
            return False

        prim_name = DSL.inverse_lookup(primitive_idx)
        prim_func = DSL.semantics[prim_name]
        for arg_idx, arg_type in enumerate(arg_types):
            print(f"========================================== Validating argument {arg_idx} ==========================================")
            # Attempt to extract type annotation of the argument for prim_func
            try:
                annotations = prim_func.__annotations__
                # To get the argument name for this index, use inspect:
                import inspect
                param_names = list(inspect.signature(prim_func).parameters.keys())
                arg_name = param_names[arg_idx]
                arg_type_hint = annotations[arg_name]
                print(f"Function {prim_name} argument {arg_idx} has type hint: {arg_type_hint}")
                
                print("arg_type = ", arg_type)
                if 'DSL.GridObject' in f'{arg_type_hint}':
                    if 'DSL.GridObject' in f'{arg_type}':
                        print("OK, type matches.")
                    else:
                        print("ERROR: type mismatch!")
                        return False
                elif '~COLOR' or '~DIM' in f'{arg_type_hint}':
                    if 'int' in f'{arg_type}':
                        print("OK, type matches.")
                    else:
                        print("ERROR: type mismatch!")
                        return False

            except AttributeError:
                # if lambda, any argument type is valid
                pass

        return True

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
