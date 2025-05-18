from typing import List, Tuple


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
    '''

    SOS_TOKEN = 0           # Start of sentence
    SOP_TOKEN = 1           # Start of parameters
    ARG_SEP_TOKEN = 2       # Argument separator
    EOS_TOKEN = 3           # End of sentence

    NUM_SPECIAL_TOKENS = 4

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
                    token_id = ProgUtils.resolve_token_str_to_token(arg, primitives)
                    label_seq.append(token_id)

                if arg_idx < len(args) - 1:
                    label_seq.append(ProgUtils.ARG_SEP_TOKEN)

        label_seq.append(ProgUtils.EOS_TOKEN)

        return label_seq

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
