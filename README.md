# AmotizedDSL

## Introduction and motivation

This is a DSL I am currently building for my neural program synthesis algorithm for ARC-AGI. It was designed to be easier to use when two conditions are present:

First, when the DSL is to be used in a neural program synthesis context where auto-regressive generation of tokens is part of the program generation. In this context, it is useful for the DSL (and corresponding program syntax) to be easily representable in a flat sequence, even though technically a program is more a tree or graph. My AmotizedDSL solves this by having exactly 1 output at each instruction step, and by allowing to refer to any of the previously generated outputs at any step. This "reference ID" concept means we can structure the program as a flat list without redundancies, yet still access the output of any of the previous program steps. If it is not clear what kind of redundancies and inefficiencies can arise from using a more traditional DSL and program syntax, see discussion in my paper ["Towards Efficient Neurally-Guided Program Induction for ARC-AGI"](https://arxiv.org/abs/2411.17708).

Second, when the neural program synthesis approach needs ongoing access to the current state of the program as it is being constructed and applied on the input grids. This is a crucial part of my new approach, where a program builds the target grid in a kind of "recursive assembly" procedure. At each step, it creates new variables (not necessarily Grid outputs, it can be integers, booleans, lists of booleans, etc.) that get added to the assembly pool, and later steps must figure out, conditionally on the current program state (this "assembly pool"), how to arrive at the target grid. This is inspired by what I called "Learning the Transform Space" in the above paper.

In summary, the way this DSL works and how I structure my program synthesis output is that a program is a list of "instruction steps". What I call an "instruction step" is a relatively small token sequence (usually < 20 tokens) that contains 1 primitive function token, and the parameters to this function. A parameter can be a constant or a reference ID to a previously generated variable (or the original input grids).

Each instruction step outputs exactly 1 variable (though it can be a list) that gets added to the program state, so that it can be referred to by subsequent instruction steps.

It helps to understand the larger context of how my synthesis algorithm works, to understand the design choices for this DSL:

My program inference/search process is an iterative loop (maybe recursive is a better word, since each time I get access to the output of the previous step) of:

1 - token sequence prediction of an instruction step, conditional on the program state so far

2 - executing that instruction step to generate the output variable

3 - adding the tokenized, then encoded variable to my encoder memory

4 - repeating until an "end of program" sequence is generated, or when the search algorithm decides it's completed.

## DSL.GridObject and DSL.Pixel classes

The GridObject class in DSL.py represents an ARC-AGI grid or sub-grid. Aside from containing the cells (list of Pixel instances), it also has a height attribute, a width attribute, a ul_x (upper left x coordinate), and a ul_y (upper left y coordinate) attribute. While these upper left corner coordinates are (0, 0) for a full ARC-AGI grid, their value is can be non-zero in the case of sub-grids. For example, sub-grids can be a way to represent objects in a grid.

The actual grid content can be accessed in two forms: _.pixels_ and _.cells_. _.pixels_ is a flat list of (x, y, color) triples. _.cells_ is a 2-D numpy array of colors. Each color is an integer from 0 to 9 inclusively.

To instantiate a grid, the only mandatory argument is pixels, which is a list of Pixel instances. You can also provide the 2nd and 3rd arguments, which are optional: ul_x and ul_y. This is when you are instantiating a sub-grid.

The Pixel class is a simple data structure representing a cell or pixel in a grid. I has 3 attributes: .x, .y, and .c. The latter is the color integer value (between 0 and 9 inclusively).

## Building and executing a program

### Special tokens

This program syntax uses the following special tokens:
 
> SOS_TOKEN = 0           # Start of sentence

> SOP_TOKEN = 1           # Start of parameters

> ARG_SEP_TOKEN = 2       # Argument separator

> EOS_TOKEN = 3           # End of sentence

### Reference IDs & Object attributes

Referring to previously generated state variables is done by using a token value equal to N+idx, where N is the number of primitives + the number of special tokens, and idx is the 0-indexed number of the state variable. The output of the first instruction step is N+1, the output of the second instruction step is N+2, and so forth. N+0 is the input grid itself. This token value, or "reference ID", can be used anywhere there is a primitive argument to be set.

Note that reference ID are essentially objects, in an object-oriented programming sense. They can either be used as is, when the argument type is of the object being referred to (e.g., a Grid instance), but it is also possible to access some of their attributes. The attributes are at the end of the DSL list:

> Object attributes

> '.x': lambda pixel: get_x(pixel),

> '.y': lambda pixel: get_y(pixel),

> '.c': lambda pixel: get_color(pixel),

> '.max_x': max_x,

> '.max_y': max_y,

> '.width': get_width,

> '.height': get_height,

> '.ul_x': get_ul_x,

> '.ul_y': get_ul_y

In order to use an attribute, you simply use the corresponding token value of the attribute right after the reference ID.

For example, [39, 36] will be N+0.height if 39 is the N+0 reference ID and 36 is the token ID corresponding to '.height'.

This means that an instruction step that consists of cropping some grid B's (reference ID N+1=40) upper left region based on the dimensions of some grid A (reference ID N+0=39) would look like this:

[<SOS_token>, 'crop', <SOP_TOKEN>, 40, <ARG_SEP_TOKEN>, 4, <ARG_SEP_TOKEN>, 4, <ARG_SEP_TOKEN>, 39, 35, <ARG_SEP_TOKEN>, 39, 36, <EOS_TOKEN>]

Because the arguments of the crop primitive are (grid to crop, from_x, from_y, to_x, to_y)

Note: token ID 4 corresponds to the constant 0. Tokens 35 and 36 are assumed to map to '.width' and '.height' respectively.

### Structure of a program

A program is a list of instruction steps, which themselves are a list a tokens. So a program is a list of lists of integer values representing tokens, each mapping to DSL elements or special tokens.

The last instruction step of a program can signify the end of the program by being a list of <EOS_TOKEN> tokens.

Each instruction step outputs exactly 1 state variable.

Each instruction step can access any of the state variables generated by the previous steps. The very first instruction step of a program can only access N+0, the input grid set.

_prog_utils.py_ is a utility file to convert programs between different representations:

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

## Memory management

Because in theory the program state keeps growing as the program grows, there is a special 'del' primitive that does garbage collection of specified program state components ("variables") that are no longer needed by the rest of the program.

This process also serves an additional purpose: it removes noise from the program state and forces the neural network to focus on state variables that matter at each stage. We found that this was crucial to help with generalization. You can think of it as a more restrictive attention system, similar to the limited scope of conscious working memory. By training the model to be extremely sparse with its memory, it learns to focus on what matters for generalization. This idea has been explored in the machine learning literature, for example see ["The Consciousness Prior"](https://arxiv.org/abs/1709.08568).

This is why a typical program contains a lot of del instructions, which are of format: [<SOS_TOKEN>, 'del', <SOP_TOKEN>, N+idx, <EOS_TOKEN>] where N is the total number of primitives in the DSL, and idx is the variable index to delete. (See reference IDs in section "Building and executing a program")

### Execution of a program

To execute a program, use the _program_interpreter.py_ file. Here are the most commonly used methods in this file:
- execute(token_seq_list, state, primitives): executes a whole program. _token_seq_list_ is a list of lists of integers, representing the program in "token sequence" format. _state_ is a list of (non-tokenized) Python variables representing the initial state of the program. In ARC-AGI this means the input grids. So, for a whole program, _state_ should be a list of 1 element: a _DSL.Grid_ instance for an input grid. Note that currently, programs execute only on one example at a time (though I maybe generalize this to the whole demonstration going forward). _primitives_ is simply the whole DSL package, allow to choose between different DSLs if needed.
- execute_instruction_step: executes only an instruction step sequence. _intermediate_state_ is the current list of all states, including the original input grid and all outputs of previous instruction steps.

Looking at the whole program execution code can be informative in understanding how this DSL is intended to be used:

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

Each step is executed, if the special 'del' step was executed, we remove the specified element from the current state "memory". Otherwise, we add the output of the current step to this same state "memory", and move on to the next step.

### Example

See _example.py_ for executable code with plenty of examples. This section is only a high-level overview about writing and executing a program.

**Note:** example.py expects [ARC_gym V2](https://github.com/SimonOuellette35/ARC_gym/tree/refactor/ARC_gymV2) to be installed. Also, you will need [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) cloned into your current folder, since it samples grids and sub-grids from training examples.

An example program that shifts all pixels to the right (without wrapping, and preserving the original grid width) in a grid, in "hand-written representation":

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

You can read it as follows:
1. we gather the right-shifted (+1 operation) x indices of the grid.
2. we apply to the grid the original pixel colors (N+0.c) at the right-shifted indices, causing a kind of right-shifted copy paste.
3. we garbage collect the right-shifted x indices since they were only useful for the second instruction.
4. we fill with zeros (black color) the left-most column, to have a true right-shifted "cut and paste" rather than a "copy paste".
5. garbage collection
6. finally we crop the resulting grid back to the original grid size, effectively getting rid of the new rightmost column that was created.
7. more garbage collection over the last 2 instructions.

The code to convert this to executable (token sequence) format, and to use the program interpreter to actually execute it on some input grid (a DSL.Grid instance):

    import AmotizedDSL.program_interpreter as pi
    from AmotizedDSL.prog_utils import ProgUtils
    import AmotizedDSL.DSL as primitives

    token_seq_list = ProgUtils.convert_prog_to_token_seq(program, primitives)
    output_grid = pi.execute(token_seq_list, input_grid, primitives)

### Switch statement and automatic list handling
Some primitives, such as _switch_ and addition/subtraction/division/multiplication, automatically handle list arguments. That is, you can pass a constant or a list as arguments, and it will behaves as you might intuitively expect.

For example, you can do addition(4, 5) which returns 9. But you can also do addition(\[1, 4, 5, 6\], 2) which returns \[3, 6, 7, 8\]. You can even do addition(\[1, 2, 3\], \[4, 5, 6\]), which returns \[5,7,9\].

It is worth going into detail as to how the switch statement works, because it is a bit complex, yet quite commonly useful and powerful. The switch statement has three arguments: _conditions_, _operations_, _otherwise_. All of the following forms are valid:

**switch(N+1, 6, 7)** the simplest form. Assuming N+1 refers to a single Boolean value, this statement corresponds to: _if (N+1) then return 6, else return 7_. In all cases, the _conditions_ argument of switch is expected to refer to a Boolean, a list of Booleans, or even a list of lists of Booleans (for multiple if/elif/elif/.../else statements).

**switch(N+1, 6, 7)** same as above, but here N+1 refers to a list of Booleans, for example: \[True, False, True\]. This loops through the elements, and where True it returns 6, where False it returns 7. So it would return \[6, 7, 6\].

**switch(N+3, N+1, 8)** here N+3 refers to a list of Booleans, for example \[False, True\]. N+1 refers to a list of integers, for example \[5, 9]. The logic here is that elements of the _conditions_ whose value is True will contain the corresponding elements (by index) of the N+1 _operations_ argument, otherwise they will contain 8. So this would return: \[8, 9\].

**switch(N+3, N+1, N+2)** here N+3 refers to a list of Booleans, for example \[False, True\]. N+1 refers to a list of integers, for example \[5, 9]. N+2 refers to, for example, \[6, 0]. The logic here is that elements of the _conditions_ whose value is True will contain the corresponding elements (by index) of the N+1 _operations_ argument, otherwise they will contain the corresponding element of the _otherwise_ argument. So this would return: \[6, 9\].

**switch(\[N+1, N+2\], \[0, 1\], 2)** this is an if/elif/else statement. N+1 and N+2 must be lists of the same number of elements. It iterates through these, and where N+1 is True, will return 0, where N+2 is True (and N+1 isn't), will return 1. If both are False, it will return the _otherwise_ value of 2.

Example:

    conditions = [ [True, False, False], [False, False, True] ]
    operations = [0, 1]
    otherwise = 2

    Will return: [0, 2, 1]
    
**switch(\[N+1, N+2\], \[N+3, N+4\], N+5)** the most complex form of switch: if/elif/else statement where all arguments are lists. As above, we check the value of each element of _conditions_ in order from left to right, looking for the first True value. That condition index determines which value of _operations_ is returned, as above. But, because here we have lists as individual _operations_, we also must lookup the element by index based on the element index of the condition that was True. And as usual, if none of the _conditions_ are True for a given element index, we return the corresponding element from _otherwise_.

Example:

    conditions = [ [True, False, False], [False, False, True] ]
    operations = [ [2, 3, 1], [6, 9, 7] ]
    otherwise = [5, 4, 5]
    
    Will return: [2, 4, 7]
