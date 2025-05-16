# AmotizedDSL

## Introduction and motivation

This is a DSL I am currently building for my neural program synthesis algorithm for ARC-AGI.

The advantages of this DSL and program synthesis syntax are:
1. the primitives are quite low-level and general (e.g. even the basic arithmetic operations are available)
2. each instruction step can refer to the output of previous variables by index, so at each step, all the previously generated variables are available.
3. this allows defining a program as a flat sequence of steps without redundancies.

If it is not clear what kind of redundancies and inefficiencies can arise from using a more traditional DSL and program syntax, see discussion in my paper ["Towards Efficient Neurally-Guided Program Induction for ARC-AGI"](https://arxiv.org/abs/2411.17708).

In summary, the way this DSL works and how I structure my program synthesis output is that a program is a list of "instruction steps". What I call an "instruction step" is a relatively small token sequence (usually < 20 tokens) that contains 1 primitive function token, and the parameters to this function. A parameter can be a constant or a reference ID to a previously generated variable (or the original input grids).

Each instruction step outputs exactly 1 variable (though it can be a list) that gets added to the program state, so that it can be referred to by subsequent instruction steps.

It helps to understand the larger context of how my synthesis algorithm works, to understand the design choices for this DSL:

My program inference/search process is an iterative loop (maybe recursive is a better word, since each time I get access to the output of the previous step) of:

1 - token sequence prediction of an instruction step, conditional on the program state so far

2 - executing that instruction step to generate the output variable

3 - adding the tokenized, then encoded variable to my encoder memory

4 - repeating until an "end of program" sequence is generated, or when the search algorithm decides it's completed.

## Building and executing a program

### Special tokens

This program syntax uses the following special tokens:
 
> SOS_TOKEN = 0           # Start of sentence

> SOP_TOKEN = 1           # Start of parameters

> ARG_SEP_TOKEN = 2       # Argument separator

> EOS_TOKEN = 3           # End of sentence

### Reference IDs & Object attributes

Referring to previously generated state variables is done by using a token value equal to N+idx, where N is the number of primitives + the number of special tokens, and idx is the 0-indexed number of the state variable. The output of the first instruction step is N+0, the output of the second instruction step is N+1, and so forth. This token value, or "reference ID", can be used anywhere there is a primitive argument to be set.

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

For example, [39, 36] will be N+0.height() if 39 is the N+0 reference ID and 36 is the token ID corresponding to '.height'.

This means that an instruction step that consists of cropping some grid B's (reference ID N+1=40) upper left region based on the dimensions of some grid A (reference ID N+0=39) would look like this:

[<SOS_token>, 'crop', <SOP_TOKEN>, 40, <ARG_SEP_TOKEN>, 4, <ARG_SEP_TOKEN>, 4, <ARG_SEP_TOKEN>, 39, 35, <ARG_SEP_TOKEN>, 39, 36, <EOS_TOKEN>]

Note: token ID 4 corresponds to the constant 0. Tokens 35 and 36 are assumed to map to '.width' and '.height' respectively.

### Structure of a program

**TODO**

### Execution of a program

**TODO**

### Example

**TODO**

## Memory management

Because in theory the program state keeps growing as the program grows, there is a special 'del' primitive that does garbage collection of program state components ("variables") that are no longer needed by the rest of the program.

This process also serves an additional purpose: it removes noise from the program state and forces the neural network to focus on state variables that matter at each stage. We found that this was crucial to help with generalization. You can think of it as a more restrictive attention system, similar to the limited scope of conscious working memory. By training the model to be extremely sparse with its memory, it learns to focus on what matters for generalization.

This is why a typical program contains a lot of del instructions, which are of format: [<SOS_TOKEN>, 'del', <SOP_TOKEN>, N+idx, <EOS_TOKEN>] where N is the total number of primitives in the DSL, and idx is the variable index to delete. (See reference IDs in section "Building and executing a program")

### Example

**TODO**

## Code organization

**TODO**
