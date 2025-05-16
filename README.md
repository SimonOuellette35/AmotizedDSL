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

(TODO)
