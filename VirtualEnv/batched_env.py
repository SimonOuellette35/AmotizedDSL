import ARC_gym.utils.tokenization as tok
import AmotizedDSL.DSL as DSL
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import copy
import numpy as np


class BatchedAmotizedDSLEnv:

    def __init__(self, input_batch, output_batch, obj_masks_batch, target_comments):
        # input_batch, output_batch are:
        # [batch_size, K, flattened tokenized grid (MAX 931)]
        self.input_batch = input_batch
        self.input_batch_DSL = []
        self.obj_masks_batch = obj_masks_batch
        self.output_batch = output_batch
        self.target_comments = target_comments

        self.batch_size = len(input_batch)
        self.k = len(input_batch[0])

        self.batched_prog_state = None
        self.batched_targets = None
        self.batched_comments = None

    def init(self):
        batch_init_state = []
        targets = []

        current_comments = []
        for batch_idx in range(self.batch_size):
            tmp_states = []        
            tmp_targets = []

            batch_comments = []
            batch_comments.append("Grid")  # the input grid, which is always there
            #batch_comments.append(target_comments[batch_idx][0])

            current_comments.append(batch_comments)

            batch_target_DSL = []
            for k in range(self.k):

                detok_outp_grid = tok.detokenize_grid_unpadded(self.output_batch[batch_idx][k])
                tmp_output_grid = DSL.GridObject.from_grid(detok_outp_grid)

                tmp_targets.append(tmp_output_grid)

                # here, add the original grid
                detok_inp_grid = tok.detokenize_grid_unpadded(self.input_batch[batch_idx][k])
                tmp_input_grid = DSL.GridObject.from_grid(detok_inp_grid)

                # Add it as a list of grids, because the state is always a list of variables, even if there
                # is only one because it's the initial state.
                tmp_states.append([tmp_input_grid])
                batch_target_DSL.append(copy.deepcopy(tmp_input_grid))

            self.input_batch_DSL.append(batch_target_DSL)
            targets.append(tmp_targets)
            batch_init_state.append(tmp_states)

        self.batched_prog_state = batch_init_state
        self.batched_targets = targets    
        self.batched_comments = current_comments
        return batch_init_state, targets

    def is_neural_primitive(self, instr_step):
        '''
        instr_step is an instruction token sequence, directly in the format in the training.json file.

        This function returns True if the primitive used in this instruction step is get_objects or get_bg. (the neural primitives)
        '''
        primitive = instr_step[0]
        if instr_step[0] == 0:
            primitive = instr_step[1]

        primitive = primitive - ProgUtils.NUM_SPECIAL_TOKENS
        prim_name = DSL.inverse_lookup(primitive)
        if prim_name in ['get_objects', 'get_bg']:
            return True, prim_name
        else:
            return False, prim_name

    def act(self, batch_action_sequences, batch_comments):

        # Execute the instruction sequence to get the next state
        tmp_batch_output = []
        for batch_idx in range(len(self.batched_prog_state)):
            batch_k_output = []
            neural_flag, prim_name = self.is_neural_primitive(batch_action_sequences[batch_idx])
            if neural_flag:
                # special case: for get_objects and get_bg, must use the object mask ground truths instead.
                for k in range(self.k):
                    input_grid = self.input_batch_DSL[batch_idx][k]

                    if prim_name == 'get_objects':
                        obj_grids = DSL.get_objects(input_grid, self.obj_masks_batch[batch_idx][k])
                        batch_k_output.append(obj_grids)
                    elif prim_name == 'get_bg':
                        bg_obj = DSL.get_bg(input_grid, self.obj_masks_batch[batch_idx][k])
                        batch_k_output.append(bg_obj)

                tmp_batch_output.append(batch_k_output)                    
            else:
                instr_step = batch_action_sequences[batch_idx]
                intermediate_state = self.batched_prog_state[batch_idx]
                tmp_output = []
                if instr_step[0] == ProgUtils.EOS_TOKEN:
                    for k in range(len(intermediate_state)):
                        tmp_output.append(None)
                else:
                    tmp_output = pi.execute_instruction_step(instr_step, intermediate_state, DSL)
                    
                tmp_batch_output.append(tmp_output)

        for batch_idx in range(len(tmp_batch_output)):
            if not isinstance(tmp_batch_output[batch_idx][0], pi.DeleteAction):
                if batch_comments is not None:
                    #print(f"Adding comment: {target_comments[batch_idx][prog_step_idx]}")
                    self.batched_comments[batch_idx].append(batch_comments[batch_idx])
                else:
                    self.batched_comments[batch_idx].append('END')
                    
        for batch_idx in range(len(tmp_batch_output)):
            if isinstance(tmp_batch_output[batch_idx][0], pi.DeleteAction):
                kept_states = []
                kept_comments = []
                for k in range(len(self.batched_prog_state[batch_idx])):
                    kept_states.append([])
                    for i, state in enumerate(self.batched_prog_state[batch_idx][k]):
                        if i != tmp_batch_output[batch_idx][k].state_idx:
                            kept_states[k].append(state)

                for i in range(len(self.batched_comments[batch_idx])):
                    if i != tmp_batch_output[batch_idx][k].state_idx:
                        kept_comments.append(self.batched_comments[batch_idx][i])

                self.batched_prog_state[batch_idx] = kept_states
                self.batched_comments[batch_idx] = kept_comments

                #print(f"A delete occurred for state_idx {tmp_batch_output[batch_idx][k].state_idx}. New current_comments: {current_comments}")
            elif tmp_batch_output[batch_idx][0] is not None:
                for k in range(len(self.batched_prog_state[batch_idx])):
                    self.batched_prog_state[batch_idx][k].append(tmp_batch_output[batch_idx][k])

        # TODO: calculate reward. Currently unused, though.
        reward_batch = np.zeros([self.batch_size])
        return self.batched_prog_state, self.batched_comments #reward_batch