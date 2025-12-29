from typing import List, Tuple
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
    def static_infer_result_type(primitive_name):
        """Get the return type of a primitive by method name.
        
        Args:
            primitive_name: Name of the primitive function
            
        Returns:
            1 if return type contains "GridObject"
            2 if return type contains int
            3 if return type contains bool
            4 if return type contains "Pixel"
            5 otherwise
        """
        if primitive_name not in DSL.semantics:
            return 5
        
        prim_func = DSL.semantics[primitive_name]
        
        # Handle non-callable primitives (e.g., integer constants)
        if not callable(prim_func):
            # Integer constants return int
            if isinstance(prim_func, int):
                return 2
            return 5
        
        # Get return annotation
        try:
            sig = inspect.signature(prim_func)
            return_annotation = sig.return_annotation
        except (ValueError, TypeError):
            # If signature can't be obtained, return default
            return 5
        
        # Convert annotation to string
        if return_annotation == inspect.Signature.empty:
            return 5
        
        return_type_str = str(return_annotation)
        
        # Check in priority order: GridObject, int, bool, Pixel
        if 'GridObject' in return_type_str:
            return 1
        elif 'int' in return_type_str or 'COLOR' in return_type_str or 'DIM' in return_type_str:
            return 2
        elif 'bool' in return_type_str:
            return 3
        elif 'Pixel' in return_type_str:
            return 4
        else:
            return 5

    @staticmethod
    def remove_unused_instructions(crossover_instrs):
        """Remove all instructions whose generated UUID is not used at all in the program.
        
        Args:
            crossover_instrs: List of instruction strings in format 
                '<uuid> = <instruction>(arguments)' or 'del(<uuid>)'
        
        Returns:
            Modified list of instruction strings with unused instructions removed
        """
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        # Find the index of the last non-del instruction
        last_non_del_idx = None
        for i in range(len(crossover_instrs) - 1, -1, -1):
            if not crossover_instrs[i].strip().startswith('del('):
                last_non_del_idx = i
                break
        
        # Collect all UUIDs that are used (appear in arguments or del statements)
        used_uuids = set()
        
        for instr in crossover_instrs:
            instr_stripped = instr.strip()
            if instr_stripped.startswith('del('):
                # For del instructions, the UUID in del() is being used
                uuid_matches = re.findall(uuid_pattern, instr_stripped, re.IGNORECASE)
                used_uuids.update(uuid_matches)
            else:
                # For non-del instructions, extract UUIDs from arguments (the part after '=')
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr_stripped)
                if match:
                    instruction_part = match.group(2).strip()
                    # Extract all UUIDs from arguments
                    uuid_matches = re.findall(uuid_pattern, instruction_part, re.IGNORECASE)
                    used_uuids.update(uuid_matches)
        
        # If there's a last non-del instruction, mark its output UUID as used
        if last_non_del_idx is not None:
            last_instr = crossover_instrs[last_non_del_idx].strip()
            if not last_instr.startswith('del('):
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', last_instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        used_uuids.add(output_uuid_match.group(0))
        
        # Filter out instructions whose output UUID is never used
        result = []
        for i, instr in enumerate(crossover_instrs):
            instr_stripped = instr.strip()
            if instr_stripped.startswith('del('):
                # Keep all del instructions
                result.append(instr)
            else:
                # For non-del instructions, check if output UUID is used
                # Always keep the last non-del instruction
                if i == last_non_del_idx:
                    result.append(instr)
                else:
                    match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr_stripped)
                    if match:
                        output_uuid_str = match.group(1).strip()
                        output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                        if output_uuid_match:
                            output_uuid = output_uuid_match.group(0)
                            # Only keep if the output UUID is used somewhere
                            if output_uuid in used_uuids:
                                result.append(instr)
                        else:
                            # If no valid output UUID, keep the instruction
                            result.append(instr)
                    else:
                        # If format doesn't match, keep the instruction
                        result.append(instr)
        
        return result


    @staticmethod
    def reassign_invalid_uuids(crossover_instrs, range_start):
        """Reassign invalid UUIDs in crossover instructions.
        
        Goes through program instructions starting from range_start. For each instruction,
        if it has arguments that refer to a UUID that does not exist (i.e. there is no 
        previous instruction that generates that UUID), change it to a UUID that actually 
        exists (randomly selected among the previously generated UUIDs).
        
        When replacing a UUID, if the replacement UUID's primitive returns GridObject (type 1),
        randomly either leave the UUID as is, or append one of the possible attributes (like .x, .c, .max_y, etc.).
        
        Args:
            crossover_instrs: List of instruction strings in format 
                '<uuid> = <instruction>(arguments)' or 'del(<uuid>)'
            range_start: Index to start processing from
        
        Returns:
            Modified list of instruction strings with invalid UUIDs replaced
        """
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        # Collect all UUIDs generated by instructions before range_start and map them to their primitive names
        valid_uuids = set()
        uuid_to_primitive = {}  # Maps UUID to the primitive name that generated it
        
        # Add 'input_grid' as a valid absolute reference (always exists)
        valid_uuids.add('input_grid')
        
        for i in range(range_start):
            instr = crossover_instrs[i].strip()
            if not instr.startswith('del('):
                # Extract output UUID (the UUID before '=') and primitive name
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        output_uuid = output_uuid_match.group(0)
                        valid_uuids.add(output_uuid)
                        
                        # Extract primitive name from instruction part (e.g., "get_objects(...)" -> "get_objects")
                        instruction_part = match.group(2).strip()
                        prim_match = re.match(r'^(\w+)\(', instruction_part)
                        if prim_match:
                            uuid_to_primitive[output_uuid] = prim_match.group(1)
        
        # Get all available attributes from DSL (names starting with '.')
        available_attributes = [attr for attr in DSL.prim_indices.keys() if attr.startswith('.')]
        
        # If no valid UUIDs exist, we can't reassign anything
        if not valid_uuids:
            return crossover_instrs
        
        def get_replacement_uuid_with_attr(preserve_attr=None):
            """Get a replacement UUID, potentially with an attribute appended if it's a GridObject.
            
            Args:
                preserve_attr: If provided, randomly decide whether to preserve, change, or remove this attribute
            
            Returns:
                Replacement UUID with optional attribute
            """
            replacement = np.random.choice(list(valid_uuids))
            
            # Check if the replacement UUID's primitive returns GridObject
            is_gridobject = False
            if replacement in uuid_to_primitive:
                prim_name = uuid_to_primitive[replacement]
                return_type = ProgUtils.static_infer_result_type(prim_name)
                is_gridobject = (return_type == 1)
            
            # If there was an original attribute, randomly decide what to do with it
            if preserve_attr:
                if not is_gridobject or not available_attributes:
                    # If replacement is not a GridObject or no attributes available, remove attribute
                    return replacement
                
                # Randomly decide: preserve (1/3), change (1/3), or remove (1/3)
                rand_val = np.random.random()
                if rand_val < 0.33:
                    # Preserve the original attribute
                    return replacement + preserve_attr
                elif rand_val < 0.67:
                    # Change to a random attribute
                    attr = np.random.choice(available_attributes)
                    return replacement + attr
                else:
                    # Remove the attribute
                    return replacement
            
            # No original attribute - randomly decide whether to add one
            if is_gridobject:
                # Randomly decide whether to append an attribute
                if np.random.random() < 0.5 and available_attributes:
                    # Append a random attribute
                    attr = np.random.choice(available_attributes)
                    return replacement + attr
            return replacement
        
        result = list(crossover_instrs)
        
        # Process instructions from range_start onwards
        for i in range(range_start, len(crossover_instrs)):
            instr = result[i].strip()
            
            if instr.startswith('del('):
                # For del instructions, extract UUID from del(<uuid>)
                uuid_matches = list(re.finditer(uuid_pattern, instr, re.IGNORECASE))
                attr_pattern = r'(\.\w+)+'
                instr_copy = instr
                for match in reversed(uuid_matches):
                    uuid_val = match.group(0)
                    match_end = match.end()
                    
                    # Check if there's an attribute after the UUID
                    attr_match = re.match(attr_pattern, instr[match_end:])
                    attr_str = None
                    if attr_match:
                        attr_str = attr_match.group(0)
                        match_end = match_end + len(attr_str)
                    
                    # Check if base UUID (without attribute) is valid
                    base_uuid = uuid_val
                    if base_uuid not in valid_uuids:
                        # Replace with random valid UUID (no attributes for del instructions, ignore attr_str)
                        replacement = np.random.choice(list(valid_uuids))
                        instr_copy = instr_copy[:match.start()] + replacement + instr_copy[match_end:]
                # Note: 'input_grid' won't match UUID pattern, so it won't be reassigned in del() instructions
                result[i] = instr_copy
            else:
                # Non-del instruction: format is '<output_uuid> = <instruction>(arguments)'
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    instruction_part = match.group(2).strip()
                    
                    # Extract output UUID
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    output_uuid = None
                    if output_uuid_match:
                        output_uuid = output_uuid_match.group(0)
                    
                    # Extract primitive name for this instruction
                    prim_match = re.match(r'^(\w+)\(', instruction_part)
                    prim_name = prim_match.group(1) if prim_match else None
                    
                    # Extract all UUIDs from arguments (the instruction part)
                    uuid_matches = list(re.finditer(uuid_pattern, instruction_part, re.IGNORECASE))
                    
                    # Pattern to match attributes (one or more attribute chains like .x, .y.x, etc.)
                    attr_pattern = r'(\.\w+)+'
                    
                    # Replace invalid UUIDs in arguments
                    # Note: 'input_grid' references won't match UUID pattern, so they won't be reassigned
                    # (input_grid is always valid and is included in valid_uuids)
                    instr_copy = instruction_part
                    for match in reversed(uuid_matches):
                        uuid_val = match.group(0)
                        match_end = match.end()
                        
                        # Check if there's an attribute after the UUID
                        attr_match = re.match(attr_pattern, instruction_part[match_end:])
                        attr_str = None
                        if attr_match:
                            attr_str = attr_match.group(0)
                            match_end = match_end + len(attr_str)
                        
                        # Check if base UUID (without attribute) is valid
                        base_uuid = uuid_val
                        if base_uuid not in valid_uuids:
                            # Get replacement UUID, preserving attribute if it exists
                            replacement = get_replacement_uuid_with_attr(preserve_attr=attr_str)
                            instr_copy = instr_copy[:match.start()] + replacement + instr_copy[match_end:]
                    
                    # Reconstruct the instruction
                    if output_uuid:
                        result[i] = f"{output_uuid} = {instr_copy}"
                        # Add output UUID to valid set for subsequent instructions
                        valid_uuids.add(output_uuid)
                        # Store primitive name for this UUID
                        if prim_name:
                            uuid_to_primitive[output_uuid] = prim_name
                    else:
                        result[i] = f"{output_uuid_str} = {instr_copy}"
                else:
                    # Fallback: if format doesn't match, try to replace UUIDs anyway
                    uuid_matches = list(re.finditer(uuid_pattern, instr, re.IGNORECASE))
                    attr_pattern = r'(\.\w+)+'
                    instr_copy = instr
                    for match in reversed(uuid_matches):
                        uuid_val = match.group(0)
                        match_end = match.end()
                        
                        # Check if there's an attribute after the UUID
                        attr_match = re.match(attr_pattern, instr[match_end:])
                        attr_str = None
                        if attr_match:
                            attr_str = attr_match.group(0)
                            match_end = match_end + len(attr_str)
                        
                        # Check if base UUID (without attribute) is valid
                        base_uuid = uuid_val
                        if base_uuid not in valid_uuids:
                            replacement = get_replacement_uuid_with_attr(preserve_attr=attr_str)
                            instr_copy = instr_copy[:match.start()] + replacement + instr_copy[match_end:]
                    # Note: 'input_grid' won't match UUID pattern, so it won't be reassigned
                    result[i] = instr_copy
        
        return result

        
    @staticmethod
    def map_uuids_to_refIDs(uuid_instructions):
        """Transform absolute UUIDs back into relative refIDs (N+X) for each step.
        
        This is the reverse operation of map_refIDs_to_uuids.
        
        Args:
            uuid_instructions: List of instruction strings with UUIDs in format 
                ['<uuid> = <instruction>(arguments)', 'del(<uuid>)', ...]
        
        Returns:
            List of instruction strings with UUIDs replaced by refIDs relative to current stack size
        """
        # UUID pattern: 8-4-4-4-12 hexadecimal digits separated by hyphens
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        # Collect all output UUIDs (the ones on the left side of '=')
        output_uuids = set()
        all_uuids = set()
        
        for instr in uuid_instructions:
            original_instr = instr.strip()
            if not original_instr.startswith('del('):
                # Non-del instruction: format is '<output_uuid> = <instruction>(arguments)'
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', original_instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        output_uuids.add(output_uuid_match.group(0))
            
            # Collect all UUIDs from this instruction
            uuids = re.findall(uuid_pattern, original_instr, re.IGNORECASE)
            all_uuids.update(uuids)
        
        if not all_uuids:
            return uuid_instructions
        
        # The initial UUID is the one that appears in arguments but is never an output
        # Find the first instruction and get the UUID from its arguments
        initial_uuid = None
        if uuid_instructions:
            first_instr = uuid_instructions[0].strip()
            if not first_instr.startswith('del('):
                # Extract UUIDs from arguments (the part after '=')
                match = re.match(r'^[^=]+\s*=\s*(.+)$', first_instr)
                if match:
                    instruction_part = match.group(1).strip()
                    arg_uuids = re.findall(uuid_pattern, instruction_part, re.IGNORECASE)
                    # The initial UUID is the one in arguments that is not an output
                    for uuid_val in arg_uuids:
                        if uuid_val not in output_uuids:
                            initial_uuid = uuid_val
                            break
        
        if initial_uuid is None:
            # Fallback: use first UUID that's not an output
            for uuid_val in all_uuids:
                if uuid_val not in output_uuids:
                    initial_uuid = uuid_val
                    break
        
        if initial_uuid is None:
            # Last resort: use first UUID encountered
            first_instr = uuid_instructions[0] if uuid_instructions else ""
            first_uuids = re.findall(uuid_pattern, first_instr, re.IGNORECASE)
            if first_uuids:
                initial_uuid = first_uuids[0]
        
        # Check if "input_grid" appears in any instruction - if so, it should be the initial object
        has_input_grid = any('input_grid' in instr for instr in uuid_instructions)
        
        # Now simulate the stack state at each instruction
        # Stack tracks which UUID is at each position
        # If input_grid is used, it should be the initial object (like in map_refIDs_to_uuids)
        if has_input_grid:
            stack = ["input_grid"]  # Initial object is "input_grid"
        elif initial_uuid is not None:
            stack = [initial_uuid]  # Initial object
        else:
            return uuid_instructions
        
        transformed = []
        
        for i, instr in enumerate(uuid_instructions):
            original_instr = instr.strip()
            
            # Check if this is a del instruction or an assignment format instruction
            if original_instr.startswith('del('):
                # Delete instruction: format is 'del(<uuid>)'
                instr_copy = original_instr
                
                # Check if "input_grid" is being deleted (before replacement)
                has_input_grid = 'input_grid' in original_instr
                
                # Replace "input_grid" with "N+0" first
                instr_copy = instr_copy.replace('input_grid', 'N+0')
                
                # Extract all UUIDs from this instruction
                uuid_matches = list(re.finditer(uuid_pattern, instr_copy, re.IGNORECASE))
                
                # Replace each UUID with its refID based on current stack state
                # Process in reverse order to maintain string positions
                for match in reversed(uuid_matches):
                    obj_uuid = match.group(0)
                    try:
                        idx = stack.index(obj_uuid)
                        ref_id = f"N+{idx}"
                        instr_copy = instr_copy[:match.start()] + ref_id + instr_copy[match.end():]
                    except ValueError:
                        # UUID not in current stack - this shouldn't happen if the input is valid
                        # But handle gracefully by keeping the UUID
                        pass
                
                transformed.append(instr_copy)
                
                # Update stack: remove the deleted UUID
                if uuid_matches:
                    del_uuid = uuid_matches[0].group(0)  # Get the UUID being deleted
                    if del_uuid in stack:
                        stack.remove(del_uuid)
                elif has_input_grid:
                    # Handle "input_grid" deletion - it was replaced with "N+0" but we need to remove it from stack
                    if 'input_grid' in stack:
                        stack.remove('input_grid')
            else:
                # Non-del instruction: format is '<output_uuid> = <instruction>(arguments)'
                # Extract the output UUID (before the '=') and the instruction part (after the '=')
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', original_instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    instruction_part = match.group(2).strip()
                    
                    # Extract the output UUID
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        output_uuid = output_uuid_match.group(0)
                        
                        # Replace each UUID in arguments with its refID based on current stack state
                        instr_copy = instruction_part
                        
                        # First, handle "input_grid" references (they don't match UUID pattern)
                        if 'input_grid' in instr_copy:
                            if 'input_grid' in stack:
                                idx = stack.index('input_grid')
                                ref_id = f"N+{idx}"
                                # Replace "input_grid" (and any attributes like "input_grid.x")
                                # Use regex to match "input_grid" followed by optional attribute
                                input_grid_pattern = r'input_grid(\.\w+)*'
                                instr_copy = re.sub(input_grid_pattern, lambda m: ref_id + (m.group(1) if m.group(1) else ''), instr_copy)
                        
                        # Then, extract and replace UUIDs
                        uuid_matches = list(re.finditer(uuid_pattern, instr_copy, re.IGNORECASE))
                        for match in reversed(uuid_matches):
                            obj_uuid = match.group(0)
                            try:
                                idx = stack.index(obj_uuid)
                                ref_id = f"N+{idx}"
                                # Check if there's an attribute after the UUID
                                attr_pattern = r'(\.\w+)+'
                                match_end = match.end()
                                attr_match = re.match(attr_pattern, instr_copy[match_end:])
                                attr_str = ''
                                if attr_match:
                                    attr_str = attr_match.group(0)
                                    match_end = match_end + len(attr_str)
                                instr_copy = instr_copy[:match.start()] + ref_id + attr_str + instr_copy[match_end:]
                            except ValueError:
                                # UUID not in current stack - this shouldn't happen if the input is valid
                                # But handle gracefully by keeping the UUID
                                pass
                        
                        transformed.append(instr_copy)
                        
                        # Update stack: add the output UUID
                        stack.append(output_uuid)
                    else:
                        # Fallback: if format doesn't match, try old format
                        instr_copy = original_instr
                        # Replace "input_grid" with "N+0" first
                        instr_copy = instr_copy.replace('input_grid', 'N+0')
                        uuid_matches = list(re.finditer(uuid_pattern, instr_copy, re.IGNORECASE))
                        for match in reversed(uuid_matches):
                            obj_uuid = match.group(0)
                            try:
                                idx = stack.index(obj_uuid)
                                ref_id = f"N+{idx}"
                                instr_copy = instr_copy[:match.start()] + ref_id + instr_copy[match.end():]
                            except ValueError:
                                pass
                        transformed.append(instr_copy)
                else:
                    # Fallback: if format doesn't match, try old format
                    instr_copy = original_instr
                    # Replace "input_grid" with "N+0" first
                    instr_copy = instr_copy.replace('input_grid', 'N+0')
                    uuid_matches = list(re.finditer(uuid_pattern, instr_copy, re.IGNORECASE))
                    for match in reversed(uuid_matches):
                        obj_uuid = match.group(0)
                        try:
                            idx = stack.index(obj_uuid)
                            ref_id = f"N+{idx}"
                            instr_copy = instr_copy[:match.start()] + ref_id + instr_copy[match.end():]
                        except ValueError:
                            pass
                    transformed.append(instr_copy)
        
        return transformed

    @staticmethod
    def map_refIDs_to_uuids(prog_instructions):
        """Transform relative refIDs (N+X) into absolute UUIDs for each unique object.
        
        Args:
            prog_instructions: List of instruction strings like ["get_objects(N+0)", "del(N+0)", ...]
        
        Returns:
            List of instruction strings in format '<uuid> = <instruction>(arguments)' for non-del instructions,
            or 'del(<uuid>)' for del instructions, where refIDs are replaced by UUIDs
        """
        import uuid
        
        # Stack tracks which object UUID each position refers to
        # Stack starts with one object (N+0)
        stack = ["input_grid"]  # Initial object is "input_grid"
        
        # Map from refID (integer) to UUID for current stack state
        # N+0 refers to the first object (stack[0]), N+1 to second object (stack[1]), etc.
        def get_uuid_for_refid(ref_id):
            """Get the UUID for a refID based on current stack state."""
            if ref_id < len(stack):
                return stack[ref_id]
            return None
        
        transformed = []
        
        for instr in prog_instructions:
            original_instr = instr.strip()
            instr = original_instr
            
            # Extract all refIDs from this instruction
            ref_matches = list(re.finditer(r'N\+(\d+)', instr))
            
            # Replace each refID with its UUID
            # Process in reverse order to maintain string positions
            for match in reversed(ref_matches):
                ref_id = int(match.group(1))
                obj_uuid = get_uuid_for_refid(ref_id)
                if obj_uuid:
                    instr = instr[:match.start()] + obj_uuid + instr[match.end():]
            
            # Update stack based on instruction type (check original instruction)
            if original_instr.startswith('del('):
                # Delete instruction: extract refID and remove that object from stack
                del_match = re.search(r'del\(N\+(\d+)\)', original_instr)
                if del_match:
                    del_ref_id = int(del_match.group(1))
                    if del_ref_id < len(stack):
                        stack.pop(del_ref_id)
                # For del instructions, output format is just 'del(<uuid>)'
                transformed.append(instr)
            else:
                # Non-del instruction: creates new object, push to stack
                output_uuid = str(uuid.uuid4())
                stack.append(output_uuid)
                # Format as '<output_uuid> = <instruction>(arguments)'
                transformed.append(f"{output_uuid} = {instr}")
        
        return transformed

    @staticmethod
    def remove_dels(instructions):
        """Remove all del instructions from the instructions program.
        
        Args:
            instructions: List of instruction strings
            
        Returns:
            List of instruction strings with all del instructions removed
        """
        return [instr for instr in instructions if not instr.strip().startswith('del(')]

    @staticmethod
    def auto_add_dels(uuid_instructions):
        '''
        Here, those are instructions where the redIDs are actually UUIDs. They are absolute references.

        This function automatically adds a del statement once an object is no longer referenced for the rest of the
        program.
        '''
        # UUID pattern: 8-4-4-4-12 hexadecimal digits separated by hyphens
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        
        # Track all UUIDs that appear and which ones have been deleted
        all_uuids = set()
        deleted_uuids = set()
        
        # Track the last occurrence index of each UUID
        uuid_last_occurrence = {}
        
        # Track input_grid separately (it's not a UUID)
        input_grid_last_occurrence = None
        input_grid_deleted = False
        
        # First pass: find the last occurrence of each UUID and track deleted UUIDs
        for i, instr in enumerate(uuid_instructions):
            # Track deleted UUIDs
            if instr.strip().startswith('del('):
                deleted_uuids_in_instr = re.findall(uuid_pattern, instr, re.IGNORECASE)
                deleted_uuids.update(deleted_uuids_in_instr)
                # Check if input_grid is being deleted
                if 'input_grid' in instr:
                    input_grid_deleted = True
                continue
            
            # Extract all UUIDs from this instruction
            uuids = re.findall(uuid_pattern, instr, re.IGNORECASE)
            all_uuids.update(uuids)
            for uuid in uuids:
                uuid_last_occurrence[uuid] = i
            
            # Track input_grid occurrences
            if 'input_grid' in instr:
                input_grid_last_occurrence = i
        
        # Extract output UUID from the last instruction (if it exists)
        last_output_uuid = None
        if uuid_instructions:
            last_instr = uuid_instructions[-1].strip()
            if not last_instr.startswith('del('):
                # Extract output UUID (the UUID before '=')
                match = re.match(r'^([^=]+)\s*=\s*(.+)$', last_instr)
                if match:
                    output_uuid_str = match.group(1).strip()
                    output_uuid_match = re.match(uuid_pattern, output_uuid_str, re.IGNORECASE)
                    if output_uuid_match:
                        last_output_uuid = output_uuid_match.group(0)
        
        # Second pass: insert del statements after last occurrences
        result = []
        dels_to_insert = {}  # Maps index -> list of UUIDs to delete at that position
        
        for uuid, last_idx in uuid_last_occurrence.items():
            # Insert del after the last instruction that references this UUID
            # Only if there are more instructions after it
            if last_idx < len(uuid_instructions) - 1:
                # Store at last_idx so we insert after that instruction
                if last_idx not in dels_to_insert:
                    dels_to_insert[last_idx] = []
                dels_to_insert[last_idx].append(uuid)
        
        # Handle input_grid deletion
        if input_grid_last_occurrence is not None and not input_grid_deleted:
            # Insert del(input_grid) after the last instruction that references it
            if input_grid_last_occurrence < len(uuid_instructions) - 1:
                if input_grid_last_occurrence not in dels_to_insert:
                    dels_to_insert[input_grid_last_occurrence] = []
                dels_to_insert[input_grid_last_occurrence].append('input_grid')
        
        # Build result with del statements inserted
        for i, instr in enumerate(uuid_instructions):
            result.append(instr)
            
            # Insert del statements after this instruction if needed
            if i in dels_to_insert:
                for uuid in dels_to_insert[i]:
                    result.append(f"del({uuid})")
                    if uuid != 'input_grid':
                        deleted_uuids.add(uuid)  # Track UUIDs deleted by inserted del statements
        
        # Add del statements at the end for all non-deleted UUIDs except the last output
        remaining_uuids = all_uuids - deleted_uuids
        if last_output_uuid:
            remaining_uuids.discard(last_output_uuid)
        
        for uuid in sorted(remaining_uuids):
            result.append(f"del({uuid})")
        
        return result

    @staticmethod
    def convert_instruction_string_to_token_seq(instr_str):
        # TODO: this is the new streamlined instruction text format. It is possible that some of the older conversion
        # methods are now obsolete because of this. To be reviewed. Maybe I can also implement the ground truth program
        # strings in the LLM format directly and get rid of the other text format. The multitude of different program
        # representations in this codebase is starting to get confusing and I need to look into simplifying/refactoring.
        """
        Convert an instruction string (e.g., "get_objects(N+0)") to token sequence format.
        
        Args:
            instr_str: Instruction string in format "primitive_name(arg1, arg2, ...)"
                      where arguments can be:
                      - "N+X" for reference IDs (X is integer)
                      - Integer constants (0-9)
                      - Attribute references like "N+X.y" or "N+X.x"
        
        Returns:
            List of integers representing the token sequence: [SOS, primitive, SOP, args..., EOS]
        """
        # Parse primitive name and arguments
        match = re.match(r'(\w+)\((.*)\)', instr_str.strip())
        if not match:
            return None
        
        prim_name = match.group(1)
        args_str = match.group(2).strip()
        
        # Get primitive token ID
        if prim_name not in DSL.prim_indices:
            return None
        prim_idx = DSL.prim_indices[prim_name]
        prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS
        
        # Build token sequence
        token_seq = [ProgUtils.SOS_TOKEN, prim_token, ProgUtils.SOP_TOKEN]
        
        # Parse arguments
        if args_str:
            # Split arguments by comma, but be careful with nested structures
            args = []
            current_arg = ""
            paren_depth = 0
            
            for char in args_str:
                if char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char
            
            if current_arg.strip():
                args.append(current_arg.strip())
            
            # Convert each argument to tokens
            for arg_idx, arg in enumerate(args):
                arg = arg.strip()
                if not arg:
                    continue
                
                # Strip quotes if present (e.g., "param1" -> param1)
                if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                    arg = arg[1:-1]
                
                # Check if it's an attribute reference (e.g., "N+0.x")
                if '.' in arg:
                    parts = arg.split('.', 1)
                    obj_ref = parts[0].strip()
                    attr_name = '.' + parts[1].strip()
                    
                    # Parse object reference (N+X)
                    obj_match = re.match(r'N\+(\d+)', obj_ref)
                    if not obj_match:
                        return None
                    ref_id = int(obj_match.group(1))
                    obj_token = len(DSL.semantics) + ref_id + ProgUtils.NUM_SPECIAL_TOKENS
                    
                    # Parse attribute
                    if attr_name not in DSL.prim_indices:
                        return None
                    attr_idx = DSL.prim_indices[attr_name]
                    attr_token = attr_idx + ProgUtils.NUM_SPECIAL_TOKENS
                    
                    token_seq.append(obj_token)
                    token_seq.append(attr_token)
                else:
                    # Check if it's a reference (N+X) or constant
                    if arg.startswith('N+'):
                        ref_match = re.match(r'N\+(\d+)', arg)
                        if not ref_match:
                            return None
                        ref_id = int(ref_match.group(1))
                        ref_token = len(DSL.semantics) + ref_id + ProgUtils.NUM_SPECIAL_TOKENS
                        token_seq.append(ref_token)
                    else:
                        # Check if it's a param placeholder (e.g., "param1", "param2")
                        if re.match(r'^param\d+$', arg):
                            # Keep param placeholder as string in token sequence
                            token_seq.append(arg)
                        else:
                            # Try to parse as integer constant
                            try:
                                const_val = int(arg)
                                if 0 <= const_val <= 9:
                                    const_token = const_val + ProgUtils.NUM_SPECIAL_TOKENS
                                    token_seq.append(const_token)
                                else:
                                    return None
                            except ValueError:
                                return None
                
                # Add argument separator if not last argument
                if arg_idx < len(args) - 1:
                    token_seq.append(ProgUtils.ARG_SEP_TOKEN)
        
        token_seq.append(ProgUtils.EOS_TOKEN)
        return token_seq

    @staticmethod
    def validate_ref_ids(prog_list):
        """
        Validate reference IDs in a program.
        
        Args:
            prog_list: List of instruction strings in format "primitive_name(arg1, arg2, ...)"
                     Each instruction string represents one program step.
                     
                     Expected format examples:
                     - "get_objects(N+0)" - primitive with reference argument
                     - "del(N+0)" - delete instruction with reference
                     - "crop(N+0, N+1)" - primitive with multiple references
                     - "new_grid(5, 3)" - primitive with integer constants
                     - "get_index(N+0, N+1.x)" - primitive with attribute reference
                     
                     Arguments can be:
                     - "N+X" for reference IDs (X is integer, refers to variable at index X in the stack)
                     - Integer constants (0-9) as plain integers
                     - Attribute references like "N+X.y", "N+X.x", or "N+X.c" for accessing object attributes
        
        Returns:
            True if all reference IDs are valid (i.e., all references point to existing variables in the stack),
            False otherwise
        """
        stack_counter = 1
        
        for prog_row in prog_list:
            # Convert instruction string to token sequence
            instr = ProgUtils.convert_instruction_string_to_token_seq(prog_row)
            if instr is None:
                return False
            
            # Extract all reference IDs from this instruction
            arguments = ProgUtils.parse_arguments(instr)

            for arg in arguments:
                if len(arg) == 0:
                    continue
                
                # Skip validation for param placeholders (e.g., "param1", "param2")
                if isinstance(arg[0], str) and re.match(r'^param\d+$', arg[0]):
                    continue
                
                # Check the first token (object reference or constant)
                arg_val = arg[0] - ProgUtils.NUM_SPECIAL_TOKENS
                
                # If it's a reference (not a constant < 10)
                if arg_val >= 10:
                    ref_idx = arg_val - len(DSL.semantics)
                    
                    # Check if ref_idx is in valid range [0, stack_counter-1]
                    if ref_idx < 0 or ref_idx >= stack_counter:
                        return False
            
            # Update stack counter based on instruction type
            # Check if this is a del instruction: SOS (0), primitive (51), SOP (1)
            if len(instr) >= 3 and instr[0] == 0 and instr[1] == 51 and instr[2] == 1:
                stack_counter -= 1
                if stack_counter < 0:
                    return False
            else:
                stack_counter += 1
        
        return True
        
    @staticmethod
    def get_variable_type_code(var):
        """
        Convert a variable from intermediate state to an integer type code.
        
        Args:
            var: A variable from intermediate state (GridObject, int, bool, Pixel, or list)
            
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
