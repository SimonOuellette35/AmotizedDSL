from AmotizedDSL.prog_utils import ProgUtils


def test_filter_invalid_reference_ids1():
    """Test filtering instructions that have reference IDs > num_vars."""
    instruction = [
        0,
        21,
        1,
        61,
        2,
        62,
        2,
        5,
        2,
        4,
        2,
        0,
        3
    ]

    instr_list = [
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 23, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 24, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 26, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3]
    ]
   
    error_msg = "impossible for reference IDs to refer to ids > 1"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 0

def test_filter_invalid_reference_ids2():
    """Test filtering instructions that have reference IDs > num_vars."""
    instruction = [
        0,
        21,
        1,
        61,
        2,
        62,
        2,
        5,
        2,
        4,
        2,
        0,
        3
    ]

    instr_list = [
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 23, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 24, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 26, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "impossible for reference IDs to refer to ids > 1"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 2

def test_filter_invalid_reference_ids3():
    """Test filtering instructions that have reference IDs > num_vars."""
    instruction = [
        0,
        21,
        1,
        61,
        2,
        62,
        2,
        5,
        2,
        4,
        2,
        0,
        3
    ]

    instr_list = [
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 21, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 23, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 24, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 26, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "impossible for reference IDs to refer to ids > 3"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == len(instr_list)

def test_filter_del():

    """Test filtering del when there is only 1 state variable."""
    instruction = [
        0,
        51,
        1,
        61,
        3
    ]

    instr_list = [
        [0, 51, 1, 61, 3],
        [0, 51, 1, 61, 3],
        [0, 51, 1, 61, 3],
        [0, 23, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 24, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 26, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "cannot have a del instruction when there is only 1 state variable"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 5

def test_filter_new_grid():

    """Test filtering new_grid with 0 value for width or height"""
    instruction = [
        0,
        42,
        1,
        5,
        2,
        4,
        2,
        4,
        3
    ]

    instr_list = [
        [0, 42, 1, 5, 2, 4, 2, 8, 3],
        [0, 42, 1, 4, 2, 5, 2, 9, 3],
        [0, 42, 1, 4, 2, 4, 2, 12, 3],
        [0, 42, 1, 5, 2, 8, 2, 4, 3],
        [0, 42, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 26, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "new_grid cannot have 0 as its first or second argument."
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 5

def test_filter_constant():

    """Test filtering the use of a constant for a non-int argument."""
    instruction = [
        0,
        38,
        1,
        4,
        2,
        62,
        2,
        63,
        2,
        61,
        3
    ]

    instr_list = [
        [0, 38, 1, 9, 2, 62, 2, 63, 2, 6, 3],
        [0, 38, 1, 12, 2, 62, 2, 63, 2, 9, 3],
        [0, 38, 1, 61, 2, 7, 2, 63, 2, 12, 3],
        [0, 38, 1, 61, 2, 62, 2, 4, 2, 4, 3],
        [0, 38, 1, 61, 2, 62, 2, 4, 2, 64, 3],
        [0, 38, 1, 61, 2, 62, 2, 4, 2, 64, 3],
        [0, 42, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 26, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "ERROR: type mismatch on argument 0 (arg type: DSL.GridObject, arg val: [4])"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 8

def test_switch_constant():

    """Test filtering the use of a constant for a non-int argument."""
    instruction = [
        0,
        21,
        1,
        6,
        2,
        62,
        2,
        63,
        3
    ]

    instr_list = [
        [0, 21, 1, 4, 2, 62, 2, 63, 3],
        [0, 21, 1, 7, 2, 62, 2, 63, 3],
        [0, 21, 1, 61, 2, 7, 2, 4, 3],
        [0, 21, 1, 61, 2, 62, 2, 4, 3],
        [0, 38, 1, 61, 2, 62, 2, 4, 2, 64, 3],
        [0, 38, 1, 61, 2, 62, 2, 4, 2, 64, 3],
        [0, 42, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 26, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "ERROR: type mismatch on argument 0 (arg type: bool, arg val: [6])"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 8

def test_filter_ref_id():

    """Test filtering the use of a constant for a non-int argument."""
    instruction = [
        0,
        26,
        1,
        12,
        2,
        61,
        3
    ]

    instr_list = [
        [0, 26, 1, 62, 2, 61, 3],
        [0, 26, 1, 61, 2, 61, 3],
        [0, 26, 1, 13, 2, 61, 3],
        [0, 26, 1, 61, 2, 8, 3],
        [0, 26, 1, 6, 2, 5, 3],
        [0, 38, 1, 61, 2, 62, 2, 4, 2, 64, 3],
        [0, 42, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 28, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "ERROR: type mismatch on argument 1 (arg type: int, arg val: [61])"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 7

def test_filter_attr():

    """Test filtering the use of a constant for a non-int argument."""
    instruction = [
        0,
        48,
        1,
        61,
        58,
        2,
        62,
        3
    ]

    instr_list = [
        [0, 48, 1, 61, 2, 61, 3],
        [0, 48, 1, 62, 2, 61, 56, 3],
        [0, 48, 1, 61, 57, 2, 61, 3],   # will be filtered
        [0, 48, 1, 62, 55, 2, 61, 3],   # will be filtered
        [0, 48, 1, 61, 52, 2, 62, 3],   # will be filtered
        [0, 48, 1, 62, 54, 2, 61, 3],   # will be filtered
        [0, 42, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 25, 1, 61, 2, 62, 2, 5, 2, 4, 2, 0, 3],
        [0, 28, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3],
        [0, 27, 1, 61, 2, 61, 2, 5, 2, 4, 2, 0, 3]
    ]

    error_msg = "ERROR: type mismatch on argument 0 (arg type: GridObject, arg val: [61, 58])"
    
    filtered_list = ProgUtils.filter_invalid(instruction, instr_list, error_msg)
    
    # The function should return a filtered list
    assert len(filtered_list) == 6
