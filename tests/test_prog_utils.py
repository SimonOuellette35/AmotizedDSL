from AmotizedDSL.prog_utils import ProgUtils
import AmotizedDSL.DSL as DSL


def test_infer_type_new_grid():
    primitive_name = 'new_grid'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS
            
    width_const = 3 + ProgUtils.NUM_SPECIAL_TOKENS
    height_const = 3 + ProgUtils.NUM_SPECIAL_TOKENS
    color_const = 0 + ProgUtils.NUM_SPECIAL_TOKENS

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        width_const,
        ProgUtils.ARG_SEP_TOKEN,
        height_const,
        ProgUtils.ARG_SEP_TOKEN,
        color_const,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type is TYPE_GRIDOBJECT (since new_grid returns a GridObject)
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"

def test_infer_type_unique():
    primitive_name = 'unique'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    ref_id = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        ref_id,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_get_index():
    primitive_name = 'index'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    ref_id = 61
    idx = 5 + ProgUtils.NUM_SPECIAL_TOKENS

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        ref_id,
        ProgUtils.ARG_SEP_TOKEN,
        idx,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_BOOL, \
        f"Expected TYPE_BOOL ({ProgUtils.TYPE_BOOL}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"

def test_infer_type_less_than():
    primitive_name = 'less_than'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    a = 5
    b = 6

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_BOOL, \
        f"Expected TYPE_BOOL ({ProgUtils.TYPE_BOOL}), got {result_type}"

    a = 5
    b = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_BOOL, \
        f"Expected TYPE_LIST_BOOL ({ProgUtils.TYPE_LIST_BOOL}), got {result_type}"


    a = 62
    b = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_BOOL, \
        f"Expected TYPE_LIST_BOOL ({ProgUtils.TYPE_LIST_BOOL}), got {result_type}"


def test_infer_type_count_items():
    primitive_name = 'count_items'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    data_list = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        data_list,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_PIXEL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"


def test_infer_type_count_values():
    primitive_name = 'count_values'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    values = 61
    data_list = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        values,
        ProgUtils.ARG_SEP_TOKEN,
        data_list,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_GRIDOBJECT]

    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)

    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"


def test_infer_type_sin_half_pi():
    primitive_name = 'sin_half_pi'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    val = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        val,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    val = 61

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        val,
        ProgUtils.EOS_TOKEN
    ]

    state_var_types = [ProgUtils.TYPE_LIST_INT]

    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)

    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_addition():
    primitive_name = 'add'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    a = 5
    b = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN

    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = []
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    a = 61
    b = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN

    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_switch():
    primitive_name = 'switch'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    cond1 = 61
    cond2 = 62
    op1 = 63
    op2 = 64
    otherwise = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        cond1,
        ProgUtils.ARG_SEP_TOKEN,
        cond2,
        ProgUtils.ARG_SEP_TOKEN,
        op1,
        ProgUtils.ARG_SEP_TOKEN,
        op2,
        ProgUtils.ARG_SEP_TOKEN,
        otherwise,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    cond1 = 61
    cond2 = 62
    cond3 = 63
    op1 = 64
    op2 = 65
    op3 = 66
    otherwise = 5

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        cond1,
        ProgUtils.ARG_SEP_TOKEN,
        cond2,
        ProgUtils.ARG_SEP_TOKEN,
        cond3,
        ProgUtils.ARG_SEP_TOKEN,
        op1,
        ProgUtils.ARG_SEP_TOKEN,
        op2,
        ProgUtils.ARG_SEP_TOKEN,
        op3,
        ProgUtils.ARG_SEP_TOKEN,
        otherwise,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_LIST_BOOL, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

def test_infer_type_keep():
    primitive_name = 'keep'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    data_list = 61
    flags = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        data_list,
        ProgUtils.ARG_SEP_TOKEN,
        flags,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT, ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"


def test_infer_type_set_difference():
    primitive_name = 'set_difference'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    a = 61
    b = 62

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        a,
        ProgUtils.ARG_SEP_TOKEN,
        b,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_PIXEL, ProgUtils.TYPE_LIST_PIXEL]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_PIXEL, \
        f"Expected TYPE_LIST_PIXEL ({ProgUtils.TYPE_LIST_PIXEL}), got {result_type}"


def test_infer_type_set_pixels():
    primitive_name = 'set_pixels'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    grid = 61
    x = 62
    y = 63
    c = 64

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        grid,
        ProgUtils.ARG_SEP_TOKEN,
        x,
        ProgUtils.ARG_SEP_TOKEN,
        y,
        ProgUtils.ARG_SEP_TOKEN,
        c,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT, ProgUtils.TYPE_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_GRIDOBJECT, ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"


def test_infer_type_crop():
    primitive_name = 'crop'
    prim_idx = DSL.prim_indices[primitive_name]
    prim_token = prim_idx + ProgUtils.NUM_SPECIAL_TOKENS

    grid = 61
    x1 = 4
    y1 = 4
    x2 = 10
    y2 = 10

    instruction_seq = [
        ProgUtils.SOS_TOKEN,
        prim_token,
        ProgUtils.SOP_TOKEN,
        grid,
        ProgUtils.ARG_SEP_TOKEN,
        x1,
        ProgUtils.ARG_SEP_TOKEN,
        y1,
        ProgUtils.ARG_SEP_TOKEN,
        x2,
        ProgUtils.ARG_SEP_TOKEN,
        y2,
        ProgUtils.EOS_TOKEN
    ]
            
    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"
