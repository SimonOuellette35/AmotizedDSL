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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_BOOL, \
        f"Expected TYPE_BOOL ({ProgUtils.TYPE_BOOL}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_INT, \
        f"Expected TYPE_INT ({ProgUtils.TYPE_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_PIXEL]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_GRIDOBJECT]

    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)

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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)

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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_BOOL]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_INT, \
        f"Expected TYPE_LIST_INT ({ProgUtils.TYPE_LIST_INT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_LIST_PIXEL, ProgUtils.TYPE_LIST_PIXEL]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_GRIDOBJECT, ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT, ProgUtils.TYPE_LIST_INT]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
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
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_LIST_GRIDOBJECT, \
        f"Expected TYPE_LIST_GRIDOBJECT ({ProgUtils.TYPE_LIST_GRIDOBJECT}), got {result_type}"

    # State variable types: empty list since new_grid doesn't use state variables
    state_var_types = [ProgUtils.TYPE_GRIDOBJECT]
    
    # Call infer_result_type
    result_type = ProgUtils.dynamic_infer_result_type(instruction_seq, state_var_types)
    
    # Verify the result type
    assert result_type == ProgUtils.TYPE_GRIDOBJECT, \
        f"Expected TYPE_GRIDOBJECT ({ProgUtils.TYPE_GRIDOBJECT}), got {result_type}"

def test_convert_user_format_to_tuple_format():

    # This program is: Object Completion1
    prog = [
        'get_objects(N+0)',
        'get_bg(N+0)',
        'del(N+0)',
        'index(N+1.c, 0)',
        'equal(N+0.c, N+2)',
        'del(N+2)',
        'switch(N+2, "param1", N+0.c)',
        'del(N+2)',
        'set_color(N+0, N+2)',
        'del(N+0)',
        'del(N+1)',
        'rebuild_grid(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    tuple_fmt_prog = ProgUtils.convert_user_format_to_tuple_format(prog, len(DSL.semantics))

    expected_tuples = [
        ('get_objects', [57]),
        ('get_bg', [57]),
        ('del', [57]),
        ('index', [(58, '.c'), 0]),
        ('equal', [(57, '.c'), 59]),
        ('del', [59]),
        ('switch', [59, 'param1', (57, '.c')]),
        ('del', [59]),
        ('set_color', [57, 59]),
        ('del', [57]),
        ('del', [58]),
        ('rebuild_grid', [57, 58]),
        ('del', [57]),
        ('del', [57])
    ]

    assert tuple_fmt_prog == expected_tuples, f"Expected {expected_tuples}, but got {tuple_fmt_prog}"

def test_convert_user_instruction_to_token_seq():

    # This program is: Object Completion1
    prog = [
        'get_objects(N+0)',
        'get_bg(N+0)',
        'del(N+0)',
        'index(N+1.c, 0)',
        'equal(N+0.c, N+2)',
        'del(N+2)',
        'switch(N+2, "param1", N+0.c)',
        'del(N+2)',
        'set_color(N+0, N+2)',
        'del(N+0)',
        'del(N+1)',
        'rebuild_grid(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    token_fmt_prog = ProgUtils.convert_user_format_to_token_seq(prog)

    expected_tokens = [
        [0,15,1,61,3],
        [0,16,1,61,3],
        [0,51,1,61,3],
        [0,22,1,62,54,2,4,3],
        [0,18,1,61,54,2,63,3],
        [0,51,1,63,3],
        [0,21,1,63,2,"param1",2,61,54,3],
        [0,51,1,63,3],
        [0,41,1,61,2,63,3],
        [0,51,1,61,3],
        [0,51,1,62,3],
        [0,48,1,61,2,62,3],
        [0,51,1,61,3],
        [0,51,1,61,3]
    ]

    assert token_fmt_prog == expected_tokens, f"Expected {expected_tokens}, but got {token_fmt_prog}"

def test_convert_user_format_to_token_seq_switch_statement():

    prog1 = [
        'add(N+0.x, 1)', 
        'set_pixels(N+0, N+1, N+0.y, N+0.c)',
        'del(N+1)',
        'del(N+0)',
        'set_pixels(N+0, 0, N+0.y, 0)',
        'del(N+0)',
        'crop(N+0, 0, 0, N+0.max_x, N+0.height)',
        'del(N+0)',
        'equal(N+0.c, param1)',
        'equal(N+0.c, param2)',
        'switch(N+1, N+2, param2, param1, N+0.c)',
        'del(N+1)',
        'del(N+1)',
        'set_pixels(N+0, N+0.x, N+0.y, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    token_fmt_prog1 = ProgUtils.convert_user_format_to_token_seq(prog1)

    expected_tokens1 = [
        [0, 24, 1, 61, 52, 2, 5, 3],
        [0, 38, 1, 61, 2, 62, 2, 61, 53, 2, 61, 54, 3],
        [0, 51, 1, 62, 3],
        [0, 51, 1, 61, 3],
        [0, 38, 1, 61, 2, 4, 2, 61, 53, 2, 4, 3],
        [0, 51, 1, 61, 3],
        [0, 36, 1, 61, 2, 4, 2, 4, 2, 61, 55, 2, 61, 58, 3],
        [0, 51, 1, 61, 3],
        [0, 18, 1, 61, 54, 2, 'param1', 3],
        [0, 18, 1, 61, 54, 2, 'param2', 3],
        [0, 21, 1, 62, 2, 63, 2, 'param2', 2, 'param1', 2, 61, 54, 3],
        [0, 51, 1, 62, 3],
        [0, 51, 1, 62, 3],
        [0, 38, 1, 61, 2, 61, 52, 2, 61, 53, 2, 62, 3],
        [0, 51, 1, 61, 3],
        [0, 51, 1, 61, 3]
    ]

    assert token_fmt_prog1 == expected_tokens1, f"Expected {expected_tokens1}, but got {token_fmt_prog1}"

    prog2 = [
        'div(N+0.width, 2)',
        'colorOf(N+0, N+1, N+1)',
        'add(N+1, 1)',
        'del(N+1)',
        'colorOf(N+0, N+2, N+2)',
        'add(N+2, 1)',
        'del(N+2)',
        'colorOf(N+0, N+3, N+3)',
        'del(N+3)',
        'equal(N+0.c, N+1)',
        'equal(N+0.c, N+2)',
        'switch(N+4, N+5, N+2, N+3, N+1)',
        'del(N+1)',
        'del(N+1)',
        'del(N+1)',
        'del(N+1)',
        'del(N+1)',
        'set_color(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    token_fmt_prog2 = ProgUtils.convert_user_format_to_token_seq(prog2)

    expected_tokens2 = [
      [0, 26, 1, 61, 57, 2, 6, 3],
      [0, 37, 1, 61, 2, 62, 2, 62, 3],
      [0, 24, 1, 62, 2, 5, 3],
      [0, 51, 1, 62, 3],
      [0, 37, 1, 61, 2, 63, 2, 63, 3],
      [0, 24, 1, 63, 2, 5, 3],
      [0, 51, 1, 63, 3],
      [0, 37, 1, 61, 2, 64, 2, 64, 3],
      [0, 51, 1, 64, 3],
      [0, 18, 1, 61, 54, 2, 62, 3],
      [0, 18, 1, 61, 54, 2, 63, 3],
      [0, 21, 1, 65, 2, 66, 2, 63, 2, 64, 2, 62, 3],
      [0, 51, 1, 62, 3],
      [0, 51, 1, 62, 3],
      [0, 51, 1, 62, 3],
      [0, 51, 1, 62, 3],
      [0, 51, 1, 62, 3],
      [0, 41, 1, 61, 2, 62, 3],
      [0, 51, 1, 61, 3],
      [0, 51, 1, 61, 3]
    ]

    assert token_fmt_prog2 == expected_tokens2, f"Expected {expected_tokens2}, but got {token_fmt_prog2}"
