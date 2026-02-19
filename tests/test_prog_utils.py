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

    ref_id = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS

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

    ref_id = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
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
    b = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS

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


    a = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1
    b = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS

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

    data_list = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS

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

    values = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    data_list = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1

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

    val = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS

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

    a = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    b = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1

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

    cond1 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    cond2 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1
    op1 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 2
    op2 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 3
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

    cond1 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    cond2 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1
    cond3 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 2
    op1 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 3
    op2 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 4
    op3 = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 5
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

    data_list = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    flags = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1

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

    a = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    b = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1

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

    grid = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    x = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 1
    y = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 2
    c = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS + 3

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

    grid = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
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

    ref_id = len(DSL.semantics)
    expected_tuples = [
        ('get_objects', [ref_id]),
        ('get_bg', [ref_id]),
        ('del', [ref_id]),
        ('index', [(ref_id+1, '.c'), 0]),
        ('equal', [(ref_id, '.c'), ref_id+2]),
        ('del', [ref_id+2]),
        ('switch', [ref_id+2, 'param1', (ref_id, '.c')]),
        ('del', [ref_id+2]),
        ('set_color', [ref_id, ref_id+2]),
        ('del', [ref_id]),
        ('del', [ref_id+1]),
        ('rebuild_grid', [ref_id, ref_id+1]),
        ('del', [ref_id]),
        ('del', [ref_id])
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

    def lookup(name):
        return DSL.prim_indices[name] + ProgUtils.NUM_SPECIAL_TOKENS

    ref_id = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    print("ref_id = ", ref_id)
    expected_tokens = [
        [0,lookup('get_objects'),1,ref_id,3],
        [0,lookup('get_bg'),1,ref_id,3],
        [0,lookup('del'),1,ref_id,3],
        [0,lookup('index'),1,ref_id+1,lookup('.c'),2,4,3],
        [0,lookup('equal'),1,ref_id,lookup('.c'),2,ref_id+2,3],
        [0,lookup('del'),1,ref_id+2,3],
        [0,lookup('switch'),1,ref_id+2,2,"param1",2,ref_id,lookup('.c'),3],
        [0,lookup('del'),1,ref_id+2,3],
        [0,lookup('set_color'),1,ref_id,2,ref_id+2,3],
        [0,lookup('del'),1,ref_id,3],
        [0,lookup('del'),1,ref_id+1,3],
        [0,lookup('rebuild_grid'),1,ref_id,2,ref_id+1,3],
        [0,lookup('del'),1,ref_id,3],
        [0,lookup('del'),1,ref_id,3]
    ]

    print(f"token_fmt_prog = {token_fmt_prog}")
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

    def lookup(name):
        return DSL.prim_indices[name] + ProgUtils.NUM_SPECIAL_TOKENS

    ref_id = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    expected_tokens1 = [
        [0, lookup('add'), 1, ref_id, lookup('.x'), 2, 5, 3],
        [0, lookup('set_pixels'), 1, ref_id, 2, ref_id+1, 2, ref_id, lookup('.y'), 2, ref_id, lookup('.c'), 3],
        [0, lookup('del'), 1, ref_id+1, 3],
        [0, lookup('del'), 1, ref_id, 3],
        [0, lookup('set_pixels'), 1, ref_id, 2, 4, 2, ref_id, lookup('.y'), 2, 4, 3],
        [0, lookup('del'), 1, ref_id, 3],
        [0, lookup('crop'), 1, ref_id, 2, 4, 2, 4, 2, ref_id, lookup('.max_x'), 2, ref_id, lookup('.height'), 3],
        [0, lookup('del'), 1, ref_id, 3],
        [0, lookup('equal'), 1, ref_id, lookup('.c'), 2, 'param1', 3],
        [0, lookup('equal'), 1, ref_id, lookup('.c'), 2, 'param2', 3],
        [0, lookup('switch'), 1, ref_id+1, 2, ref_id+2, 2, 'param2', 2, 'param1', 2, ref_id, lookup('.c'), 3],
        [0, lookup('del'), 1, ref_id+1, 3],
        [0, lookup('del'), 1, ref_id+1, 3],
        [0, lookup('set_pixels'), 1, ref_id, 2, ref_id, lookup('.x'), 2, ref_id, lookup('.y'), 2, ref_id+1, 3],
        [0, lookup('del'), 1, ref_id, 3],
        [0, lookup('del'), 1, ref_id, 3]
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

    ref_id = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
    expected_tokens2 = [
      [0, lookup('div'), 1, ref_id, lookup('.width'), 2, 6, 3],
      [0, lookup('colorOf'), 1, ref_id, 2, ref_id+1, 2, ref_id+1, 3],
      [0, lookup('add'), 1, ref_id+1, 2, 5, 3],
      [0, lookup('del'), 1, ref_id+1, 3],
      [0, lookup('colorOf'), 1, ref_id, 2, ref_id+2, 2, ref_id+2, 3],
      [0, lookup('add'), 1, ref_id+2, 2, 5, 3],
      [0, lookup('del'), 1, ref_id+2, 3],
      [0, lookup('colorOf'), 1, ref_id, 2, ref_id+3, 2, ref_id+3, 3],
      [0, lookup('del'), 1, ref_id+3, 3],
      [0, lookup('equal'), 1, ref_id, lookup('.c'), 2, ref_id+1, 3],
      [0, lookup('equal'), 1, ref_id, lookup('.c'), 2, ref_id+2, 3],
      [0, lookup('switch'), 1, ref_id+4, 2, ref_id+5, 2, ref_id+2, 2, ref_id+3, 2, ref_id+1, 3],
      [0, lookup('del'), 1, ref_id+1, 3],
      [0, lookup('del'), 1, ref_id+1, 3],
      [0, lookup('del'), 1, ref_id+1, 3],
      [0, lookup('del'), 1, ref_id+1, 3],
      [0, lookup('del'), 1, ref_id+1, 3],
      [0, lookup('set_color'), 1, ref_id, 2, ref_id+1, 3],
      [0, lookup('del'), 1, ref_id, 3],
      [0, lookup('del'), 1, ref_id, 3]
    ]

    assert token_fmt_prog2 == expected_tokens2, f"Expected {expected_tokens2}, but got {token_fmt_prog2}"
