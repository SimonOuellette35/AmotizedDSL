import AmotizedDSL.program_interpreter as pi

def test_expand_subroutine():

    prog1 = [
        'rot90(N+0)'
    ]

    prog2 = [
        'rot90(N+0)',
        'concat_h(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    prog3 = [
        'concat_h(N+0, N+0)',
        'concat_v(N+1, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    prog4 = [
        'flip_h(N+0)',
        'flip_v(N+0)',
        'del(N+0)',
        'concat_h(N+0, N+1)',
        'concat_h(N+1, N+0)',
        'del(N+0)',
        'del(N+0)',
        'concat_v(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]    

    prog1_expanded = pi.expand_subroutines(prog1)
    prog2_expanded = pi.expand_subroutines(prog2)
    prog3_expanded = pi.expand_subroutines(prog3)
    prog4_expanded = pi.expand_subroutines(prog4)

    expected_prog1 = [
        'set_y(N+0, N+0.x)',
        'sub(N+0.max_y, N+0.y)',
        'del(N+0)',
        'set_x(N+0, N+1)',
        'del(N+0)',
        'del(N+0)'
    ]

    expected_prog2 = [
        # rot90(N+0)
        'set_y(N+0, N+0.x)',
        'sub(N+0.max_y, N+0.y)',
        'del(N+0)',
        'set_x(N+0, N+1)',
        'del(N+0)',
        'del(N+0)',
        # concat_h(N+0, N+1)
        'add(N+1.width, N+1.x)',
        'set_pixels(N+0, N+2, N+0.y, N+1.c)',
        'del(N+0)',
        'del(N+0)',
        'del(N+0)',

        # last 2 dels
        'del(N+0)',
        'del(N+0)'
    ]

    expected_prog3 = [
        # concat_h(N+0, N+0)
        'add(N+0.width, N+0.x)',
        'set_pixels(N+0, N+2, N+0.y, N+0.c)',
        'del(N+0)',
        'del(N+0)',
        'del(N+0)',

        # concat_v(N+1, N+1)
        'add(N+1.height, N+1.y)',
        'set_pixels(N+1, N+1.x, N+2, N+1.c)',
        'del(N+0)',
        'del(N+0)',
        'del(N+0)',

        # 2 last dels
        'del(N+0)',
        'del(N+0)'
    ]

    expected_prog4 = [
        # flip_h(N+0)
        'sub(N+0.max_x, N+0.x),'
        'set_pixels(N+0, N+1, N+0.y, N+0.c)',
        'del(N+0)',
        'del(N+0)',

        # flip_v(N+0)
        'sub(N+0.max_y, N+0.y)',
        'set_pixels(N+0, N+0.x, N+1, N+0.c)',
        'del(N+0)',
        'del(N+0)',

        # del(N+0)
        'del(N+0)',

        # concat_h(N+0, N+1)
        'add(N+1.width, N+1.x)',
        'set_pixels(N+0, N+2, N+0.y, N+1.c)',
        'del(N+0)', 
        'del(N+0)',
        'del(N+0)',

        # concat_h(N+1, N+0)
        'add(param2.width, N+0.x)',
        'set_pixels(N+1, N+2, N+1.y, N+0.c)',
        'del(N+0)', 
        'del(N+0)',
        'del(N+0)',

        # 2 del stements
        'del(N+0)',
        'del(N+0)',
        
        # concat_v(N+0, N+1)
        'add(N+1.height, N+1.y)',
        'set_pixels(N+0, N+0.x, N+2, N+1.c)',
        'del(N+0)',
        'del(N+0)',
        'del(N+0)',

        # 2 last dels
        'del(N+0)',
        'del(N+0)'
    ]            

    print(f"prog1_expanded: {prog1_expanded}")
    print(f"prog2_expanded: {prog2_expanded}")
    print(f"prog3_expanded: {prog3_expanded}")
    print(f"prog4_expanded: {prog4_expanded}")
    
    assert prog1_expanded == expected_prog1, "prog1 does not match expected_prog1"
    assert prog2_expanded == expected_prog2, "prog2 does not match expected_prog2"
    assert prog3_expanded == expected_prog3, "prog3 does not match expected_prog3"
    assert prog4_expanded == expected_prog4, "prog4 does not match expected_prog4"