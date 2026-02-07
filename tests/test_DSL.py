# Unit tests for DSL primitives
import AmotizedDSL.DSL as DSL
from AmotizedDSL.DSL import Pixel, GridObject


def test_sort_by_case1():
    # Case 1: flat List[T] and List[int] — sort data_list by corresponding sort_list values (order by key)
    data_list = ["b", "a", "c"]
    sort_list = [2, 0, 1]  # "b"->2, "a"->0, "c"->1 => order 0,1,2 => "a","c","b"
    got = DSL.sort_by(data_list, sort_list)
    assert got == ["a", "c", "b"]


def test_sort_by_case2():
    # Case 2: List[List[T]] and List[int] — sort outer list (reorder inner lists) by one int per inner list
    data_list = [[1, 2], [3, 4], [5, 6]]
    sort_list = [30, 10, 20]
    got = DSL.sort_by(data_list, sort_list)
    assert got == [[3, 4], [5, 6], [1, 2]]


def test_sort_by_case3():
    # Case 3: List[List[T]] and List[List[int]] — sort each inner list by its corresponding inner sort list
    data_list = [["b", "a", "c"], ["z", "x", "y"]]
    sort_list = [[2, 0, 1], [1, 2, 0]]  # first: keys 2,0,1 => "a","c","b"; second: keys 1,2,0 => "y","z","x"
    got = DSL.sort_by(data_list, sort_list)
    assert got == [["a", "c", "b"], ["y", "z", "x"]]


# --- new_grid ---
def test_new_grid():
    g = DSL.new_grid(3, 2, 5)
    assert g.width == 3 and g.height == 2
    assert len(g.pixels) == 6
    assert all(p.c == 5 for p in g.pixels)


# --- get_objects ---
def test_get_objects():
    # 2x2 grid; obj_mask: 0=bg, 1=one pixel at (0,0), 2=one pixel at (1,1)
    grid = GridObject.from_grid([[1, 0], [0, 2]])
    obj_mask = [[1, 0], [0, 2]]
    objs = DSL.get_objects(grid, obj_mask)
    assert len(objs) == 2
    assert len(objs[0].pixels) == 1 and objs[0].pixels[0].c == 1
    assert len(objs[1].pixels) == 1 and objs[1].pixels[0].c == 2


# --- get_bg ---
def test_get_bg():
    grid = GridObject.from_grid([[1, 0], [0, 2]])  # bg color 0 at (1,0) and (0,1)
    obj_mask = [[1, 0], [0, 2]]
    bg = DSL.get_bg(grid, obj_mask)
    assert len(bg.pixels) == 4  # full grid: 2 bg + 2 non-bg (painted with most_common)
    assert bg.width == 2 and bg.height == 2


# --- color_set (colorSet) ---
def test_colorSet():
    grid = GridObject.from_grid([[1, 2], [1, 2]])
    got = DSL.colorSet(grid)
    assert got == [1, 2]


# --- get_width / get_height ---
def test_get_width_height():
    g = GridObject.from_grid([[0] * 5 for _ in range(3)])
    assert DSL.get_width(g) == 5 and DSL.get_height(g) == 3


# --- unique ---
def test_unique():
    assert DSL.unique([1, 2, 1, 3]) == [1, 2, 3]
    assert DSL.unique([[1, 1, 2], [2, 2]]) == [[1, 2], [2]]


# --- get_index (index) ---
def test_get_index():
    assert DSL.get_index([10, 20, 30], 1) == 20
    assert DSL.get_index([[1, 2], [3, 4]], 0) == [1, 3]
    assert DSL.get_index([10, 20], 9) == 0  # out of range -> 0


# --- less_than ---
def test_less_than():
    assert DSL.less_than(2, 5) is True
    assert DSL.less_than(5, 2) is False
    assert DSL.less_than([1, 5, 3], [2, 2, 2]) == [True, False, False]


# --- addition (add) ---
def test_addition():
    assert DSL.addition(3, 7) == 10
    assert DSL.addition([1, 2, 3], [0, 1, 2]) == [1, 3, 5]
    assert DSL.addition([1, 2], 10) == [11, 12]


# --- subtraction (sub) ---
def test_subtraction():
    assert DSL.subtraction(10, 3) == 7
    assert DSL.subtraction([5, 8], [1, 2]) == [4, 6]


# --- division (div) ---
def test_division():
    assert DSL.division(10, 3) == 3
    assert DSL.division([10, 9], [3, 2]) == [3, 4]


# --- multiplication (mul) ---
def test_multiplication():
    assert DSL.multiplication(4, 5) == 20
    assert DSL.multiplication([1, 2, 3], 2) == [2, 4, 6]


# --- modulo (mod) ---
def test_modulo():
    assert DSL.modulo(10, 3) == 1
    assert DSL.modulo([10, 11], 3) == [1, 2]


# --- sin_half_pi ---
def test_sin_half_pi():
    # sin(pi/2 * 0)=0, sin(pi/2 * 1)=1
    assert DSL.sin_half_pi(0) == 0 and DSL.sin_half_pi(1) == 1
    assert DSL.sin_half_pi([0, 1]) == [0, 1]


# --- cos_half_pi ---
def test_cos_half_pi():
    assert DSL.cos_half_pi(0) == 1 and DSL.cos_half_pi(1) == 0
    assert DSL.cos_half_pi([0, 1]) == [1, 0]


# --- equal ---
def test_equal():
    assert DSL.equal(3, 3) is True and DSL.equal(3, 4) is False
    assert DSL.equal([1, 2, 2], 2) == [False, True, True]


# --- not_equal ---
def test_not_equal():
    assert DSL.not_equal(3, 4) is True and DSL.not_equal(3, 3) is False
    assert DSL.not_equal([1, 2, 2], 2) == [True, False, False]


# --- logical_or ---
def test_logical_or():
    assert DSL.logical_or(True, False) is True
    assert DSL.logical_or([True, False], [False, False]) == [True, False]


# --- logical_and ---
def test_logical_and():
    assert DSL.logical_and(True, True) is True and DSL.logical_and(True, False) is False
    assert DSL.logical_and([True, False], [True, True]) == [True, False]


# --- logical_xor ---
def test_logical_xor():
    assert DSL.logical_xor(True, True) is False and DSL.logical_xor(True, False) is True
    assert DSL.logical_xor([True, False], [False, False]) == [True, False]


# --- arg_min / arg_max ---
def test_arg_min_arg_max():
    assert DSL.arg_min([10, 20, 30], [3, 1, 2]) == 20  # min val 1 at index 1 -> arg_list[1]=20
    assert DSL.arg_max([10, 20, 30], [1, 3, 2]) == 20  # max val 3 at index 1 -> arg_list[1]=20


# --- count_items ---
def test_count_items():
    assert DSL.count_items([1, 2, 3]) == 3
    assert DSL.count_items([[1, 2], [3]]) == [2, 1]


# --- count_values ---
def test_count_values():
    assert DSL.count_values([1, 2, 3], [1, 1, 2, 1]) == [3, 1, 0]
    grid = GridObject.from_grid([[1, 2], [1, 1]])
    assert DSL.count_values([1, 2], grid) == [3, 1]


# --- switch ---
def test_switch():
    # single condition: conditions must be a list (one element)
    assert DSL.switch([True], 5, 10) == 5
    assert DSL.switch([False], 5, 10) == 10
    # list output: conditions and operations as list-of-list so first branch is list
    assert DSL.switch([[True, False, True]], [[2, 3, 4]], 99) == [2, 99, 4]


# --- colorOf ---
def test_colorOf():
    grid = GridObject.from_grid([[1, 2], [3, 4]])
    assert DSL.colorOf(grid, 0, 0) == 1 and DSL.colorOf(grid, 1, 1) == 4
    assert DSL.colorOf(grid, [0, 1], [0, 1]) == [1, 4]


# --- keep ---
def test_keep():
    assert DSL.keep([1, 2, 3, 4], [True, False, True, False]) == [1, 3]
    assert DSL.keep([[1, 2], [3, 4]], [[True, False], [False, True]]) == [[1], [4]]


# --- exclude ---
def test_exclude():
    assert DSL.exclude([1, 2, 3, 4], [True, False, True, False]) == [2, 4]


# --- set_difference ---
def test_set_difference():
    assert DSL.set_difference([1, 2, 3], [2]) == [1, 3]
    assert DSL.set_difference([[1, 2], [3, 4]], [[2], [4]]) == [[1], [3]]


# --- rebuild_grid ---
def test_rebuild_grid():
    bg = GridObject.from_grid([[0, 0], [0, 0]])
    obj = GridObject([Pixel(0, 0, 7)], 0, 0)
    out = DSL.rebuild_grid(bg, obj)
    assert out.cells[0][0] == 7 and out.cells[1][0] == 0


# --- set_pixels ---
def test_set_pixels():
    grid = GridObject.from_grid([[0, 0], [0, 0]])
    out = DSL.set_pixels(grid, [0, 1], [0, 0], [5, 6])
    assert DSL.colorOf(out, 0, 0) == 5 and DSL.colorOf(out, 1, 0) == 6


# --- set_x / set_y / set_color ---
def test_set_x_set_y_set_color():
    # grid with one pixel at (0,0) color 1
    grid = GridObject.from_grid([[1]])
    out_x = DSL.set_x(grid, [2])
    assert out_x.pixels[0].x == 2 and out_x.pixels[0].y == 0
    out_y = DSL.set_y(grid, [3])
    assert out_y.pixels[0].y == 3
    out_c = DSL.set_color(grid, [9])
    assert out_c.pixels[0].c == 9


# --- crop ---
def test_crop():
    grid = GridObject.from_grid([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    out = DSL.crop(grid, 1, 1, 3, 3)
    assert out.width == 2 and out.height == 2
    assert DSL.colorOf(out, 0, 0) == 4 and DSL.colorOf(out, 1, 1) == 8


# --- max_x / max_y / get_ul_x / get_ul_y ---
def test_max_x_max_y_ul():
    grid = GridObject.from_grid([[1, 2], [3, 4]])
    assert DSL.max_x(grid) == 1 and DSL.max_y(grid) == 1
    assert DSL.get_ul_x(grid) == 0 and DSL.get_ul_y(grid) == 0


# --- neighbours4 / neighbours8 ---
def test_neighbours4_neighbours8():
    # 2x2 grid: all four pixels
    grid = GridObject.from_grid([[1, 2], [3, 4]])
    n4 = DSL.neighbours4(grid)
    assert len(n4) == 4
    # corner (0,0) has 2 neighbours: (1,0) and (0,1)
    assert len(n4[0]) == 2
    n8 = DSL.neighbours8(grid)
    assert len(n8) == 4
    assert len(n8[0]) == 3  # (0,0) has 3 neighbours in 8-connectivity


# --- get_x / get_y / get_color (pixel attributes) ---
def test_get_x_get_y_get_color():
    p = Pixel(2, 3, 4)
    assert DSL.get_x(p) == 2 and DSL.get_y(p) == 3 and DSL.get_color(p) == 4
    pixels = [Pixel(0, 0, 1), Pixel(1, 0, 2)]
    assert DSL.get_x(pixels) == [0, 1] and DSL.get_color(pixels) == [1, 2]

