# Unit tests for DSL primitives
import AmotizedDSL.DSL as DSL


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
