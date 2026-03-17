## DSL Reference - Primitives and their functionality

### Object / background extraction

**get_objects**: returns a list of `GridObject` instances representing the objects identified in the grid. Can also return a list of sub-objects if the grid parameter was a list of `GridObject` instances instead.
  
  Parameters:
  - grid (a variable reference): `GridObject` instance or `List[GridObject]`. This is the grid to segment or list of grid objects to further segment into their respective objects.

  Returns:
  - a `GridObject` instance or list of `GridObject` instances, depending on the type of the grid parameter.
    
**get_bg**: returns the background part (with foreground objects removed) of a grid.

  Parameters:
  - grid (a variable reference): `GridObject` instance from which to extract the background.

  Returns:
  - a `GridObject` instance.
 
### Basic functional primitives

**identity**: returns its input unchanged.

  Parameters:
  - x: any type.

  Returns:
  - same value as `x`.

**color_set**: returns the sorted set of distinct colors present in a grid or list of grids.

  Parameters:
  - grid: `GridObject` or `List[GridObject]`.

  Returns:
  - `List[int]` or `List[List[int]]` of unique color ids.

**equal**: elementwise equality comparison between scalars, lists, nested lists, or lists of `GridObject`.

  Parameters:
  - a, b: scalars, lists, nested lists, or lists of `GridObject` with compatible shapes.

  Returns:
  - bool, `List[bool]`, or nested lists of bools mirroring the structure of the inputs.

**not_equal**: negated version of `equal`, with the same input conventions.

  Parameters:
  - a, b: same as for `equal`.

  Returns:
  - bool, `List[bool]`, or nested lists of bools indicating inequality.

**less_than**: elementwise `<` comparison supporting scalars, lists, and nested lists.

  Parameters:
  - a, b: int or list / nested list of int, with shapes compatible for elementwise comparison or broadcasting.

  Returns:
  - bool, `List[bool]`, or nested lists of bools.

**switch**: generalized if/elif/else operator with list overloading.

  Parameters:
  - conditions: bool, `List[bool]`, or `List[List[bool]]` (possibly nested for multiple branches).
  - operations: value(s) to use when a condition is true; can be scalar, list, or nested lists aligned with `conditions`.
  - otherwise: value(s) to use when no condition holds; scalar or list / nested lists.

  Returns:
  - a value, list, or nested lists whose shape follows the condition structure (see `DSL.py` for detailed overloading rules).

**index**: returns elements at a given index from a list or list of lists.

  Parameters:
  - list: `List[T]` or `List[List[T]]`.
  - i: integer index. Out‑of‑range indices return 0.

  Returns:
  - element at index `i`, or list of elements if the input is a list of lists.

**unique**: removes duplicates while preserving order, with support for nested lists.

  Parameters:
  - data: `List[int]`, `List[List[int]]`, or `List[List[List[int]]]`.

  Returns:
  - same structure as `data`, but with duplicates removed within each list.

### Arithmetic primitives

**add**: elementwise or broadcasted integer addition.

  Parameters:
  - a, b: int, `List[int]`, or nested lists of int with compatible shapes for elementwise or broadcasted addition.

  Returns:
  - sum with the same list / nesting structure as the inputs.

**sub**: elementwise or broadcasted integer subtraction (`a - b`).

  Parameters:
  - a, b: int, `List[int]`, or nested lists of int with compatible shapes.

  Returns:
  - difference with the same list / nesting structure as the inputs.

**div**: integer floor division (`//`) with broadcasting.

  Parameters:
  - a, b: int, float, `List[int]`, or `List[List[int]]`.

  Returns:
  - integer quotients with structure matching the list inputs.

**mul**: elementwise or broadcasted integer multiplication.

  Parameters:
  - a, b: int, `List[int]`, or `List[List[int]]`.

  Returns:
  - products with structure matching the list inputs.

**mod**: elementwise or broadcasted modulo (`a % b`).

  Parameters:
  - a, b: int, `List[int]`, or `List[List[int]]`.

  Returns:
  - remainders with structure matching the list inputs.

**sin_half_pi**: applies `int(sin(pi/2 * v))` to scalars or lists.

  Parameters:
  - val: int, `List[int]`, or `List[List[int]]`.

  Returns:
  - corresponding integer results with matching structure.

**cos_half_pi**: applies `int(cos(pi/2 * v))` to scalars or lists.

  Parameters:
  - val: int, `List[int]`, or `List[List[int]]`.

  Returns:
  - corresponding integer results with matching structure.

**arg_min**: selects argument(s) associated with minimum values.

  Parameters:
  - arg_list: list of candidate arguments.
  - val_list (optional): list of values of same length; if omitted, `np.argmin(arg_list)` is used.

  Returns:
  - index of minimum (when `val_list` is `None`) or the element of `arg_list` whose value is minimal.

**arg_max**: selects argument(s) associated with maximum values.

  Parameters:
  - arg_list: list of candidate arguments.
  - val_list (optional): list of values of same length; if omitted, `np.argmax(arg_list)` is used.

  Returns:
  - index of maximum (when `val_list` is `None`) or the element of `arg_list` whose value is maximal.

### Logical primitives

**or**: logical OR with support for pairwise and reduce modes.

  Parameters:
  - a: bool or `List[bool]` (or list of lists of bools).
  - b (optional): bool or `List[bool]`. If omitted, performs a reduce over `a`; if provided, performs pairwise OR.

  Returns:
  - bool or list / nested lists of bool, depending on input shapes.

**and**: logical AND with support for pairwise and reduce modes.

  Parameters:
  - a: bool or `List[bool]` (or list of lists of bools).
  - b (optional): bool or `List[bool]`. If omitted, performs a reduce over `a`; if provided, performs pairwise AND.

  Returns:
  - bool or list / nested lists of bool, depending on input shapes.

**xor**: logical XOR with support for pairwise and reduce modes.

  Parameters:
  - a: bool or `List[bool]`.
  - b (optional): bool or `List[bool]`. If omitted, performs a reduce over `a`; if provided, performs pairwise XOR.

  Returns:
  - bool or list of bools, depending on inputs.

### Counting and histogram primitives

**count_items**: returns the length of a list, list of lists, or list of list of lists.

  Parameters:
  - data_list: `List[T]`, `List[List[T]]`, or `List[List[List[T]]]`.

  Returns:
  - int, `List[int]`, or nested `List[List[int]]` of lengths.

**count_values**: histogram‑like count of how often given values occur.

  Parameters:
  - values: `List[int]` of values to count.
  - data_list: either a `GridObject` (counts over pixel colors) or `List[int]`.

  Returns:
  - `List[int]` where each element is the count of the corresponding value in `values`.

### Grid construction and editing

**new_grid**: creates a new rectangular grid filled with a background color.

  Parameters:
  - w: width (int).
  - h: height (int).
  - bg_color: background color (int).

  Returns:
  - `GridObject` of size `h × w` filled with `bg_color`.

**set_pixels**: sets specific coordinates in a grid (or list of grids) to given color values, expanding the grid if needed.

  Parameters:
  - target_grid: `GridObject` or `List[GridObject]` to modify.
  - set_x: x coordinate(s); int, `List[int]`, or list of lists.
  - set_y: y coordinate(s); int, `List[int]`, or list of lists.
  - colors: color(s) to assign; int, `List[int]`, or list of lists.

  Returns:
  - modified `GridObject` or `List[GridObject]`.

**set_x**: rewrites the x‑coordinates of all pixels in a grid or list of grids.

  Parameters:
  - grid: `GridObject` or `List[GridObject]`.
  - x_values: `List[int]` or `List[List[int]]` with one x value per pixel.

  Returns:
  - new `GridObject` or list of `GridObject` with updated x positions.

**set_y**: rewrites the y‑coordinates of all pixels in a grid or list of grids.

  Parameters:
  - grid: `GridObject` or `List[GridObject]`.
  - y_values: `List[int]` or `List[List[int]]` with one y value per pixel.

  Returns:
  - new `GridObject` or list of `GridObject` with updated y positions.

**set_color**: rewrites the color of all pixels in a grid or list of grids.

  Parameters:
  - grid: `GridObject` or `List[GridObject]`.
  - c_values: int, `List[int]`, or `List[List[int]]` with color(s) per pixel.

  Returns:
  - new `GridObject` or list of `GridObject` with updated colors.

**crop**: crops a rectangular region out of a grid (or list of grids).

  Parameters:
  - g: `GridObject` or `List[GridObject]`.
  - x1, y1: inclusive upper‑left corner of crop.
  - x2, y2: exclusive lower‑right corner of crop.

  Returns:
  - `GridObject` or `List[GridObject]` restricted to the cropped region, with updated upper‑left coordinates.

**rebuild_grid**: overlays one or more objects onto a background grid.

  Parameters:
  - bg_grid: background `GridObject`.
  - obj_list: `GridObject` or `List[GridObject]` to paste onto the background at their own coordinates.

  Returns:
  - new `GridObject` with all objects pasted onto `bg_grid`.

### Neighbourhood / adjacency

**neighbours4**: for each pixel, returns its 4‑connected neighbours (up, down, left, right) that lie inside the same object.

  Parameters:
  - grid: `GridObject` or `List[GridObject]`.

  Returns:
  - `List[List[Pixel]]` (or `List[List[List[Pixel]]]` for multiple grids), where each inner list holds the neighbours of a single pixel.

**neighbours8**: for each pixel, returns its 8‑connected neighbours (including diagonals) that lie inside the same object.

  Parameters:
  - grid: `GridObject` or `List[GridObject]`.

  Returns:
  - `List[List[Pixel]]` (or `List[List[List[Pixel]]]`) of neighbours per pixel.

### Sorting and indexing helpers

**sort_by**: sorts data according to associated sort keys.

  Parameters:
  - data_list: `List[T]` or `List[List[T]]`.
  - sort_list: `List[int]` or `List[List[int]]` containing sort keys aligned with `data_list`.

  Returns:
  - `data_list` reordered according to ascending sort keys.

**gather**: gathers elements from nested lists by index.

  Parameters:
  - data_list: `List[List[T]]` or `List[List[List[T]]]`.
  - idx: int, `List[int]`, or `List[List[int]]`, interpreted as indices to pull from each inner list or inner‑inner list.

  Returns:
  - list or nested lists of gathered elements.

### Color / coordinate queries

**colorOf**: returns the color(s) at given coordinates in one grid or a list of grids.

  Parameters:
  - g: `GridObject` or `List[GridObject]`.
  - x: int or `List[int]` of x‑coordinates (per grid if `g` is a list).
  - y: int or `List[int]` of y‑coordinates (per grid if `g` is a list).

  Returns:
  - int or `List[int]` of colors at the queried positions.

### List filtering and set operations

**keep**: filters elements from a list (or list of lists) based on boolean flags.

  Parameters:
  - input_list: `List[T]` or `List[List[T]]`.
  - flags: `List[bool]` or `List[List[bool]]` of the same shape as `input_list`.

  Returns:
  - filtered list with only the elements where the corresponding flag is `True`.

**exclude**: inverse of `keep` for a flat list; removes elements where the flag is `True`.

  Parameters:
  - input_list: `List[T]`.
  - flags: `List[bool]` (same length as `input_list`).

  Returns:
  - list of elements where the corresponding flag is `False`.

**set_difference**: removes from `a` any element that also appears in `b`, supporting nested lists and `Pixel` / `GridObject` elements via `__eq__`.

  Parameters:
  - a: list or nested lists of elements.
  - b: element, list, or nested lists to subtract from `a`.

  Returns:
  - list or nested lists with elements of `b` removed.

### Special primitives

**del**: special “delete from state” primitive. Implemented at a deeper level in the program execution code.

### Attribute access primitives

These are implemented as primitives in the DSL and are used as postfix accessors.

**.x**: returns the x‑coordinate(s) of a pixel or list / nested lists of pixels.

  Parameters:
  - pixel: `Pixel`, `List[Pixel]`, or nested lists of `Pixel`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.y**: returns the y‑coordinate(s) of a pixel or list / nested lists of pixels.

  Parameters:
  - pixel: `Pixel`, `List[Pixel]`, or nested lists of `Pixel`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.c**: returns the color(s) of a pixel or list / nested lists of pixels.

  Parameters:
  - pixel: `Pixel`, `List[Pixel]`, or nested lists of `Pixel`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.max_x**: returns the maximum x coordinate within a grid or grids.

  Parameters:
  - grid: `GridObject`, `List[GridObject]`, or `List[List[GridObject]]`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.max_y**: returns the maximum y coordinate within a grid or grids.

  Parameters:
  - grid: `GridObject`, `List[GridObject]`, or `List[List[GridObject]]`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.width**: returns the width of a grid or list of grids.

  Parameters:
  - grid: `GridObject`, `List[GridObject]`, or `List[List[GridObject]]`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.height**: returns the height of a grid or list of grids.

  Parameters:
  - grid: `GridObject`, `List[GridObject]`, or `List[List[GridObject]]`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.ul_x**: returns the stored upper‑left x coordinate of a grid or grids.

  Parameters:
  - grid: `GridObject`, `List[GridObject]`, or `List[List[GridObject]]`.

  Returns:
  - int, `List[int]`, or nested lists of int.

**.ul_y**: returns the stored upper‑left y coordinate of a grid or grids.

  Parameters:
  - grid: `GridObject`, `List[GridObject]`, or `List[List[GridObject]]`.

  Returns:
  - int, `List[int]`, or nested lists of int.

