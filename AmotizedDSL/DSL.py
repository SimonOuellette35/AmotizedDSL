from typing import List, TypeVar, Union
import numpy as np
import math
import copy


T = TypeVar('T')
COLOR = TypeVar('COLOR', bound=int)
DIM = TypeVar('COLOR', bound=int)

class Pixel:
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c

    def __str__(self):
        """
        Returns a string representation of the Grid instance.
        """
        return f"Pixel({self.x}, {self.y}, {self.c})"

    def __repr__(self):
        """
        Returns a string representation of the Grid instance.
        This method is used when the object is represented in the interactive shell.
        """
        return self.__str__()

class GridObject:

    def __init__(self, pixels, ul_x=0, ul_y=0):
        self.ul_x = ul_x
        self.ul_y = ul_y
        
        # Subtract ul_x from x and ul_y from y for all pixels
        def adjust_pixels(pixels):
            adjusted = []
            for p in pixels:
                tmp_pixel = Pixel(p.x - ul_x, p.y - ul_y, p.c)
                adjusted.append(tmp_pixel)
            return adjusted

        self.pixels = adjust_pixels(pixels)
        self.min_y = min(pixel.y for pixel in self.pixels)
        self.max_y = max(pixel.y for pixel in self.pixels)
        self.min_x = min(pixel.x for pixel in self.pixels)
        self.max_x = max(pixel.x for pixel in self.pixels)
        
        self.height = self.max_y - self.min_y + 1
        self.width = self.max_x - self.min_x + 1


    @staticmethod
    def from_grid(cells):
        pixels = []
        for y, row in enumerate(cells):
            for x, color in enumerate(row):
                pixels.append(Pixel(int(x), int(y), int(color)))
        return GridObject(pixels)

    @property
    def cells(self):
        return self.to_grid_tuples()

    def cells_as_numpy(self):
        return self.to_grid_numpy()

    def to_llm(self):
        ''' 
        Outputs the grid as a string for LLM prompts
        '''
        grid = self.to_grid()
        grid_str = '['
        for row in grid:
            row_str = ''
            for cell in row:
                row_str += str(int(cell))
            row_str += '\n'
            grid_str += row_str
        grid_str += ']\n'

        return grid_str

    def to_grid(self):
        # Create a 2D list filled with zeros (black)
        grid = np.zeros((self.max_y+1, self.max_x+1))

        # Fill in the colors from self.pixels
        for px in self.pixels:
            grid[px.y, px.x] = px.c

        return grid

    def to_grid_tuples(self):
        if len(self.pixels) == 0:
            return tuple()

        grid = self.to_grid()

        # Convert the 2D list to a tuple of tuples
        return tuple(tuple(row) for row in grid)

    def to_grid_numpy(self):
        if len(self.pixels) == 0:
            return np.array([])

        grid = self.to_grid()
        return grid

    @staticmethod
    def get_grid_list(grid, objects_mask):
        ''' 
        From a grid and an object instances mask, generate a list of Grid instances
        corresponding to the separate objects in the grid (excluding the background)

        Objects_mask is a 2D matrix in which each (x, y) coordinate corresponds to the index
        of the object instance this pixel belongs to.

        Returns a list of GridObject instances that are all the foreground objects.
        '''

        # Find all unique instance IDs in the object mask (excluding 0 which is background)
        instance_ids = np.unique(objects_mask)
        instance_ids = instance_ids[instance_ids != 0]  # Exclude 0 (background)

        grid_list = []
        for id in instance_ids:
            # Get all (x, y) coordinates in objects_mask that correspond to the value id
            coords = np.argwhere(objects_mask == id)
            coords_set = set(map(tuple, coords))  # Convert to set of (y, x) tuples for fast lookup

            # From grid.pixels, select all pixel instances whose .x and .y match these coordinates
            object_pixels = [pixel for pixel in grid.pixels if (pixel.y, pixel.x) in coords_set]

            ul_x = min(pixel.x for pixel in object_pixels) if object_pixels else 0
            ul_y = min(pixel.y for pixel in object_pixels) if object_pixels else 0

            # create a new Grid instance for this object and add to grid_list
            new_grid = GridObject(object_pixels, ul_x, ul_y)
            grid_list.append(new_grid)

        # get all pixels for mask idx 0 (the background)
        bg_coords = np.argwhere(objects_mask == 0)
        bg_coords_set = set(map(tuple, bg_coords))  # (y, x) tuples
        bg_pixels = [pixel for pixel in grid.pixels if (pixel.y, pixel.x) in bg_coords_set]

        # Find the most common color in the background pixels
        if bg_pixels:
            from collections import Counter
            color_counts = Counter([pixel.c for pixel in bg_pixels])
            most_common_color = color_counts.most_common(1)[0][0]
        else:
            most_common_color = 0  # fallback if no background pixels

        # Now, fill in missing background pixels (i.e., grid cells where mask is NOT 0)

        # Find all coords that are not in the background (i.e., mask != 0)
        non_bg_coords = set((pixel.y, pixel.x) for pixel in grid.pixels if objects_mask[pixel.y, pixel.x] != 0)

        # Find missing background coords (i.e., those not in bg_coords_set)
        missing_bg_coords = non_bg_coords - set((pixel.y, pixel.x) for pixel in bg_pixels)

        # Add missing background pixels with the most common color
        for y, x in missing_bg_coords:
            bg_pixels.append(type(bg_pixels[0])(x, y, most_common_color) if bg_pixels else Pixel(x, y, most_common_color))

        bg = GridObject(bg_pixels)

        return grid_list, bg
    
    def __str__(self):
        """
        Returns a string representation of the Grid instance.
        """
        header = f"Upper-left coords: ({self.ul_x}, {self.ul_y})\n"
        header += f"Height: {self.height}, Width: {self.width}\n"
        header += "Cells:\n"

        # Convert cells to a formatted string, ensuring integer display
        cells_str = '\n'.join([' '.join(map(lambda x: str(int(x)), row)) for row in self.cells])
        return header + cells_str

    def __repr__(self):
        """
        Returns a string representation of the Grid instance.
        This method is used when the object is represented in the interactive shell.
        """
        return self.__str__()

def inverse_lookup(idx):
    for key, val in prim_indices.items():
        if val == idx:
            return key
        
    return None

def code_to_token_id(code):
    """
    Given a code (value from text_to_code), find the corresponding key in text_to_code,
    then look up this key in prim_indices to get the token id.
    """
    for k, v in text_to_code.items():
        if v == code:
            return prim_indices[k]
    
    return prim_indices[code]

# =================================================================== Primitives and their associated indices ===================================================================

prim_indices = {
    # Integer constants
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,

    # Main functional primitives
    'identity': 10, 
    'get_objects1': 11,
    'color_set': 12,
    'equal': 13,
    'not_equal': 14,
    'switch': 15,
    'index': 16,
    'add': 17,
    'sub': 18,
    'div': 19,
    'mul': 20,
    'mod': 21,
    'sin_half_pi': 22,
    'cos_half_pi': 23,
    'or': 24,
    'and': 25,
    'xor': 26,    
    'arg_min': 27,
    'arg_max': 28,
    'crop': 29,
    'colorOf': 30,
    'set_pixels': 31,
    'set_x': 32,
    'set_y': 33,
    'new_grid': 34,
    'keep': 35,
    'exclude': 36,
    'count_values': 37,
    'rebuild_grid': 38,
    'del': 39,

    # Object attributes
    '.x': 40,        # PIXEL attribute
    '.y': 41,        # PIXEL attribute
    '.c': 42,        # PIXEL attribute
    '.max_x': 43,    # Grid attribute
    '.max_y': 44,    # Grid attribute
    '.width': 45,    # Grid attribute
    '.height': 46,    # Grid attribute
    '.ul_x': 47,     # Grid attribute
    '.ul_y': 48      # Grid attribute
}

text_to_code = {
    # Main functional primitives
    'identity': 'id', 
    'get_objects': 'obj',
    'color_set': 'col',
    'equal': 'eq',
    'not_equal': 'neq',
    'switch': 'if',
    'index': 'idx',
    'add': '+',
    'sub': '-',
    'div': '/',
    'mul': '*',
    'mod': '%',
    'sin_half_pi': 'sin',
    'cos_half_pi': 'cos',
    'set_pixels': 'spx',
    'set_x': 'sx',
    'set_y': 'sy',
    'new_grid': 'new',
    'exclude': 'exc',
    'count_values': 'cval',
    'rebuild_grid': 'rbld',

    '.max_x': '.mx',    # Grid attribute
    '.max_y': '.my',    # Grid attribute
    '.width': '.w',    # Grid attribute
    '.height': '.h',    # Grid attribute

}
# ======================================================================== Implementation of DSL ========================================================================

def new_grid(w: int, h: int, bg_color) -> GridObject:
    if isinstance(bg_color, List):
        bg_color = bg_color[0]

    if isinstance(w, List):
        w = w[0]

    if isinstance(h, List):
        h = h[0]

    cells = np.ones((h, w)) * bg_color
    return GridObject.from_grid(cells)

def get_width(g: Union[GridObject, List[GridObject]]) -> Union[DIM, List[DIM]]:
    if isinstance(g, GridObject):
        return g.width
    else:
        widths = []
        for grid in g:
            widths.append(grid.width)
        return widths

def get_height(g: Union[GridObject, List[GridObject]]) -> Union[DIM, List[DIM]]:
    if isinstance(g, GridObject):
        return g.height
    else:
        heights = []
        for grid in g:
            heights.append(grid.height)
        return heights

def get_x(p: Union[Pixel, List[Pixel], List[List[Pixel]]]) -> Union[COLOR, List[COLOR], List[List[COLOR]]]:
    if isinstance(p, List) and (isinstance(p[0], List) or isinstance(p[0], np.ndarray)):
        x_value_lists = []
        for px_list in p:
            x_values = []
            for px in px_list:
                x_values.append(px.x)

            x_value_lists.append(x_values)
        return x_value_lists
    
    elif isinstance(p, List) or isinstance(p, np.ndarray):
        x_values = []
        for px in p:
            x_values.append(px.x)
        return x_values
    else:
        return p.x

def get_y(p: Union[Pixel, List[Pixel], List[List[Pixel]]]) -> Union[COLOR, List[COLOR], List[List[COLOR]]]:
    if isinstance(p, List) and (isinstance(p[0], List) or isinstance(p[0], np.ndarray)):
        y_value_lists = []
        for px_list in p:
            y_values = []
            for px in px_list:
                y_values.append(px.y)

            y_value_lists.append(y_values)
        return y_value_lists

    elif isinstance(p, List) or isinstance(p, np.ndarray):
        y_values = []
        for px in p:
            y_values.append(px.y)
        return y_values
    else:
        return p.y

def get_color(p: Union[Pixel, List[Pixel], List[List[Pixel]]]) -> Union[COLOR, List[COLOR], List[List[COLOR]]]:
    if isinstance(p, List) and isinstance(p[0], List):
        c_value_lists = []
        for px_list in p:
            c_values = []
            for px in px_list:
                c_values.append(px.c)

            c_value_lists.append(c_values)
        return c_value_lists
    
    elif isinstance(p, List):
        c_values = []
        for px in p:
            c_values.append(px.c)
        return c_values
    else:
        return p.c

def get_index(list: List[T], i: int) -> Union[T, List[T]]:
    if isinstance(list[0], List):
        output = []
        for sublist in list:
            output.append(sublist[i])
        return output
    else:
        return list[i]

def count_values(values: List[int], data_list: Union[List[int], GridObject]) -> List[int]:

    counts = [0] * len(values)
    if isinstance(data_list, GridObject):
        for v_idx, v in enumerate(values):
            for px in data_list.pixels:
                if v == px[2]:
                    counts[v_idx] += 1
    else:
        for v_idx, v in enumerate(values):
            for d in data_list:
                if v == d:
                    counts[v_idx] += 1

    return counts

def sin_half_pi(val) -> Union[int, List[int]]:
    # sin(pi/2 * val)
    if isinstance(val, List):
        output_list = []
        for v in val:
            output_list.append(int(math.sin((np.pi / 2.) * v)))

        return output_list
    else:
        return int(math.sin((np.pi / 2.) * val))

def cos_half_pi(val) -> Union[int, List[int]]:
    # cos(pi/2 * val)
    if isinstance(val, List):
        output_list = []
        for v in val:
            output_list.append(int(math.cos((np.pi / 2.) * v)))

        return output_list
    else:
        return int(math.cos((np.pi / 2.) * val))

def addition(a: Union[int, List[int], List[List[int]]], 
             b: Union[int, List[int], List[List[int]]]) -> Union[int, List[int]]:
    if isinstance(a, List) and isinstance(a[0], List) and isinstance(b, List) and isinstance(b[0], List):
        output_sum_lists = []
        for list_idx in range(len(a)):
            output_sums = []
            for elem_idx in range(len(a[list_idx])):
                output_sums.append(a[list_idx][elem_idx] + b[list_idx][elem_idx])
            
            output_sum_lists.append(output_sums)
        return output_sum_lists
    elif isinstance(a, List) and isinstance(a[0], List) and isinstance(b, List):
        output_sum_lists = []
        for list_idx in range(len(a)):
            output_sums = []
            for elem_idx in range(len(a[list_idx])):
                output_sums.append(a[list_idx][elem_idx] + b[list_idx])
            output_sum_lists.append(output_sums)
        return output_sum_lists
    elif isinstance(b, List) and isinstance(b[0], List) and isinstance(a, List):
        output_sum_lists = []
        for list_idx in range(len(b)):
            output_sums = []
            for elem_idx in range(len(b[list_idx])):
                output_sums.append(b[list_idx][elem_idx] + a[list_idx])
            output_sum_lists.append(output_sums)
        return output_sum_lists    
    elif isinstance(a, List) and isinstance(b, List):
        output_sums = []
        for idx in range(len(a)):
            output_sums.append(a[idx] + b[idx])
        return output_sums
    elif isinstance(a, List):
        output_sums = []
        for idx in range(len(a)):
            output_sums.append(a[idx] + b)
        return output_sums
    elif isinstance(b, List):
        output_sums = []
        for idx in range(len(b)):
            output_sums.append(a + b[idx])
        return output_sums
    else:
        return a + b

def subtraction(a: Union[int, List[int], List[List[int]]], 
                b: Union[int, List[int], List[List[int]]]) -> Union[int, List[int]]:
    if isinstance(a, List) and isinstance(a[0], List) and isinstance(b, List) and isinstance(b[0], List):
        output_subs_lists = []
        for list_idx in range(len(a)):
            output_subs = []
            for elem_idx in range(len(a[list_idx])):
                result = a[list_idx][elem_idx] - b[list_idx][elem_idx]
                output_subs.append(result)
            
            output_subs_lists.append(output_subs)
        return output_subs_lists
    elif isinstance(a, List) and isinstance(a[0], List) and isinstance(b, List):
        output_sub_lists = []
        for list_idx in range(len(a)):
            output_subs = []
            for elem_idx in range(len(a[list_idx])):
                result = a[list_idx][elem_idx] - b[list_idx]
                output_subs.append(result)
            output_sub_lists.append(output_subs)
        return output_sub_lists
    elif isinstance(b, List) and isinstance(b[0], List) and isinstance(a, List):
        output_sub_lists = []
        for list_idx in range(len(b)):
            output_subs = []
            for elem_idx in range(len(b[list_idx])):
                result = a[list_idx] - b[list_idx][elem_idx]
                output_subs.append(result)
            output_sub_lists.append(output_subs)
        return output_sub_lists    
    elif isinstance(a, list) and isinstance(b, list):
        output_subs = []
        for idx in range(len(a)):
            result = a[idx] - b[idx]
            output_subs.append(result)
        return output_subs
    elif isinstance(a, list):
        output_subs = []
        for idx in range(len(a)):
            result = a[idx] - b
            output_subs.append(result)
        return output_subs
    elif isinstance(b, list):
        output_subs = []
        for idx in range(len(b)):
            result = a - b[idx]
            output_subs.append(result)
        return output_subs    
    else:
        result = a - b
        return result

def division(a: Union[int, float, List[int], List[List[int]]], 
             b: Union[int, float, List[int], List[List[int]]]) -> Union[int, List[int]]:
    if isinstance(a, list) and isinstance(b, list):
        output_quotients = []
        for idx in range(len(a)):
            output_quotients.append(int(a[idx] // b[idx]))
        return output_quotients
    elif isinstance(a, list):
        output_quotients = []
        for idx in range(len(a)):
            output_quotients.append(int(a[idx] // b))
        return output_quotients
    elif isinstance(b, list):
        output_quotients = []
        for idx in range(len(b)):
            output_quotients.append(int(a // b[idx]))
        return output_quotients
    else:
        return int(a // b)

def multiplication(a: Union[int, List[int], List[List[int]]], 
                   b: Union[int, List[int], List[List[int]]]) -> Union[int, List[int]]:
    if isinstance(a, list) and isinstance(b, list):
        output_products = []
        for idx in range(len(a)):
            output_products.append(a[idx] * b[idx])
        return output_products
    elif isinstance(a, list):
        output_products = []
        for idx in range(len(a)):
            output_products.append(a[idx] * b)
        return output_products
    elif isinstance(b, list):
        output_products = []
        for idx in range(len(b)):
            output_products.append(a * b[idx])
        return output_products
    else:
        return a * b

def modulo(a: Union[int, List[int], List[List[int]]], 
           b: Union[int, List[int], List[List[int]]]) -> Union[int, List[int]]:
    if isinstance(a, list) and isinstance(b, list):
        output_mods = []
        for idx in range(len(a)):
            output_mods.append(a[idx] % b[idx])
        return output_mods
    elif isinstance(a, list):
        output_mods = []
        for idx in range(len(a)):
            output_mods.append(a[idx] % b)
        return output_mods
    elif isinstance(b, list):
        output_mods = []
        for idx in range(len(b)):
            output_mods.append(a % b[idx])
        return output_mods
    else:
        return a % b

def between(a: int, b: int, c: int) -> bool:
    if a < c and a >= b:
        return True
    else:
        return False

def equal(a, b):
    def equal1(a: int, b: int) -> bool:
        return a == b

    def equal2(a: List[int], b: int) -> List[bool]:
        output = []
        for tmp_a in a:
            output.append(tmp_a == b)

        return output

    if isinstance(a, List):
        if isinstance(a[0], List):
            output = []
            for idx in range(len(a)):
                output.append(equal2(a[idx], b[idx]))
            
            return output
        else:
            return equal2(a, b)
    else:
        return equal1(a, b)


def not_equal(a: int, b: int) -> bool:
    return a != b

def greater_than(a: int, b: int) -> bool:
    if a > b:
        return True
    else:
        return False

def rebuild_grid(bg_grid: GridObject, obj_list: List[GridObject]) -> GridObject:
    '''
    Starting from an input grid, copy-paste the list of objects from obj_list onto
    the grid.
    '''
    output_pixels = bg_grid.pixels

    for obj_id, obj in enumerate(obj_list):
        # Then, set all pixels in output_pixels whose (x, y) match those in obj.pixels to their '.c' values
        pixel_map = {(p.x + obj.ul_x, p.y + obj.ul_y): p.c for p in obj.pixels}
        for pixel in output_pixels:
            if (pixel.x, pixel.y) in pixel_map:
                pixel.c = pixel_map[(pixel.x, pixel.y)]

    return GridObject(output_pixels)

def switch(conditions, operations, otherwise):
    '''
    This is essentially an if/else statement. The logic of this primitive is somewhat complex because we implicitly
    support "list overloading" in various ways. That is, switch(True, 0, 2) is valid, but also switch([True, False, False], 3, 9),
    and also switch([False, False, False, True, True], [2, 5, 3, 0, 1], [5, 6, 5, 1, 1]), etc. See comments later in the code to better
    understand how these work (also the README file).

    Parameters:
    @param conditions: can be a single boolean value, a list of booleans, or a list of lists of booleans.
                       A list of booleans implies a single if/else statement, but with a list type of output.
                       A list of lists implies as N-1 'elif' statements, where N is the length of the outer lists.
    @param operations: what to return when the condition is True. Can be a constant (integer/boolean), a list of
                       constants, or even a list of lists in the if/elif/.../else case.
    @param otherwise:  what to return when the "else" part of the statement if reached. Can be a constant or a list of
                       constants.

    When one of the output arguments (operations/otherwise) are lists of constants, it implies that we return the element of
    the list corresponding to the current element index being evaluated in the conditions list.
    '''

    def switch_single_constant(conditions, operations, otherwise):
        '''
        Here, we have 1 single boolean condition, or many conditions. If the latter, they have been preprocessed to be 
        individual booleans. This is why we loop over conditions.

        In the single constant case, this is basically:
        if conditions:
            return operations
        else:
            return otherwise
        '''
        for idx, cond in enumerate(conditions):
            if cond:
                if isinstance(operations, List) and len(operations) > 1:
                    return operations[idx]
                else:
                    return operations

        return otherwise

    def switch_single_list(conditions, operations, otherwise):
        '''
        Here we have a simple if/else statement, but the conditions and potentially operations and otherwise are lists.

        For example, we can have:
        conditions = [True, False, True]
        operations = 6
        otherwise = 3

        Which is:
        for cond in conditions:
            if cond:
                return operations
            else:
                return otherwise

        But we can also have:
        conditions = [True, False, True]
        operations = [2, 3, 5]
        otherwise = [1, 6, 9]

        Which is:
        output_list = []
        for idx, cond in enumerate(conditions):
            if cond:
                output_list.append(operations[idx])
            else:
                output_list.append(return otherwise[idx])
        '''
        list_output = []
        for elem_idx in range(len(conditions[0])):

            inner_conditions = []
            inner_operations = []
            inner_otherwise = []
            
            for cond_idx in range(len(conditions)):
                inner_conditions.append(conditions[cond_idx][elem_idx])

            if isinstance(operations, List):
                if len(operations) > 1:
                    inner_operations = operations[elem_idx]
                else:
                    inner_operations = operations[0]
            else:
                inner_operations = operations

            if isinstance(otherwise, List):
                inner_otherwise = otherwise[elem_idx]
            else:
                inner_otherwise = otherwise

            tmp_val = switch_single_constant(inner_conditions, inner_operations, inner_otherwise)
            list_output.append(tmp_val)

        return list_output

    def switch_many_lists(conditions, operations, otherwise):
        '''
        Here we have at least one "else if" branch in our statement. We are also forced to use
        lists as conditions, so as not to be ambiguous with respect to the single-condition list case.
        (These lists, however, can be of 1 element if needed)

        Example:
        conditions = [ [True, False, False], [False, False, True] ]
        operations = [8, 3]
        otherwise = [1, 5, 9]

        Which is:
        n = len(conditions[0])  # assumption: lists of many-condition cases must have the same number of elements.
        output_list = []
        for elem_idx in range(n):
            cond_found = False
            for cond_idx in range(len(conditions)):
                if conditions[cond_idx][elem_idx]:
                    output_list.append(operations[cond_idx])
                    cond_found = True
                    break

            if not cond_found:
                output_list.append(otherwise[elem_idx])
        '''
        list_output = []
        num_objects = len(conditions[0])

        for elem_idx in range(num_objects):
            inner_conditions = []
            inner_operations = []
            inner_otherwise = []

            for cond_idx in range(len(conditions)):
                inner_conditions.append(conditions[cond_idx][elem_idx])

            if isinstance(operations[0], List):
                for cond_idx in range(len(conditions)):
                    inner_operations.append(operations[cond_idx][elem_idx])
            else:
                inner_operations = operations

            if isinstance(otherwise, List):
                inner_otherwise = otherwise[elem_idx]
            else:
                inner_otherwise = otherwise

            tmp_val = switch_single_constant(inner_conditions, inner_operations, inner_otherwise)
            list_output.append(tmp_val)

        return list_output

    # Overloading for list processing. Here, we differentiate the 3 main cases: one single constant condtion,
    # 1 list condition, or many if "branches" (if/elif/elif/.../else).
    if isinstance(conditions[0], List):
        if len(conditions) > 1:
            # Here we have many conditions.
            return switch_many_lists(conditions, operations, otherwise)
        else:
            # Here there is 1 condition, but it's a list.
            return switch_single_list(conditions, operations, otherwise)
    else:
        # Here the condition is just a constant.
        return switch_single_constant(conditions, operations, otherwise)

def colorOf(g: GridObject, x, y) -> COLOR:
    def single_grid_colorOf(g, x, y):   
        if isinstance(x, list) and isinstance(y, list):
            color_list = []
            for idx in range(len(x)):
                color_list.append(int(g.cells[y[idx]][x[idx]]))
            return color_list
        else:
            return int(g.cells[y][x])
        
    if isinstance(g, List):
        outputs = []
        for idx, tmp_g in enumerate(g):
            outputs.append(single_grid_colorOf(tmp_g, x[idx], y[idx]))
        return outputs
    else:
        return single_grid_colorOf(g, x, y)

def colorSet(g: Union[GridObject, List[GridObject]]) -> Union[List[COLOR], List[List[COLOR]]]:
    if isinstance(g, GridObject):
        pixels = [pixel[2] for pixel in g.pixels]
        colors = list(set(pixels))
        colors.sort()
        return colors
    else:
        all_colors = []
        for grid in g:
            pixels = [pixel[2] for pixel in grid.pixels]
            colors = list(set(pixels))
            colors.sort()
            all_colors.append(colors)

        return all_colors

def keep(input_list: Union[List[int], List[GridObject]], flags: List[bool]) -> Union[List[int], List[GridObject]]:
    output = []
    for idx in range(len(input_list)):
        if flags[idx]:
            output.append(input_list[idx])
    return output

def exclude(input_list: Union[List[int], List[GridObject]], flags: List[bool]) -> Union[List[int], List[GridObject]]:
    output = []
    for idx in range(len(input_list)):
        if not flags[idx]:
            output.append(input_list[idx])
    
    return output

def arg_min(arg_list: List[int], val_list: List[int]) -> List[int]:
    if len(arg_list) != len(val_list):
        raise ValueError("arg_list and val_list must have same length")
    
    min_val = min(val_list)
    min_idx = val_list.index(min_val)
    return arg_list[min_idx]

def arg_max(arg_list: List[int], val_list: List[int]) -> List[int]:
    if len(arg_list) != len(val_list):
        raise ValueError("arg_list and val_list must have same length")
    
    max_val = max(val_list)
    max_idx = val_list.index(max_val)
    return arg_list[max_idx]

def logical_or(a: Union[bool, List[bool]], b: Union[bool, List[bool]]) -> Union[bool, List[bool]]:
    if isinstance(a, List) and isinstance(b, List):
        # List vs List: pairwise OR
        if len(a) != len(b):
            raise ValueError("Lists must have same length for pairwise OR")
        return [a[i] or b[i] for i in range(len(a))]
    elif isinstance(a, List):
        # List vs Single: each element OR'd with single boolean
        return [x or b for x in a]
    elif isinstance(b, List):
        # Single vs List: single boolean OR'd with each element
        return [a or x for x in b]
    else:
        # Single vs Single: simple OR
        return a or b

def logical_and(a: Union[bool, List[bool]], b: Union[bool, List[bool]]) -> Union[bool, List[bool]]:
    if isinstance(a, List) and isinstance(b, List):
        # List vs List: pairwise AND
        if len(a) != len(b):
            raise ValueError("Lists must have same length for pairwise OR")
        return [a[i] and b[i] for i in range(len(a))]
    elif isinstance(a, List):
        # List vs Single: each element AND'ed with single boolean
        return [x and b for x in a]
    elif isinstance(b, List):
        # Single vs List: single boolean AND'ed with each element
        return [a and x for x in b]
    else:
        # Single vs Single: simple AND
        return a and b

def logical_xor(a: Union[bool, List[bool]], b: Union[bool, List[bool]]) -> Union[bool, List[bool]]:
    if isinstance(a, List) and isinstance(b, List):
        # List vs List: pairwise XOR
        if len(a) != len(b):
            raise ValueError("Lists must have same length for pairwise XOR")
        return [a[i] ^ b[i] for i in range(len(a))]
    elif isinstance(a, List):
        # List vs Single: each element XOR'd with single boolean
        return [x ^ b for x in a]
    elif isinstance(b, List):
        # Single vs List: single boolean XOR'd with each element
        return [a ^ x for x in b]
    else:
        # Single vs Single: simple XOR
        return a ^ b

def set_pixels(target_grid: Union[GridObject, List[GridObject]], 
               set_x: Union[List[DIM], List[List[DIM]]], 
               set_y: Union[List[DIM], List[List[DIM]]],
               colors: Union[List[COLOR], List[List[COLOR]]]) -> Union[GridObject, List[GridObject]]:

    # if the target coord is out-of-bounds, extend the target grid as needed (this is especially useful for tiling tasks)
    def set_single_grid_pixels(target_grid: GridObject, set_x: Union[DIM, List[DIM]], set_y: Union[DIM, List[DIM]], colors: Union[COLOR, List[COLOR]]) -> GridObject:
        
        if isinstance(set_x, List):
            set_x = [int(x) for x in set_x]
        else:
            set_x = int(set_x)

        if isinstance(set_y, List):
            set_y = [int(y) for y in set_y]
        else:
            set_y = int(set_y)

        if isinstance(colors, List):
            colors = [int(c) for c in colors]
        else:
            colors = int(colors)

        # At least one of set_x, set_y or colors must be a List and this indicates the number of elements to populate
        # for the integer cases.
        n = 0
        if isinstance(set_x, List):
            n = len(set_x)
        elif isinstance(set_y, List):
            n = len(set_y)
        elif isinstance(colors, List):
            n = len(colors)
        else:
            # ERROR: set_single_grid_pixels must have at least 1 list-type argument among set_x, set_y and colors.
            return None

        if isinstance(set_x, int) or isinstance(set_x, np.int64):
            set_x = np.ones(n) * set_x

        if isinstance(set_y, int) or isinstance(set_y, np.int64):
            set_y = np.ones(n) * set_y

        if isinstance(colors, int) or isinstance(colors, np.int64):
            colors = np.ones(n) * colors

        # Filter out negative coordinates
        valid_indices = [i for i in range(n) if set_x[i] >= 0 and set_y[i] >= 0]
        if not valid_indices:
            return target_grid

        # Use only valid coordinates for max calculations
        valid_x = [set_x[i] for i in valid_indices]
        valid_y = [set_y[i] for i in valid_indices]
        
        max_x = max(target_grid.width, max(valid_x) + 1)
        max_y = max(target_grid.height, max(valid_y) + 1)
        new_cells = np.zeros((max_y, max_x))
        for y in range(target_grid.height):
            for x in range(target_grid.width):
                new_cells[y][x] = target_grid.cells[y][x]

        for idx in valid_indices:
            x_coord = int(set_x[idx])
            y_coord = int(set_y[idx])
            color = colors[idx]

            new_cells[y_coord, x_coord] = color

        return GridObject.from_grid(new_cells)

    if isinstance(target_grid, List):
        output_grids = []
        for idx, grid in enumerate(target_grid):
            tmp_out = set_single_grid_pixels(grid, set_x[idx], set_y[idx], colors[idx])
            output_grids.append(tmp_out)

        return output_grids
    elif isinstance(set_x, List) and isinstance(set_x[0], List):
        # In this case, there is one Grid instance, but we have a list of lists to set.
        for list_idx in range(len(set_x)):
            target_grid = set_single_grid_pixels(target_grid, set_x[list_idx], set_y[list_idx], colors[list_idx])
        return target_grid
    else:
        return set_single_grid_pixels(target_grid, set_x, set_y, colors)

def set_x(grid: Union[GridObject, List[GridObject]], x_values: Union[List[int], List[List[int]]]) -> Union[GridObject, List[GridObject]]:
    def set_grid_x(grid: GridObject, x_values: List[int]) -> GridObject:

        # Assign to each grid.pixels element's .x attribute the corresponding value in x_values
        new_pixels = []
        for idx, pixel in enumerate(grid.pixels):
            new_pixel = Pixel(x_values[idx] + grid.ul_x, pixel.y + grid.ul_y, pixel.c)
            new_pixels.append(new_pixel)
        return GridObject(new_pixels, grid.ul_x, grid.ul_y)
    
    if isinstance(grid, GridObject):
        return set_grid_x(grid, x_values)
    else:
        grid_list = []
        for g_idx, g in enumerate(grid):
            tmp_g = set_grid_x(g, x_values[g_idx])
            grid_list.append(tmp_g)

        return grid_list

def set_y(grid: Union[GridObject, List[GridObject]], y_values: Union[List[int], List[List[int]]]) -> Union[GridObject, List[GridObject]]:
    def set_grid_y(grid: GridObject, y_values: List[int]) -> GridObject:
        # Assign to each grid.pixels element's .y attribute the corresponding value in y_values
        new_pixels = []
        for idx, pixel in enumerate(grid.pixels):
            new_pixel = Pixel(pixel.x + grid.ul_x, y_values[idx] + grid.ul_y, pixel.c)
            new_pixels.append(new_pixel)
        return GridObject(new_pixels, grid.ul_x, grid.ul_y)
    
    if isinstance(grid, GridObject):
        return set_grid_y(grid, y_values)
    else:
        grid_list = []
        for g_idx, g in enumerate(grid):
            tmp_g = set_grid_y(g, y_values[g_idx])
            grid_list.append(tmp_g)

        return grid_list

def crop(g: Union[GridObject, List[GridObject]], x1, y1, x2, y2) -> GridObject:
    def crop_grid(g, x1, y1, x2, y2):
        new_pixels = []
        for pixel in g.pixels:
            if pixel[0] >= x1 and pixel[0] < x2:
                if pixel[1] >= y1 and pixel[1] < y2:
                    adjusted_pixel = (pixel[0] - x1, pixel[1] - y1, pixel[2])
                    new_pixels.append(adjusted_pixel)

        return GridObject(new_pixels)

    if isinstance(g, GridObject):
        return crop_grid(g, x1, y1, x2, y2)
    else:
        output_grids = []
        for g_idx, tmp_g in enumerate(g):
            if isinstance(x1, List):
                x1_inp = x1[g_idx]
            else:
                x1_inp = x1
                
            if isinstance(x2, List):
                x2_inp = x2[g_idx]
            else:
                x2_inp = x2

            if isinstance(y1, List):
                y1_inp = y1[g_idx]
            else:
                y1_inp = y1

            if isinstance(y2, List):
                y2_inp = y2[g_idx]
            else:
                y2_inp = y2

            tmp_out = crop_grid(tmp_g, x1_inp, y1_inp, x2_inp, y2_inp)
            output_grids.append(tmp_out)

        return output_grids

def max_x(g: Union[GridObject, List[GridObject]]) -> Union[int, List[DIM]]:
    if isinstance(g, List):
        output = []
        for tmp_g in g:
            output.append(tmp_g.max_x)
        return output
    else:
        return g.max_x

def max_y(g: Union[GridObject, List[GridObject]]) -> Union[int, List[DIM]]:
    if isinstance(g, List):
        output = []
        for tmp_g in g:
            output.append(tmp_g.max_y)
        return output
    else:
        return g.max_y

def get_ul_x(g: Union[GridObject, List[GridObject]]) -> Union[DIM, List[DIM]]:
    if isinstance(g, GridObject):
        return g.ul_x
    else:
        output = []
        for grid in g:
            output.append(grid.ul_x)
        return output

def get_ul_y(g: Union[GridObject, List[GridObject]]) -> Union[DIM, List[DIM]]:
    if isinstance(g, GridObject):
        return g.ul_y
    else:
        output = []
        for grid in g:
            output.append(grid.ul_y)
        return output
   

# =================================================================== Number of arguments for each primitive =================================================================

arg_counts = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    2,
    3,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    5,
    3,
    4,  # set_pixels
    2,  # set_x
    2,  # set_y
    3,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
]

# ======================================================= Mappings from primitive name to actual function ======================================================================

semantics = {
    # Integer constants
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,

    # Main functional primitives
    'identity': lambda x: x,
    'get_objects': lambda x: x,        # TODO: to be implemented as a neural primitive
    'color_set': colorSet,
    'equal': equal,
    'not_equal': not_equal,
    'switch': switch,
    'index': get_index,
    'add': addition,
    'sub': subtraction,
    'div': division,
    'mul': multiplication,
    'mod': modulo,
    'sin_half_pi': sin_half_pi,
    'cos_half_pi': cos_half_pi,
    'or': logical_or,
    'and': logical_and,
    'xor': logical_xor,
    'arg_min': arg_min,
    'arg_max': arg_max,
    'crop': crop,
    'colorOf': colorOf,

    # Given a list of x coordinates and y coordinates for the pixels to modify in the target grid,
    # is sets those pixels' colors to the values passed as fourth argument.
    'set_pixels': set_pixels,
    'set_x': set_x,
    'set_y': set_y,
    'new_grid': new_grid,
    'keep': keep,
    'exclude': exclude,
    'count_values': count_values,
    'rebuild_grid': rebuild_grid,
    'del': lambda x: x,       # This is actually a special primitive that is implemented at the program execution level
                              # where state memroy management is possible.

    # Object attributes
    '.x': lambda pixel: get_x(pixel),
    '.y': lambda pixel: get_y(pixel),
    '.c': lambda pixel: get_color(pixel),
    '.max_x': max_x,
    '.max_y': max_y,
    '.width': get_width,
    '.height': get_height,
    '.ul_x': get_ul_x,
    '.ul_y': get_ul_y
}

pixel_attributes = ['.x', '.y', '.c']

# ============================================================================== Useful sub-routines =========================================================================

def get_subroutine_rot90(ctx_ref, start_ref):
    program = []

    program.append(('sub', [(ctx_ref+0, '.max_y'), (ctx_ref+0, '.y')]))
    program.append(('colorOf', [ctx_ref+0, (ctx_ref+0, '.x'), start_ref+1]))
    program.append(('del', [start_ref+1]))

    return program, start_ref+1

def get_subroutine_rot180(ctx_ref, start_ref):
    program = []

    program.append(('sub', [(ctx_ref+0, '.max_x'), (ctx_ref+0, '.x')]))
    program.append(('sub', [(ctx_ref+0, '.max_y'), (ctx_ref+0, '.y')]))
    program.append(('colorOf', [ctx_ref+0, start_ref+1, start_ref+2]))
    program.append(('del', [start_ref+1]))
    program.append(('del', [start_ref+1]))

    return program, start_ref+1

def get_subroutine_rot270(ctx_ref, start_ref):
    program = []

    program.append(('sub', [(ctx_ref+0, '.max_x'), (ctx_ref+0, '.x')])),
    program.append(('colorOf', [ctx_ref+0, start_ref+1, (ctx_ref+0, '.y')]))
    program.append(('del', [start_ref+1]))

    return program, start_ref+1

def get_subroutine_hmirror(ctx_ref, start_ref):
    program = []

    program.append(('sub', [(ctx_ref+0, '.max_x'), (ctx_ref+0, '.x')])),
    program.append(('colorOf', [ctx_ref+0, start_ref+1, (ctx_ref+0, '.y')]))
    program.append(('del', [start_ref+1]))

    return program, start_ref+1

def get_subroutine_vmirror(ctx_ref, start_ref):
    program = []

    program.append(('sub', [(ctx_ref+0, '.max_y'), (ctx_ref+0, '.y')])),
    program.append(('colorOf', [ctx_ref+0, (ctx_ref+0, '.x'), start_ref+1]))
    program.append(('del', [start_ref+1]))

    return program, start_ref+1
