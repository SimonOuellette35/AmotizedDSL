from typing import List, TypeVar, Union
import numpy as np
import math
import copy
from collections import Counter


T = TypeVar('T')
COLOR = TypeVar('COLOR', bound=int)
DIM = TypeVar('COLOR', bound=int)

class Pixel:
    def __init__(self, x, y, c):
        self.x = int(x)
        self.y = int(y)
        self.c = int(c)

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

    def __eq__(self, pixel):
        if self.x == pixel.x and self.y == pixel.y and self.c == pixel.c:
            return True
        else:
            return False


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
    def from_grid(cells, ul_x = 0, ul_y = 0):
        pixels = []
        for y, row in enumerate(cells):
            for x, color in enumerate(row):
                pixels.append(Pixel(int(x) + ul_x, int(y) + ul_y, int(color)))

        return GridObject(pixels, ul_x, ul_y)

    @property
    def cells(self):
        return self.to_grid_tuples()

    def cells_as_numpy(self):
        return self.to_grid_numpy().astype(int)

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
            grid[px.y, px.x] = int(px.c)

        return grid

    def to_grid_tuples(self):
        if len(self.pixels) == 0:
            return tuple()

        grid = self.to_grid()

        # Ensure all values are integers before converting to tuple of tuples
        return tuple(tuple(int(cell) for cell in row) for row in grid)

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

        grid: [k, GridObject]
        objects_mask: [k, np.ndarray]

        Returns a list of GridObject instances that are all the foreground objects.
        '''
        k_obj_lists = []
        k_bgs = []
        for k in range(len(grid)):
            # Find all unique instance IDs in the object mask (excluding 0 which is background)
            instance_ids = np.unique(objects_mask[k])
            instance_ids = instance_ids[instance_ids != 0]  # Exclude 0 (background)

            grid_list = []
            for id in instance_ids:
                # Get all (x, y) coordinates in objects_mask that correspond to the value id
                coords = np.argwhere(objects_mask[k] == id)
                coords_set = set(map(tuple, coords))  # Convert to set of (y, x) tuples for fast lookup

                # From grid.pixels, select all pixel instances whose .x and .y match these coordinates
                object_pixels = [pixel for pixel in grid[k].pixels if (pixel.y, pixel.x) in coords_set]

                ul_x = min(pixel.x for pixel in object_pixels) if object_pixels else 0
                ul_y = min(pixel.y for pixel in object_pixels) if object_pixels else 0

                # create a new Grid instance for this object and add to grid_list
                new_grid = GridObject(object_pixels, ul_x, ul_y)
                grid_list.append(new_grid)

            k_obj_lists.append(grid_list)

            # get all pixels for mask idx 0 (the background)
            bg_coords = np.argwhere(objects_mask[k] == 0)
            bg_coords_set = set(map(tuple, bg_coords))  # (y, x) tuples
            bg_pixels = [pixel for pixel in grid[k].pixels if (pixel.y, pixel.x) in bg_coords_set]

            # Find the most common color in the background pixels
            if bg_pixels:
                color_counts = Counter([pixel.c for pixel in bg_pixels])
                most_common_color = color_counts.most_common(1)[0][0]
            else:
                most_common_color = 0  # fallback if no background pixels

            # Now, fill in missing background pixels (i.e., grid cells where mask is NOT 0)

            # Find all coords that are not in the background (i.e., mask != 0)
            non_bg_coords = set((pixel.y, pixel.x) for pixel in grid[k].pixels if objects_mask[k][pixel.y, pixel.x] != 0)

            # Find missing background coords (i.e., those not in bg_coords_set)
            missing_bg_coords = non_bg_coords - set((pixel.y, pixel.x) for pixel in bg_pixels)

            # Add missing background pixels with the most common color
            for y, x in missing_bg_coords:
                bg_pixels.append(type(bg_pixels[0])(x, y, most_common_color) if bg_pixels else Pixel(x, y, most_common_color))

            bg = GridObject(bg_pixels)
            k_bgs.append(bg)

        return k_obj_lists, k_bgs
    
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
    'get_objects': 11,
    'get_bg': 12,
    'color_set': 13,
    'equal': 14,
    'not_equal': 15,
    'less_than': 16,
    'switch': 17,
    'index': 18,
    'unique': 19,
    'add': 20,
    'sub': 21,
    'div': 22,
    'mul': 23,
    'mod': 24,
    'sin_half_pi': 25,
    'cos_half_pi': 26,
    'or': 27,
    'and': 28,
    'xor': 29,    
    'arg_min': 30,
    'arg_max': 31,
    'crop': 32,
    'colorOf': 33,
    'set_pixels': 34,
    'set_x': 35,
    'set_y': 36,
    'set_color': 37,
    'new_grid': 38,
    'keep': 39,
    'exclude': 40,
    'set_difference': 41,
    'count_values': 42,
    'count_items': 43,
    'rebuild_grid': 44,
    'neighbours4': 45,
    'neighbours8': 46,
    'del': 47,

    # Object attributes
    '.x': 48,        # PIXEL attribute
    '.y': 49,        # PIXEL attribute
    '.c': 50,        # PIXEL attribute
    '.max_x': 51,    # Grid attribute
    '.max_y': 52,    # Grid attribute
    '.width': 53,    # Grid attribute
    '.height': 54,    # Grid attribute
    '.ul_x': 55,     # Grid attribute
    '.ul_y': 56      # Grid attribute
}

text_to_code = {
    # Main functional primitives
    'identity': 'id', 
    'get_objects': 'obj',
    'get_bg': 'bg',
    'color_set': 'col',
    'equal': 'eq',
    'not_equal': 'neq',
    'switch': 'if',
    'index': 'idx',
    'unique': 'uq',
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
    'set_color': 'sc',
    'new_grid': 'new',
    'exclude': 'exc',
    'set_difference': 'dif',
    'count_values': 'cval',
    'rebuild_grid': 'rbld',
    'neighbours4': 'nb4',
    'neighbours8': 'nb8',
    'count_items': 'len',
    'less_than': '<',

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

def get_objects(grid: GridObject, obj_mask: List[int]) -> List[GridObject]:
    """
    Extracts all distinct objects from the grid using the object mask.
    Each unique non-zero value in obj_mask corresponds to a separate object.
    For each such value, collect all pixels with that value, using their color from the grid,
    and create a GridObject for that object.
    """
    objects = []
    # Find all unique instance IDs in the object mask (excluding 0 which is background)
    instance_ids = np.unique(obj_mask)
    instance_ids = instance_ids[instance_ids != 0]  # Exclude 0 (background)

    for obj_id in instance_ids:
        # Get all (y, x) coordinates in obj_mask that correspond to the value obj_id
        coords = np.argwhere(obj_mask == obj_id)
        pixels = []
        for y, x in coords:
            color = None
            # grid can be GridObject or a numpy array
            if isinstance(grid, GridObject):
                # Find the pixel in grid.pixels with matching x, y
                for px in grid.pixels:
                    if px.x == x and px.y == y:
                        color = px.c
                        break
                if color is None:
                    # fallback: get color from grid.to_grid()
                    color = int(grid.to_grid()[y, x])
            else:
                color = int(grid[y, x])
            pixels.append(Pixel(int(x), int(y), int(color)))
        if pixels:
            # Optionally, set ul_x, ul_y to min x/y of pixels
            ul_x = min(pixel.x for pixel in pixels)
            ul_y = min(pixel.y for pixel in pixels)
            obj = GridObject(pixels, ul_x, ul_y)
            objects.append(obj)
    
    return objects


def get_bg(grid: GridObject, obj_mask: List[int]) -> GridObject:
    """
    Extracts the background object (instance id 0) from the grid using the object mask.
    All pixels in obj_mask with value 0 are considered background.
    Returns a single GridObject representing the background.
    """
    # Get all (y, x) coordinates in obj_mask that correspond to the value 0 (background)
    bg_coords = np.argwhere(obj_mask == 0)
    pixels = []
    for y, x in bg_coords:
        color = None
        if isinstance(grid, GridObject):
            # Find the pixel in grid.pixels with matching x, y
            for px in grid.pixels:
                if px.x == x and px.y == y:
                    color = px.c
                    break
            if color is None:
                color = int(grid.to_grid()[y, x])
        else:
            color = int(grid[y, x])
        pixels.append(Pixel(int(x), int(y), int(color)))
    if pixels:
        ul_x = min(pixel.x for pixel in pixels)
        ul_y = min(pixel.y for pixel in pixels)
        bg_obj = GridObject(pixels, ul_x, ul_y)
        return bg_obj
    else:
        # If no background pixels, return an empty GridObject
        return GridObject([])


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

def unique(data: Union[List[int], List[List[int]], List[List[List[int]]]]) -> Union[List[int], List[List[int]], List[List[List[int]]]]:
    if isinstance(data[0], List):
        if isinstance(data[0][0], List):
            unique_sets = []
            for inner_list in data:
                unique_set_list = []
                for inner_inner_list in inner_list:
                    unique_set_list.append(list(dict.fromkeys(inner_inner_list)))

                unique_sets.append(unique_set_list)

            return unique_sets
        else:
            unique_sets = []
            for inner_list in data:
                unique_sets.append(list(dict.fromkeys(inner_list)))

            return unique_sets
    else:
        return list(dict.fromkeys(data))
    
def get_x(p: Union[Pixel, List[Pixel], List[List[Pixel]]]) -> Union[COLOR, List[COLOR], List[List[COLOR]]]:
    if isinstance(p, List) and (isinstance(p[0], List) or isinstance(p[0], np.ndarray)):
        x_value_lists = []
        for px_list in p:
            x_values = []
            if isinstance(px_list[0], List):
                for px_inner_list in px_list:
                    x_value_list = []
                    for px in px_inner_list:
                        x_value_list.append(px.x)

                    x_values.append(x_value_list)
            else:
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
            if isinstance(px_list[0], List):
                for px_inner_list in px_list:
                    y_value_list = []
                    for px in px_inner_list:
                        y_value_list.append(px.y)

                    y_values.append(y_value_list)
            else:
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
            if i < 0 or i >= len(sublist):
                output.append(0)
            else:
                output.append(sublist[i])
        return output
    else:
        if i < 0 or i >= len(list):
            return 0
        else:
            return list[i]

def less_than(a: Union[int, List[int], List[List[int]]], b: Union[int, List[int], List[List[int]]]) -> Union[bool, List[bool], List[List[bool]]]:
    # Handle all combinations of a and b being int or List (including nested Lists)
    if isinstance(a, int) and isinstance(b, int):
        return a < b
    elif isinstance(a, int) and isinstance(b, list):
        # a is scalar, b is list
        return [a < bi for bi in b]
    elif isinstance(a, list) and isinstance(b, int):
        if isinstance(a[0], list):
            if isinstance(a[0][0], list):
                output_list = []
                for inner_list in a:
                    tmp_list = [[aij < b for aij in ai] for ai in inner_list]
                    output_list.append(tmp_list)

                return output_list
            else:
                # a is list of lists, b is scalar
                return [[aij < b for aij in ai] for ai in a]
        else:
            # a is list, b is scalar
            return [ai < b for ai in a]
    elif isinstance(a, list) and isinstance(b, list):
        # Both are lists
        if len(a) != len(b):
            raise ValueError("Lists must have same length for elementwise less_than")
        # Check for nested lists
        if len(a) > 0 and isinstance(a[0], list) and isinstance(b[0], list):
            # Both are list of lists
            if len(a) != len(b):
                raise ValueError("Outer lists must have same length for elementwise less_than")
            return [less_than(ai, bi) for ai, bi in zip(a, b)]
        elif len(a) > 0 and isinstance(a[0], list):
            # a is list of lists, b is flat list
            return [less_than(ai, b) for ai in a]
        elif len(b) > 0 and isinstance(b[0], list):
            # b is list of lists, a is flat list
            return [less_than(a, bi) for bi in b]
        else:
            # Both are flat lists
            return [ai < bi for ai, bi in zip(a, b)]
    else:
        raise TypeError("Unsupported input types for less_than")

def count_items(data_list: Union[List[T], List[List[T]], List[List[List[T]]]]) -> Union[int, List[int], List[List[int]]]:
    if isinstance(data_list[0], List):
        if isinstance(data_list[0][0], List):
            len_list = []
            for inner_list in data_list:
                len_inner_list = []
                for inner_inner_list in inner_list:
                    len_inner_list.append(len(inner_inner_list))
                
                len_list.append(len_inner_list)

            return len_list

        else:
            len_list = []
            for inner_list in data_list:
                len_list.append(len(inner_list))

            return len_list
    else:
        return len(data_list)

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

def sin_half_pi(val) -> Union[int, List[int], List[List[int]]]:
    # sin(pi/2 * val)
    if isinstance(val, List):
        if isinstance(val[0], List):
            output_list_list = []
            for object_idx in range(len(val)):
                object_list = []
                for v in val[object_idx]:
                    object_list.append(int(math.sin((np.pi / 2.) * v)))

                output_list_list.append(object_list)

            return output_list_list
        else:
            output_list = []
            for v in val:
                output_list.append(int(math.sin((np.pi / 2.) * v)))

            return output_list
    else:
        return int(math.sin((np.pi / 2.) * val))

def cos_half_pi(val) -> Union[int, List[int]]:
    # cos(pi/2 * val)
    if isinstance(val, List):
        if isinstance(val[0], List):
            output_list_list = []
            for object_idx in range(len(val)):
                object_list = []
                for v in val[object_idx]:
                    object_list.append(int(math.cos((np.pi / 2.) * v)))

                output_list_list.append(object_list)

            return output_list_list
        else:
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
    elif isinstance(a, List) and isinstance(a[0], List):
        object_sums = []
        for obj_idx in range(len(a)):
            output_sums = []
            for idx in range(len(a[obj_idx])):
                output_sums.append(a[obj_idx][idx] + b)

            object_sums.append(output_sums)

        return object_sums
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
    elif isinstance(a, List) and isinstance(b, List):
        output_subs = []
        for idx in range(len(a)):
            result = a[idx] - b[idx]
            output_subs.append(result)
        return output_subs
    elif isinstance(a, List):
        output_subs = []
        for idx in range(len(a)):
            if isinstance(a[0], List):
                output_lists = []
                for inner_idx in range(len(a[idx])):
                    output_lists.append(a[idx][inner_idx] - b)

                output_subs.append(output_lists)
            else:
                result = a[idx] - b
                output_subs.append(result)
        return output_subs
    elif isinstance(b, List):
        if isinstance(b[0], List):
            object_list = []
            for obj_idx in range(len(b)):
                output_subs = []
                for idx in range(len(b[obj_idx])):
                    result = a - b[obj_idx][idx]
                    output_subs.append(result)

                object_list.append(output_subs)
            return object_list

        else:
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
                if isinstance(b, List):
                    output.append(equal2(a[idx], b[idx]))
                else:
                    output.append(equal2(a[idx], b))
            
            return output
        else:
            return equal2(a, b)
    else:
        return equal1(a, b)


def not_equal(a, b):
    def not_equal1(a: int, b: int) -> bool:
        return a != b

    def not_equal2(a: List[int], b: int) -> List[bool]:
        output = []
        for tmp_a in a:
            output.append(tmp_a != b)

        return output

    if isinstance(a, List):
        if isinstance(a[0], List):
            output = []
            for idx in range(len(a)):
                output.append(not_equal2(a[idx], b[idx]))
            
            return output
        else:
            return not_equal2(a, b)
    else:
        return not_equal1(a, b)

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
                if (isinstance(operations, List) or isinstance(operations, np.ndarray)) and len(operations) > 1:
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
        def get_otherwise(otherwise, elem_idx, obj_elem_idx):
            if isinstance(otherwise, List) or isinstance(otherwise, np.ndarray):
                if isinstance(otherwise[elem_idx], List) or isinstance(otherwise[elem_idx], np.ndarray):
                    return otherwise[elem_idx][obj_elem_idx]
                else:
                    return otherwise[elem_idx]
            else:
                return otherwise

        def get_operations(operations, elem_idx, obj_elem_idx):
            if isinstance(operations, List) or isinstance(operations, np.ndarray):
                if isinstance(operations[elem_idx], List) or isinstance(operations[elem_idx], np.ndarray):
                    return operations[elem_idx][obj_elem_idx]
                else:
                    return operations[elem_idx]
            else:
                return operations

        list_output = []
        num_objects = len(conditions)
        for elem_idx in range(num_objects):

            object_conditions = conditions[elem_idx]

            if isinstance(object_conditions, List) or isinstance(object_conditions, np.ndarray):
                obj_list_vals = []
                for obj_cond_idx, object_cond in enumerate(object_conditions):
                    ops = get_operations(operations, elem_idx, obj_cond_idx)
                    oth = get_otherwise(otherwise, elem_idx, obj_cond_idx)

                    if object_cond:
                        obj_list_vals.append(ops)
                    else:
                        obj_list_vals.append(oth)

                list_output.append(obj_list_vals)
            else:
                ops = get_operations(operations, elem_idx, 0)
                oth = get_otherwise(otherwise, elem_idx, 0)

                if object_conditions:
                    list_output.append(ops)
                else:
                    list_output.append(oth)

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
    if isinstance(conditions[0], List) or isinstance(conditions[0], np.ndarray):
        if len(conditions) > 1:
            # Here we have many conditions.
            return switch_many_lists(conditions, operations, otherwise)
        else:
            # Here there is 1 condition, but it's a list.
            return switch_single_list(conditions[0], operations[0], otherwise)
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
        pixels = [pixel.c for pixel in g.pixels]
        colors = list(set(pixels))
        colors.sort()
        return colors
    else:
        all_colors = []
        for grid in g:
            pixels = [pixel.c for pixel in grid.pixels]
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

def set_difference(
    a: Union[List[int], List[List[int]], List[List[List[int]]], List['Pixel'], List[List['Pixel']], List[List[List['Pixel']]]],
    b: Union[List[int], List[List[int]], List[List[List[int]]], List['Pixel'], List[List['Pixel']], List[List[List['Pixel']]]]
) -> Union[List[int], List[List[int]], List[List[List[int]]], List['Pixel'], List[List['Pixel']], List[List[List['Pixel']]]]:
    """
    Computes the set difference between a and b, supporting nested lists and both int and Pixel types.
    """
    def item_in(item, collection):
        # For int, Pixel, or other types, rely on __eq__ for comparison
        for elem in collection:
            if item == elem:
                return True
        return False

    if isinstance(a, list) and isinstance(b, list):
        if len(a) > 0 and isinstance(a[0], list):
            # a is list of lists
            if len(b) > 0 and isinstance(b[0], list):
                # Both a and b are list of lists
                if len(a) != len(b):
                    raise ValueError("Outer lists must have same length for set_difference")
                if len(a) > 0 and isinstance(a[0][0], list) and isinstance(b[0][0], list):
                    # Both are list of list of lists
                    difference_list = []
                    for ai, bi in zip(a, b):
                        if not (isinstance(ai, list) and isinstance(bi, list)):
                            raise ValueError("Expected list of lists for set_difference")
                        if len(ai) != len(bi):
                            raise ValueError("Inner lists must have same length for set_difference")
                        inner_diff = []
                        for aii, bii in zip(ai, bi):
                            # aii and bii are lists
                            inner_diff.append([item for item in aii if not item_in(item, bii)])
                        difference_list.append(inner_diff)
                    return difference_list
                else:
                    # Both are list of lists
                    return [set_difference(ai, bi) for ai, bi in zip(a, b)]
            else:
                # a is list of lists, b is flat list
                return [set_difference(ai, b) for ai in a]
        elif len(b) > 0 and isinstance(b[0], list):
            # a is flat list, b is list of lists
            return [set_difference(a, bi) for bi in b]
        else:
            # Both are flat lists
            return [item for item in a if not item_in(item, b)]
    else:
        # If a or b is not a list, just return a (should not happen in this DSL)
        return a

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
        if isinstance(a[0], List) and isinstance(b[0], List):
            result = []
            for obj_idx in range(len(a)):
                obj_result = [a[obj_idx][i] or b[obj_idx][i] for i in range(len(a[obj_idx]))]
                result.append(obj_result)
        else:    
            result = [a[i] or b[i] for i in range(len(a))]

        return result
    elif isinstance(a, List):
        # List vs Single: each element OR'd with single boolean
        return [x or b for x in a]
    elif isinstance(b, List):
        # Single vs List: single boolean OR'd with each element
        return [a or x for x in b]
    else:
        # Single vs Single: simple OR
        return a or b

def logical_and(a: Union[bool, List[bool], List[List[bool]]], b: Union[bool, List[bool], List[List[bool]]]) -> Union[bool, List[bool]]:
    if isinstance(a, List) and isinstance(b, List):
        if isinstance(a[0], List) and isinstance(b[0], List):
            return [
                [
                    a[i][j] and b[i][j] for j in range(len(a[i]))
                ] for i in range(len(a))
            ]
        else:
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

def neighbours4(grid: Union[GridObject, List[GridObject]]) -> Union[List[List[Pixel]], List[List[List[Pixel]]]]:

    def single_grid_neighbours4(grid: GridObject) -> List[List[Pixel]]:
        # For each pixel in grid.pixels, find all 4-adjacent pixels in grid.pixels
        result = []
        pixel_set = set((p.x, p.y) for p in grid.pixels)
        coord_to_pixel = {(p.x, p.y): p for p in grid.pixels}
        for p in grid.pixels:
            neighbours = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = p.x + dx, p.y + dy
                if (nx, ny) in pixel_set:
                    neighbours.append(coord_to_pixel[(nx, ny)])
            result.append(neighbours)
        return result
    
    if isinstance(grid, GridObject):
        return single_grid_neighbours4(grid)
    else:
        list_of_lists = []
        for g in grid:
            tmp_neighbours4 = single_grid_neighbours4(g)
            list_of_lists.append(tmp_neighbours4)

        return list_of_lists

def neighbours8(grid: Union[GridObject, List[GridObject]]) -> Union[List[List[Pixel]], List[List[List[Pixel]]]]:

    def single_grid_neighbours8(grid: GridObject) -> List[List[Pixel]]:
        # For each pixel in grid.pixels, find all 4-adjacent pixels in grid.pixels
        result = []
        pixel_set = set((p.x, p.y) for p in grid.pixels)
        coord_to_pixel = {(p.x, p.y): p for p in grid.pixels}
        for p in grid.pixels:
            neighbours = []
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1)]:
                nx, ny = p.x + dx, p.y + dy
                if (nx, ny) in pixel_set:
                    neighbours.append(coord_to_pixel[(nx, ny)])
            result.append(neighbours)
        return result
    
    if isinstance(grid, GridObject):
        return single_grid_neighbours8(grid)
    else:
        list_of_lists = []
        for g in grid:
            tmp_neighbours8 = single_grid_neighbours8(g)
            list_of_lists.append(tmp_neighbours8)

        return list_of_lists

def set_pixels(target_grid: Union[GridObject, List[GridObject]], 
               set_x: Union[DIM, List[DIM], List[List[DIM]]], 
               set_y: Union[DIM, List[DIM], List[List[DIM]]],
               colors: Union[COLOR, List[COLOR], List[List[COLOR]]]) -> Union[GridObject, List[GridObject]]:

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
        broadcasting = True
        if isinstance(set_x, List):
            n = len(set_x)
        elif isinstance(set_y, List):
            n = len(set_y)
        elif isinstance(colors, List):
            n = len(colors)
        else:
            broadcasting = False

        if broadcasting:
            if isinstance(set_x, int) or isinstance(set_x, np.int64):
                set_x = np.ones(n) * set_x

            if isinstance(set_y, int) or isinstance(set_y, np.int64):
                set_y = np.ones(n) * set_y

            if isinstance(colors, int) or isinstance(colors, np.int64):
                colors = np.ones(n) * colors

            # Handle negative indices for set_x and set_y
            min_x = int(np.min(set_x)) if len(set_x) > 0 else 0
            min_y = int(np.min(set_y)) if len(set_y) > 0 else 0

            shift_x = -min_x if min_x < 0 else 0
            shift_y = -min_y if min_y < 0 else 0

            # Adjust max_x and max_y to account for negative indices
            max_x = max(target_grid.width + shift_x, int(np.max(set_x)) + 1 + shift_x)
            max_y = max(target_grid.height + shift_y, int(np.max(set_y)) + 1 + shift_y)
            new_cells = np.zeros((max_y, max_x))

            # Copy the original grid into the new grid at the shifted position
            for y in range(target_grid.height):
                for x in range(target_grid.width):
                    new_cells[y + shift_y][x + shift_x] = target_grid.cells[y][x]

            # Build indices: the set of x, y coords from set_x and set_y
            if isinstance(set_x, (list, np.ndarray)) and isinstance(set_y, (list, np.ndarray)):
                indices = list(range(n))
            elif isinstance(set_x, (list, np.ndarray)):
                indices = list(range(n))
            elif isinstance(set_y, (list, np.ndarray)):
                indices = list(range(n))
            elif isinstance(colors, (list, np.ndarray)):
                indices = list(range(n))
            else:
                indices = [0]

            for idx in indices:
                x_coord = int(set_x[idx]) + shift_x
                y_coord = int(set_y[idx]) + shift_y
                color = colors[idx]
                new_cells[y_coord, x_coord] = color

            return GridObject.from_grid(new_cells, target_grid.ul_x - shift_x, target_grid.ul_y - shift_y)
        else:
            new_grid = copy.deepcopy(target_grid)

            # Find the pixel in new_grid.pixels with .x == set_x and .y == set_y, and set its .c to colors
            for pixel in new_grid.pixels:
                if pixel.x == set_x and pixel.y == set_y:
                    pixel.c = colors
                    break

            return new_grid

    if isinstance(target_grid, List):
        output_grids = []
        for idx, grid in enumerate(target_grid):

            if isinstance(set_x, List):
                x_val = set_x[idx]
            else:
                x_val = set_x

            if isinstance(set_y, List):
                y_val = set_y[idx]
            else:
                y_val = set_y

            if isinstance(colors, List):
                c_val = colors[idx]
            else:
                c_val = colors

            tmp_out = set_single_grid_pixels(grid, x_val, y_val, c_val)
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

def set_color(grid: Union[GridObject, List[GridObject]], c_values: Union[int, List[int], List[List[int]]]) -> Union[GridObject, List[GridObject]]:
    def set_grid_c(grid: GridObject, c_values: Union[int, List[int]]) -> GridObject:
        # Assign to each grid.pixels element's .c attribute the corresponding value in c_values
        new_pixels = []
        for idx, pixel in enumerate(grid.pixels):
            if isinstance(c_values, List) or isinstance(c_values, np.ndarray):
                col = c_values[idx]
            else:
                col = c_values
            new_pixel = Pixel(pixel.x + grid.ul_x, pixel.y + grid.ul_y, col)
            new_pixels.append(new_pixel)
        return GridObject(new_pixels, grid.ul_x, grid.ul_y)
    
    if isinstance(grid, GridObject):
        return set_grid_c(grid, c_values)
    else:
        grid_list = []
        for g_idx, g in enumerate(grid):
            if isinstance(c_values, List):
                tmp_g = set_grid_c(g, c_values[g_idx])
            else:
                tmp_g = set_grid_c(g, c_values)
            grid_list.append(tmp_g)

        return grid_list

def crop(g: Union[GridObject, List[GridObject]], x1, y1, x2, y2) -> GridObject:
    def crop_grid(g, x1, y1, x2, y2):
        new_pixels = []
        for pixel in g.pixels:
            if pixel.x >= x1 and pixel.x < x2:
                if pixel.y >= y1 and pixel.y < y2:
                    adjusted_pixel = Pixel(pixel.x + g.ul_x, pixel.y + g.ul_y, pixel.c)
                    new_pixels.append(adjusted_pixel)

        return GridObject(new_pixels, g.ul_x + x1, g.ul_y + y1)

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
    1,  # identity
    1,  # get_objects
    1,  # get_bg
    1,  # color_set
    2,  # equal
    2,  # not equal
    2,  # less_than
    3,  # switch
    2,  # index
    1,  # unique
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
    2,  # set_color
    3,  # new_grid
    2,  # keep
    2,  # exclude
    2,  # set_difference
    2,  # count_values
    1,  # count_items
    2,  # rebuild_grid
    1,  # neighbours4
    1,  # neighbours8
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
    'get_objects': get_objects,
    'get_bg': get_bg,
    'color_set': colorSet,
    'equal': equal,
    'not_equal': not_equal,
    'less_than': less_than,
    'switch': switch,
    'index': get_index,
    'unique': unique,
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
    'set_color': set_color,
    'new_grid': new_grid,
    'keep': keep,
    'exclude': exclude,
    'set_difference': set_difference,
    'count_values': count_values,
    'count_items': count_items,
    'rebuild_grid': rebuild_grid,
    'neighbours4': neighbours4,
    'neighbours8': neighbours8,
    'del': lambda x: x,       # This is actually a special primitive that is implemented at the program execution level
                              # where state memory management is possible.

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
