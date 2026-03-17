## DSL Reference - Primitives and their functionality

**get_objects**: returns a list of GridObject instances representing the objects identified in the grid. Can also return a list of sub-objects if the grid parameter was a list of GridObject instances instead.
  
  Parameters:
  - grid (a variable reference): GridObject instance or List[GridObject]. This is the grid to segment or list of grid objects to further segment into their respective objects.

  Returns:
  - a GridObject instance or list of GridObject instances, depending on the type of the grid parameter.
    
**get_bg**: returns the background part (with foreground objects removed) of a grid.

  Parameters:
  - grid (a variable reference): GridObject instance from which to extract the background.

  Returns:
  - a GridObject instance.
 
