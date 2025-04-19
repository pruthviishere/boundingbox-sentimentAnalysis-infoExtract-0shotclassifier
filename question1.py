def count_boxes_in_grid(boxes, img_width, img_height, n_rows, n_cols):
    """
    Count the number of bounding boxes in each grid cell.
    
    Args:
        boxes: List of tuples, each containing (x_min, y_min, x_max, y_max)
        img_width: Width of the image
        img_height: Height of the image
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        
    Returns:
        A 2D list representing the count of boxes in each grid cell
    """
    # Initialize grid with zeros
    grid = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Calculate width and height of each grid cell
    cell_width = img_width / n_cols
    cell_height = img_height / n_rows
    
    # For each bounding box
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        
        # Calculate center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Determine which grid cell the center belongs to
        grid_col = int(center_x // cell_width)
        grid_row = int(center_y // cell_height)
        
        # Handle edge case where center is exactly on the right/bottom edge
        if grid_col == n_cols:
            grid_col = n_cols - 1
        if grid_row == n_rows:
            grid_row = n_rows - 1
            
        # Increment the count for that grid cell
        grid[grid_row][grid_col] += 1
    
    return grid

# Test case 1
boxes1 = [(10, 20, 50, 60), (110, 120, 150, 160), (210, 220, 250, 260), (310, 320, 350, 360)]
img_width1 = 400
img_height1 = 400
n_rows1 = 4
n_cols1 = 4

result1 = count_boxes_in_grid(boxes1, img_width1, img_height1, n_rows1, n_cols1)
print("Test Case 1 Output:")
for row in result1:
    print(row)

# Test case 2
boxes2 = [(10, 20, 50, 60), (20, 25, 55, 65), (35, 45, 75, 85), (40, 50, 80, 90), 
          (50, 60, 100, 110), (110, 120, 150, 160), (130, 140, 170, 180), 
          (160, 170, 200, 210), (175, 185, 195, 205)]
img_width2 = 200
img_height2 = 200
n_rows2 = 5
n_cols2 = 5

result2 = count_boxes_in_grid(boxes2, img_width2, img_height2, n_rows2, n_cols2)
print("\nTest Case 2 Output:")
for row in result2:
    print(row)