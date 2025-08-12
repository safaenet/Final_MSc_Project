def is_overlap(box1, box2, threshold=0.2):
    # Each box is given as (center_x, center_y, size)
    x1, y1, size1 = box1
    x2, y2, size2 = box2

    # Convert both boxes from center coordinates to top-left coordinates
    x1, y1 = x1 - size1 // 2, y1 - size1 // 2
    x2, y2 = x2 - size2 // 2, y2 - size2 // 2

    # Find the coordinates of the intersection rectangle
    xA = max(x1, x2)  # Left edge of overlap
    yA = max(y1, y2)  # Top edge of overlap
    xB = min(x1 + size1, x2 + size2)  # Right edge of overlap
    yB = min(y1 + size1, y2 + size2)  # Bottom edge of overlap

    # Calculate intersection area
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the area of each box
    boxAArea = size1 * size1
    boxBArea = size2 * size2

    # Compute Intersection over Union (IoU)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return True if IoU is greater than the threshold, meaning boxes overlap
    return iou > threshold
