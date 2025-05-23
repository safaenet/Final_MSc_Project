def is_overlap(box1, box2, threshold=0.2):
    # boxes = (x, y, size)
    x1, y1, size1 = box1
    x2, y2, size2 = box2

    # convert to top-left
    x1, y1 = x1 - size1//2, y1 - size1//2
    x2, y2 = x2 - size2//2, y2 - size2//2

    # intersection
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1+size1, x2+size2)
    yB = min(y1+size1, y2+size2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = size1 * size1
    boxBArea = size2 * size2
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou > threshold
