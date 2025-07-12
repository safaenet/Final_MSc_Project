def generate_gcode(x_pixel, y_pixel, mm_per_pixel=1.0, drop_x=10, drop_y=10):
    # Convert to mm
    x_mm = x_pixel * mm_per_pixel
    y_mm = y_pixel * mm_per_pixel
    
    commands = []
    commands.append(f"G0 X{x_mm:.2f} Y{y_mm:.2f} ; Move to object")
    commands.append("M3 ; Pick")
    commands.append(f"G0 X{drop_x:.2f} Y{drop_y:.2f} ; Move to drop location")
    commands.append("M5 ; Place")
    return commands