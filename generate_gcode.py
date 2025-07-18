def generate_gcode_script_with_z(class_name, objects_coordinations, dest, scale=0.5, safe_z=5, pick_z=0):
    gcode = []
    gcode.append("G0 Z5 ; lift to Safe Z \n")
    for i, (x, y) in enumerate(objects_coordinations):
        mm_x, mm_y = x * scale, y * scale
        gcode.append(f"; {class_name} #{i+1} :")
        gcode.append(f"G0 X{mm_x:.2f} Y{mm_y:.2f} Z{safe_z} ; move above object")
        gcode.append(f"G0 Z{pick_z} ; lower to pick")
        gcode.append("M3 ; pick")
        gcode.append(f"G0 Z{safe_z} ; lift")
        gcode.append(f"G0 X{dest[0]*scale:.2f} Y{dest[1]*scale:.2f} ; move to destination")
        gcode.append(f"G0 Z{pick_z} ; lower to place")
        gcode.append("M5 ; place")
        gcode.append(f"G0 Z{safe_z} ; lift up \n")
    gcode.append("G0 X0 Y0 ; Go to home position")
    return gcode
