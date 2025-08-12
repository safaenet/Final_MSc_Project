def generate_gcode_script_with_z(class_name, objects_coordinations, dest, scale=0.5, safe_z=5, pick_z=0):
    # Create a list to store all G-code commands
    gcode = []

    # First command: lift the tool to a safe Z height
    gcode.append("G0 Z5 ; lift to Safe Z \n")

    # Loop through each detected object and create G-code steps
    for i, (x, y) in enumerate(objects_coordinations):
        # Convert pixel coordinates to millimeters using scale
        mm_x, mm_y = x * scale, y * scale

        # Comment line showing which object this is
        gcode.append(f"; {class_name} #{i+1} :")

        # Move above the object at safe Z height
        gcode.append(f"G0 X{mm_x:.2f} Y{mm_y:.2f} Z{safe_z} ; move above object")

        # Lower down to picking height
        gcode.append(f"G0 Z{pick_z} ; lower to pick")

        # Activate the gripper or tool to pick the object
        gcode.append("M3 ; pick")

        # Lift the object back to safe Z height
        gcode.append(f"G0 Z{safe_z} ; lift")

        # Move to the destination point
        gcode.append(f"G0 X{dest[0]*scale:.2f} Y{dest[1]*scale:.2f} ; move to destination")

        # Lower down to place height
        gcode.append(f"G0 Z{pick_z} ; lower to place")

        # Release the object
        gcode.append("M5 ; place")

        # Lift up again to safe Z height
        gcode.append(f"G0 Z{safe_z} ; lift up \n")

    # After all objects, move the tool back to the home position
    gcode.append("G0 X0 Y0 ; Go to home position")

    # Return the full list of G-code commands
    return gcode
