# Best way to do this is to draw the 2D side profile and extrude

import cadquery as cq

inner_diameter = 200.0
outer_diameter = 370.0
inner_radius = inner_diameter / 2
outer_radius = outer_diameter / 2
helmholtz_radius = (inner_radius + outer_radius) / 2

bracket_radius = 0.0
bracket_width = 30.0
bracket_foot_height = 5.0
bracket_arm_width = 5.0
bracket_height = helmholtz_radius
bracket_depth = 40.0

bracket = (

    cq.Workplane("XZ").workplane(offset=bracket_radius)
    #starting in the vertical plane with an offset 

    .moveTo(-bracket_width/2, 0) # ie the bracket is centred around the central axis
    .line(bracket_width, 0)
    .line(0, bracket_foot_height)
    .line(-(bracket_width - bracket_arm_width), 0)
    .line(0, bracket_height - 2*bracket_foot_height)
    .line((bracket_width - bracket_arm_width), 0)
    .line(0, bracket_foot_height)
    .line(-bracket_width, 0)
    .close()

    .extrude(bracket_depth, both=True)
    #.edges().fillet(2)
)

# bracket slit
slit_width = bracket_depth/3
slit_height = bracket_height - 40


# slit isn't working
bracket_with_slit = (
    bracket
    .faces(">X")  # Select the first face along the x-axis (ie. the large flat face of the bracket)
    .workplane()  # Create a sketch plane on it
    .center(-bracket_width/2, bracket_height/2)
    #.rect(slit_width, slit_height)  # Draw a rectangle for the slit
    #.cutThruAll()  # Cut it through the entire part
)

cq.exporters.export(bracket, "bracket.step")
