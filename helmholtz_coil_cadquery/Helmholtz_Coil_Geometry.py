import cadquery as cq
import numpy as np

#------- Coil
inner_diameter = 200.0
outer_diameter = 370.0

inner_radius = inner_diameter / 2
outer_radius = outer_diameter / 2
coil_thickness = outer_radius - inner_radius
coil_height = coil_thickness/2
helmholtz_radius = inner_radius

outer = cq.Workplane('XY').circle(outer_radius).extrude(coil_height)
inner = cq.Workplane('XY').circle(inner_radius).extrude(coil_height)
coil = outer.cut(inner)

#------- Plate
plate_outer_overhang = 5.0
plate_inner_overhang = 10.0
plate_thickness = 5.0

bottom_plate = (
    cq.Workplane('XY')
    .circle(outer_radius + plate_outer_overhang)
    .circle(inner_radius - plate_inner_overhang)
    .extrude(-plate_thickness)
)

top_plate = (
    cq.Workplane('XY')
    .workplane(offset=coil_height)
    .circle(outer_radius + plate_outer_overhang)
    .circle(inner_radius - plate_inner_overhang)
    .extrude(plate_thickness)
)

# platforms for the inter-coil bracket

platform_count = 4
platform_depth = 40 # tangent to plate
platform_width = 10 # normal to plate
plate_outer_rad = outer_radius + plate_outer_overhang  # z direction

bottom_plate_with_platforms = bottom_plate

for i in range(platform_count):
    angle = i * (360 / platform_count)

    # Create one platform, centered at the origin
    platform = cq.Workplane("XY").box(2 * platform_width, platform_depth, plate_thickness)

    # Move the platform into position and rotate it
    transformed_platform = platform.translate(
        (plate_outer_rad, 0, -plate_thickness / 2)#  + platform_width / 2
    ).rotate((0, 0, 1), (0, 0, 0), angle)

    # Fuse the platform to the plate
    bottom_plate_with_platforms = bottom_plate_with_platforms.union(transformed_platform)


#-------Spacers

spacer_ring_radius = inner_radius - plate_inner_overhang/2
spacer_radius = 4.0
spacer_count = 12
spacer_overlap = 1  # overlap to ensure that union works correctly

spacers = (
    cq.Workplane('XY')
    .workplane(offset=-spacer_overlap)
    .polarArray(radius=spacer_ring_radius, count=spacer_count, startAngle=0, angle=360)
    .circle(spacer_radius)
    .extrude(coil_height + 2*spacer_overlap)
)

housing_combined = top_plate.union(spacers).union(bottom_plate_with_platforms)

coil_assembly = (
    cq.Assembly()
    .add(coil, name="coil", color=cq.Color(0.8, 0.5, 0.25))  # Copper color
    .add(top_plate, name="top_plate", color=cq.Color("lightgray"))
    .add(bottom_plate_with_platforms, name="bottom_plate", color=cq.Color("lightgray"))
    .add(spacers, name="spacers", color=cq.Color("gray"))
)

#assembly.save('coil_with_housing.step')

loc_coil_2 = cq.Location(
    cq.Vector(0, 0, -helmholtz_radius),  # Translation vector
    cq.Vector(1, 0, 0),                 # Rotation axis (X-axis)
    180                                 # Rotation angle
)

helmholtz_assembly = (
    cq.Assembly()
    .add(coil_assembly, name='Coil1')
    .add(coil_assembly, name='Coil2', loc=loc_coil_2)
)


#===================================================================================
# BRACKET GEOMETRY
#===================================================================================

import cadquery as cq

inner_diameter = 200.0
outer_diameter = 370.0
inner_radius = inner_diameter / 2
outer_radius = outer_diameter / 2
helmholtz_radius = inner_radius
plate_outer_overhang = 5

bracket_radius = outer_radius #+ plate_outer_overhang
bracket_width = 30.0
foot_height = 5.0
arm_width = 5.0
bracket_height = helmholtz_radius - 2*plate_thickness
bracket_depth = 40.0

new_bracket_2D = (
    cq.Workplane('YZ')
    .moveTo(-bracket_width/2, 0)
    .line(bracket_width, 0)
    .line(0, foot_height)
    .line(-(bracket_width - arm_width), 0)
    .line(0, (bracket_height - 2*foot_height))
    .line((bracket_width - arm_width), 0)
    .line(0, foot_height)
    .line(-bracket_width, 0)
    #.line(0, -bracket_height)
    .close()
)

new_bracket = new_bracket_2D.extrude(bracket_depth/2, both = True) ###ie extruding in the y_direction

fillet_radius = 3.0

new_bracket_fillet_corners = new_bracket.edges("|Z and >Y").fillet(fillet_radius)

slit_width = bracket_depth/3
slit_height = bracket_height - 50

new_bracket_w_hole = (
    new_bracket
    .faces('<Y')
    .workplane()
    .moveTo(0, bracket_height/2)
    .sketch()
    .rect(slit_width, slit_height)
    .vertices()
    .fillet(fillet_radius)
    .finalize()
    .cutThruAll()
)

filleted_bracket_rotated = (
    new_bracket_w_hole
    .faces('<Y')
    .edges('|X')
    .fillet(fillet_radius)
)

filleted_bracket = filleted_bracket_rotated.rotate((0,0,0), (0,0,1), 180)

for i in range(platform_count):
    angle = i * (360 / platform_count)

    # Define the location for each bracket
    loc = cq.Location(
        # Position Vector:
        cq.Vector(
           bracket_radius * np.cos(np.radians(angle)), # x position
           bracket_radius * np.sin(np.radians(angle)), # y position
           -(helmholtz_radius-foot_height)                           # z position (centered)
        ),
        # Rotation Axis and Angle:
        cq.Vector(0, 0, 1), # Rotate around the Z-axis
        angle + 90          # Angle to make it face radially outward
    )

    # Add the bracket to the existing assembly
    helmholtz_assembly.add(
        filleted_bracket,
        name=f"bracket_{i+1}",
        loc=loc,
        color=cq.Color("slategray")
    )

#===================================================================================
# SAVE THE FINAL MODEL
#===================================================================================

helmholtz_assembly.save('full_assembly_from_user_code.step')

