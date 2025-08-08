import cadquery as cq


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
platform_depth = 40 #ie y_direction
platform_width = 10 # x direction
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

coil_assembly.save('coil_with_housing.step')

#loc_coil_2 = cq.Location(
#    cq.Vector(0, 0, -helmholtz_radius),  # Translation vector
#    cq.Vector(1, 0, 0),                 # Rotation axis (X-axis)
#    180                                 # Rotation angle
#)

#helmholtz_assembly = (
#    cq.Assembly()
#    .add(coil_assembly, name='Coil1')
#    .add(coil_assembly, name='Coil2', loc=loc_coil_2)
#)

#helmholtz_assembly.save('double_coil_with_housing.step')