import cadquery as cq


# Coil Params
inner_diameter = 200.0
outer_diameter = 370.0

inner_radius = inner_diameter / 2
outer_radius = outer_diameter / 2
coil_thickness = outer_radius - inner_radius
coil_height = coil_thickness/2 # guessing an appropriate value given the thickness of 85


# Creating the cross section
cross_section = (
    cq.Workplane('XZ')
    .rect(coil_thickness, coil_height)
    .translate((inner_radius + coil_thickness / 2, 0))
)


outer = cq.Workplane('XY').circle(outer_radius).extrude(coil_height)
inner = cq.Workplane('XY').circle(inner_radius).extrude(coil_height)
coil2 = outer.cut(inner)

cq.exporters.export(coil2, 'test.step')

