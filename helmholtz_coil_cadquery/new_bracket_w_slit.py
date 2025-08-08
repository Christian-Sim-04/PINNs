import cadquery as cq

inner_diameter = 200.0
outer_diameter = 370.0
inner_radius = inner_diameter / 2
outer_radius = outer_diameter / 2
helmholtz_radius = inner_radius
plate_outer_overhang = 5

bracket_radius = outer_radius + plate_outer_overhang
bracket_width = 30.0
foot_height = 5.0
arm_width = 5.0
bracket_height = helmholtz_radius
bracket_depth = 40.0

new_bracket_2D = (
    cq.Workplane('XZ')
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

new_bracket = new_bracket_2D.extrude(bracket_depth, both = True) ###ie extruding in the y_direction

fillet_radius = 3

new_bracket_fillet_corners = new_bracket.edges("|Z and >X").fillet(fillet_radius)

slit_width = bracket_depth/2
slit_height = bracket_height - 40

new_bracket_w_hole = (
    new_bracket
    .faces('<X')
    .workplane()
    .moveTo(0, bracket_height/2)
    .sketch()
    .rect(slit_width, slit_height)
    .vertices()
    .fillet(fillet_radius)
    .finalize()
    .cutThruAll()
)

filleted_bracket = (
    new_bracket_w_hole
    .faces('<X')
    .edges('|Y')
    .fillet(fillet_radius)
)

#.moveTo(-slit_width/2, bracket_height/2 - slit_height/2)
 #   .line(slit_width, 0)
  #  .line(0, slit_height)
   # .line(-slit_width, 0)
    #.close()

cq.exporters.export(filleted_bracket, 'filleted_bracket.step')