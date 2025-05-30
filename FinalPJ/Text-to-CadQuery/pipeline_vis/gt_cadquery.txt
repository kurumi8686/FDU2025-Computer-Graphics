import cadquery as cq
import math
from cadquery import exporters

# --- Part 1: Rectangular Prism ---
part_1_length = 0.1149
part_1_width = 0.0718
part_1_height = 0.0431

part_1 = (
    cq.Workplane("XY")
    .rect(part_1_length, part_1_width)
    .extrude(part_1_height)
)

# Coordinate System Transformation for Part 1
part_1 = part_1.rotate((0, 0, 0), (0, 0, 1), -90)
part_1 = part_1.translate((0.1774, 0.0431, 0.253))

# --- Part 2: Small Rectangular Prism ---
part_2_length = 0.2011
part_2_width = 0.0172
part_2_height = 0.0057

part_2 = (
    cq.Workplane("XY")
    .rect(part_2_length, part_2_width)
    .extrude(part_2_height)
)

# Coordinate System Transformation for Part 2
part_2 = part_2.rotate((0, 0, 0), (0, 0, 1), -90)
part_2 = part_2.translate((0.5489, 0.0431, 0.2925))

# --- Part 3: Curved Shape ---
part_3_scale = 0.4841
part_3_height = 0.0345

part_3 = (
    cq.Workplane("XY")
    .moveTo(0, 0)
    .threePointArc((0.2384, 0.0667), (0.4841, 0.0957))
    .lineTo(0.4841, 0.1152)
    .threePointArc((0.2376, 0.0882), (0, 0.0172))
    .lineTo(0, 0)
    .close()
    .extrude(part_3_height)
)

# Coordinate System Transformation for Part 3
part_3 = part_3.rotate((0, 0, 0), (0, 0, 1), -90)
part_3 = part_3.translate((0, 0.0431, 0))

# --- Assembly ---
assembly = part_1.union(part_2).union(part_3)

# --- Export to STL ---
exporters.export(assembly, './stlcq/0010/00102156.stl')