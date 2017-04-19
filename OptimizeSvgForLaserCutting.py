import svgpathtools.path
import lasertools

# --------
# Settings
# --------

# Import
filenames = ['groups.svg']
only_import_styled_paths = True

# Processing
auto_merge_paths = True
auto_merge_distance = .25
sheet_size = [900., 360.]  # Tries to fit all the input files on a box with these x and y values in units of mm.
file_margin = 5.  # This is the space in mm that should be left empty between two files.
sheet_margin = 5.  # Space in mm that should be left empty around the rim of the sheet.

# Export
outfilename = 'groups_optimized.svg'
style = 'fill:none;stroke:#ff0000;stroke-width:0.72000003;stroke-linecap:round;stroke-linejoin:round;' + \
        'stroke-miterlimit:10;stroke-opacity:1;stroke-dasharray:none'
# style = 'fill:none;stroke:#0000ff;stroke-width:0.02834646;stroke-linecap:round;stroke-linejoin:round;' + \
#         'stroke-miterlimit:5.5;stroke-opacity:1;stroke-dasharray:none' # Formulor


# ------
# Script
# ------

# Load files
data = lasertools.load_files(filenames, only_import_styled_paths)

# Process the files
for i, filen in enumerate(filenames):
    print('Processing ' + filen + ':')
    if auto_merge_paths:
        print('    Looking for pairs in ' + str(len(data[i][0])) + ' paths...')
        paths, attributes, svg_attributes, iters, numclosed = lasertools.auto_merge_paths(data[i], auto_merge_distance)
        data[i] = (paths, attributes, svg_attributes)
        print('    Found ' + str(iters) + ' pairs. Now ' + str(len(paths)) + ' paths are left.')
        print('    ' + str(numclosed) + ' paths were loops and therefore closed.')

# Pack files to sheets
print('Pack the files on the given sheet size of ' + str(sheet_size[0]) + ' mm x ' + str(sheet_size[1]) + ' mm...')
paths, attributes = lasertools.pack_sheet(data, sheet_size, file_margin, sheet_margin, styles=style)
svg_attributes = data[0][2]
print('Files fit on the sheet.')

# Calculate length of paths
pathlength = 0.
for path in paths:
    pathlength += path.length()

pathlength = lasertools.px2mm(pathlength)
print('Laser has to cut ' + str(pathlength) + ' mm of material.')

# Optimize the path order and start/end points
print('Optimizing path order...')

idlepathlength = lasertools.calculate_idle_pathlengths(paths)
print('Idle path length before optimization is ' + str(sum(idlepathlength)) + ' mm')

paths, attributes = lasertools.optimize_path_order(paths, attributes, auto_merge_distance)

print('Using path ' + attributes[0]['id'] + ' to start with.')

idlepathlength = lasertools.calculate_idle_pathlengths(paths)
print('Idle path length is now ' + str(sum(idlepathlength)) + ' mm')

# Export results to file
print('Exporting to file ' + outfilename + '...')
svgpathtools.wsvg(paths,
                  attributes=attributes,
                  svg_attributes={'id': svg_attributes['id'],
                                  'version': svg_attributes['version'],
                                  'size': list(map(lasertools.mm2px, sheet_size))},
                  filename=outfilename)

# Plot the end result for checks
print('Plotting...')
lasertools.plot_paths(paths, attributes)

print('Done.')
