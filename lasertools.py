import svgpathtools
from svgpathtools.path import Path, bpoints2bezier
import os.path
import tempfile
import subprocess
import shutil
import numpy as np
import operator
import scipy.optimize
import matplotlib
import matplotlib.path
import matplotlib.patches
import matplotlib.pyplot as plt
import itertools

INKSCAPE_EXECUTABLE = r'C:\Program Files (x86)\Inkscape\inkscape.exe'
SVG_DPI = 90.


# MAX_CUTTING_SPEED = 34.4 # mm/s
# ACCELERATION = 150.      # mm/s²


def load_files(filenames, only_import_styled_paths=True):
    """
    Loads the given filenames and extracts metadata and paths from it.
    :param filenames: List of filenames to load. Valid file formats are: SVG or PDF. If a PDF is given, it will first be
    converted to an SVG using inkscape. The global variable INKSCAPE_EXECUTABLE has to be defined and point to inkscape.
    :param only_import_styled_paths: This paramter determines how to treat all paths that do not have a style attribute.
    The default behaviour is to delete them. This helps to get rid of clippingPaths and similar things.
    :return: List of tuples. For each file a tuple is returned, containing paths, attributes and svg_attributes.
    """

    ifilenames = list(filenames)

    # Check which files to convert
    first = True
    tdir = None
    try:
        for i, infile in enumerate(ifilenames):
            # Convert PDFs
            if infile[-4:] == '.pdf':
                if first:
                    # Create temporary directory
                    tdir = tempfile.mkdtemp()
                tfile = os.path.join(tdir, os.path.basename(infile[:-4] + '.svg'))
                # Convert using inkscape
                subprocess.check_output([INKSCAPE_EXECUTABLE, '--export-plain-svg', tfile, infile])
                ifilenames[i] = tfile

        # Parse SVG data
        ret = [svgpathtools.svg2paths(infile, convert_polylines_to_paths=False, return_svg_attributes=True)
               for infile in ifilenames]

        # Delete temporary directory
        if tdir is not None:
            shutil.rmtree(tdir)
            del tdir

        # Remove unwanted paths like workspace etc.
        for i in range(len(ret)):
            new_data = [[path, ret[i][1][j]]
                        if (len(path) > 0 and 'style' in ret[i][1][j].keys() or (not only_import_styled_paths)) else []
                        for j, path in enumerate(ret[i][0])]
            new_data = [item for sublist in new_data for item in sublist]
            ret[i] = (new_data[::2], new_data[1::2], ret[i][2])
    except:
        if tdir is not None:
            shutil.rmtree(tdir)
            del tdir
        raise

    return ret


def auto_merge_paths(data, auto_merge_distance, auto_close_paths=True):
    """
    This function connects all paths in the given dataset, for which the start or endpoints are closer than 
    auto_merge_distance.
    :param data: Should be a list or tuple containing paths, attributes, svg_attributes.
    :param auto_merge_distance: If the start or endpoint of a pair of paths is closer than this distance in units of 
    milli meters, they are automatically merged. If one of the paths has to be reversed to do so, this is automatically
    done. A line is added to the path to bridge the gap.
    :param auto_close_paths: If set the paths are automatically closed after the merging operation if the start and
    end point of one path are closer than the auto_merge_distance. It is closed by a line and it's closed flag is set.
    :return paths, attributes, svg_attributes, iters, numclosed: Modified paths, modified attributes, svg_attributes, 
    number of pairs connected and number of paths that were closed.
    """
    paths, attributes, svg_attributes = data

    def fix_first_pair(paths_, attributes_):
        """
        Helper function that fixes the next best pair of paths, if they fulfill the condition
        :rtype: NoneType in case paths_ is empty. Else fixed paths_ and attributes_.
        """
        for i_ in range(len(paths_)):
            # Get start end end points
            start1 = paths_[i_][0].start
            end1 = paths_[i_][-1].end

            for j in range(len(paths_)):
                if i_ != j:
                    start2 = paths_[j][0].start
                    end2 = paths_[j][-1].end

                    # Calculate all relevant distances for this pair
                    distance_ = px2mm(np.abs(start2 - end1))
                    distance_r1 = px2mm(np.abs(start2 - start1))
                    distance_r2 = px2mm(np.abs(end2 - end1))

                    # Perform merger
                    if distance_ < auto_merge_distance or distance_r2 < auto_merge_distance:
                        first = i_
                        second = j
                    else:
                        first = j
                        second = i_

                    if distance_r1 < auto_merge_distance or distance_r2 < auto_merge_distance:
                        # Reverse paths_[j] if necessary
                        paths_[j] = svgpathtools.path.Path(
                            *[svgpathtools.path.bpoints2bezier(segment.bpoints()[::-1]) for segment in paths_[j]])

                    if min([distance_, distance_r1, distance_r2]) < auto_merge_distance:
                        # Merge both paths
                        paths_[first] = svgpathtools.path.Path(*[segment for segment in paths_[first]] + [
                            svgpathtools.path.Line(paths_[first][-1].end, paths_[second][0].start)] +
                                                                [segment for segment in paths_[second]])
                        return paths_[:second] + paths_[second + 1:], attributes_[:second] + attributes_[second + 1:]
        return None

    iters = 0
    while True:
        ret = fix_first_pair(paths, attributes)
        if ret is not None:
            paths, attributes = ret
            iters += 1
        else:
            break

    # Make sure, paths are closed...
    numclosed = 0
    if auto_close_paths:
        for i, path in enumerate(paths):
            # Get start end end point distance
            start = path[0].start
            end = path[-1].end
            distance = px2mm(np.abs(start - end))

            if distance < auto_merge_distance:
                # Close the path
                paths[i] = svgpathtools.path.Path(*[segment for segment in path] + [svgpathtools.path.Line(end, start)])
                paths[i].closed = True
                numclosed += 1
    return paths, attributes, svg_attributes, iters, numclosed


def pack_sheet(data, sheet_size, file_margin, sheet_margin, styles=None):
    """
    Tries to pack paths from multiple files on one sheet with given dimension. Uses a slightly modified skyline
    algorithm for packing the files into the bin. The modification makes it favor positions closer to (0.,0.) if there
    is more than one favored solution.
    :param data: Paths, attributes of the individual files to pack.
    :param sheet_size: List of x and y size of sheet in units mm.
    :param file_margin: Margin around the extends of the paths in the input files to keep clear. Unit is mm.
    :param sheet_margin: Margin around the rim of the sheet to keep clear. Unit is mm.
    :param styles: New style attribute for paths to apply. If it is a list of strings, it is expected to have the same 
    length as data. In this case the style is applied to the data from the respective file.
    :return: 
    """

    # Helper functions
    def get_bboxes(paths__):
        for i_ in range(len(paths__)):
            bboxes_ = np.array([np.array(path.bbox()) for path in paths__[i_]])
            bbox = [np.min(bboxes_[:, 0]), np.max(bboxes_[:, 1]), np.min(bboxes_[:, 2]), np.max(bboxes_[:, 3])]
            yield bbox

    paths = [item[0] for item in data]
    bboxes = np.array(list(get_bboxes(paths)))

    # Initiate SKYLINE
    bin_size = [mm2px(sheet_size[0] - 2. * sheet_margin),
                mm2px(sheet_size[1] - 2. * sheet_margin)]
    box_sizes = np.array([bboxes[:, 1] - bboxes[:, 0] + mm2px(file_margin),
                          bboxes[:, 3] - bboxes[:, 2] + mm2px(file_margin)]).T
    box_positions = np.zeros((box_sizes.shape[0], 2))
    box_positions[:, :] = np.nan

    # Sort boxes
    order = list(map(int, np.array(sorted([[idx, np.max(size)] for idx, size in enumerate(box_sizes)],
                                          key=lambda item: item[1], reverse=True))[:, 0]))
    box_sizes[:, :] = box_sizes[order, :]

    # SKYLINE helpers
    def skyline_area(skyline_):
        area = 0.
        for i_ in range((skyline_.shape[0] - 1) // 2):
            if np.isnan(skyline_[i_ * 2 + 1, 0]):
                break
            else:
                area += (skyline_[i_ * 2 + 1, 0] - skyline_[i_ * 2, 0]) * skyline_[i_ * 2, 1]
        return area

    def box_area(box_positions__, box_sizes__):
        boxarea = 0.
        for i_ in range(box_positions__.shape[0]):
            if np.isnan(box_positions__[i_, 0]):
                break
            else:
                boxarea += box_sizes__[i_, 0] * box_sizes__[i_, 1]
        return boxarea

    def update_skyline(box_positions__, box_sizes__):
        # print('    Updating skyline.')
        # print('        Positions:')
        # print(box_positions)
        # print('        Sizes:')
        # print(box_sizes)

        # Get all relevant X-coordinates
        xs = []
        for i_ in range(box_positions__.shape[0]):
            if np.isnan(box_positions__[i_, 0]):
                break
            else:
                xs += [box_positions__[i_, 0], box_positions__[i_, 0] + box_sizes__[i_, 0]]
        # Make distinct and sort
        xs = list(set(xs))
        xs = sorted(xs)
        # print('Xs: ' + str(xs))

        # Get upper Y-coordinates for each x
        yxs = []
        for x_ in xs:
            yx = []
            for i_ in range(box_positions__.shape[0]):
                if np.isnan(box_positions__[i_, 0]):
                    break
                else:
                    if box_positions__[i_, 0] <= x_ <= box_positions__[i_, 0] + box_sizes__[i_, 0]:
                        yx += [[i_, box_positions__[i_, 1] + box_sizes__[i_, 1]]]
            yx = sorted(yx, key=operator.itemgetter(1))
            yxs += [yx]

        # Determine skyline
        skyline_xs = [0.]
        skyline_ys = [yxs[0][-1][1]]
        active_box = yxs[0][-1][0]
        for i_, x_ in enumerate(xs[1:]):
            # print('Checking x = ' + str(x) + ' Active = ' + str(active_box))
            if yxs[i_ + 1][-1][1] > skyline_ys[-1]:
                # print('Rising Edge!')
                # Found rising edge!
                skyline_xs += [x_, x_]
                skyline_ys += [skyline_ys[-1], yxs[i_ + 1][-1][1]]
                active_box = yxs[i_ + 1][-1][0]
                # print('New active = ' + str(active_box))
            # Falling edges or constant case can only occur on the end of a box
            elif x_ == box_positions__[active_box, 0] + box_sizes__[active_box, 0]:
                # print('End of box!')
                # Rising edge is already taken care of in last conditional; look for falling edge or constant case.
                if yxs[i_ + 1][-1][0] == active_box:
                    if len(yxs[i_ + 1]) > 1:
                        # print('Active box on top!')
                        if yxs[i_ + 1][-1][1] == yxs[i_ + 1][-2][1]:
                            # print('Constant case!')
                            # Constant case. Just change active_box:
                            active_box = yxs[i_ + 1][-2][0]
                            # print('New active = ' + str(active_box))
                        elif yxs[i_ + 1][-1][1] > yxs[i_ + 1][-2][1]:
                            # print('Falling edge!')
                            # Falling edge found!
                            skyline_xs += [x_, x_]
                            skyline_ys += [skyline_ys[-1], yxs[i_ + 1][-2][1]]
                            active_box = yxs[i_ + 1][-2][0]
                            # print('New active = ' + str(active_box))
                elif skyline_ys[-1] == yxs[i_ + 1][-1][0]:
                    # Constant case. Just change active_box:
                    # print('Constant case!')
                    active_box = yxs[i_ + 1][-1][0]
                    # print('New active = ' + str(active_box))
        # print('End.')
        # Complete it
        skyline_xs += [xs[-1], xs[-1]]
        skyline_ys += [skyline_ys[-1], 0.]

        return np.array(list(zip(skyline_xs, skyline_ys)))

    def find_best_position(bin_size_, skyline_, box_positions__, box_sizes__, new_box_idx):
        """
        This function checks all positions along the skyline and tries to insert the box.
        It has to check collisions with the skyline and the sheet boundary. Box can be moved up, until it is outside of
        the skyline. For collision with the skyline try to tilt it. Use the least wasteful method. If there is multiple
        options, use the one that puts the center closest to (0.,0.).
        :param bin_size_: 
        :param skyline_: 
        :param box_positions__: 
        :param box_sizes__: 
        :param new_box_idx:
        :return: 
        """
        box_sizes__ = np.copy(box_sizes__)
        box_size = box_sizes__[new_box_idx]
        side_x = box_size[0]
        side_y = box_size[1]

        potentials_ = []
        for i_ in range(skyline_.shape[0]):
            # print('Checking ' + str(i) + '...')
            # print('    Skyline coordinates: ' + str(skyline[i, :]) + '.')
            # print('    Height reaching to ' + str(skyline[i, 1] + side_y) + '.')
            # print('    Sheet size ' + str(bin_size[1]) + '.')
            test_pos_x = skyline_[i_, 0]
            test_pos_y = skyline_[i_, 1]
            while True:
                # Check if the height fits the sheet
                if test_pos_y + side_y < bin_size_[1]:
                    # Check if there is enough space to the right
                    right_border = bin_size_[0]
                    right_border_y = -1
                    for j_ in range(i_ + 1, skyline_.shape[0]):
                        if skyline_[j_, 1] > test_pos_y:
                            right_border = skyline_[j_, 0]
                            right_border_y = skyline_[j_, 1]
                            break
                    if test_pos_x + side_x < right_border:
                        # print('    Box can potentially be placed at ' + str(test_pos_x) + ',' + str(test_pos_y) + '.')
                        box_positions__[new_box_idx, :] = [test_pos_x, test_pos_y]
                        test_skyline = update_skyline(box_positions__, box_sizes__)
                        wasted_ = np.round(skyline_area(test_skyline) - box_area(box_positions__, box_sizes__),
                                           decimals=5)
                        space_left_top = bin_size_[1] - (test_pos_y + side_y)
                        space_left_right = right_border - side_x
                        # print('    Wasted area would then be ' + str(wasted) + '. Space left is top: ' +
                        #       str(space_left_top) + '; right: ' + str(space_left_right) + '.')
                        potentials_ += [[test_pos_x, test_pos_y, wasted_, space_left_top, space_left_right]]
                        break
                    elif right_border < bin_size_[0]:
                        # print('    Not enough space to the right at ' + str(test_pos_x) + ', ' + str(test_pos_y) +
                        #     '. Trying to move up to ' + str(right_border_y) + '.')
                        test_pos_y = right_border_y
                    else:
                        # print('    Not enough space to the right at ' + str(test_pos_x) + ', ' + str(test_pos_y) +
                        #     '. Hit boundary.')
                        break
                else:
                    # print('    Not enough space to the top at '+str(test_pos_x)+', '+str(test_pos_y)+'.')
                    break

        # Choose best position
        if len(potentials_) > 0:
            potentials_ = sorted(potentials_, key=lambda item: item[3])
            potentials_ = sorted(potentials_, key=lambda item: item[2])
            # print('        => DONE! Placing at ' + str(potentials[0][0]) + ', ' + str(potentials[0][1]))
            return potentials_[0][0], potentials_[0][1]
        else:
            # print('        => No space left over to put the file!')
            return None

    # Place boxes
    potentials = []

    # Try all possible rotation combinations
    n_boxes = box_sizes.shape[0]
    n_combinations = 2 ** n_boxes

    for i in range(n_combinations):
        box_positions_ = box_positions.copy()
        skyline = np.zeros((1, 2))

        # Rotate boxes
        rotations = np.zeros((box_sizes.shape[0],), dtype=bool)
        if i != 0:
            rots = list([bool(int(digit)) for digit in bin(i)[2:]])
            rotations[:len(rots)] = rots[::-1]
        box_sizes_ = box_sizes.copy()
        for j in range(n_boxes):
            if rotations[j]:
                box_sizes_[j, :] = box_sizes_[j, ::-1]

        # print('Running placements:')
        for j in range(n_boxes):
            # print('    Placing box ' + str(j) + ' with size ' + str(box_sizes_[j]) + ':')
            pos = find_best_position(bin_size, skyline, box_positions_, box_sizes_, j)
            if pos is not None:
                x, y = pos
                box_positions_[j, :] = [x, y]
                skyline = update_skyline(box_positions_, box_sizes_)
                # print('    New skyline: ' + str(skyline))

                if j == n_boxes - 1:
                    wasted = np.round(skyline_area(skyline) - box_area(box_positions_, box_sizes_), decimals=5)
                    # length = np.round(skyline_length(skyline), decimals=5)
                    centers = box_positions_[:, :] + box_sizes_[:, :] / 2.
                    compactness = np.sum(np.sqrt(centers[:, 0] ** 2 + centers[:, 1] ** 2))
                    potential = [wasted, compactness, box_positions_, rotations.copy(), skyline.copy()]
                    already_known = False
                    for subl in potentials:
                        already_known = np.all([np.all(potential[k] == item) for k, item in enumerate(subl)])
                        if already_known:
                            break
                    if not already_known:
                        potentials += [potential]
            else:
                break

    potentials = sorted(potentials, key=lambda item: item[1])
    potentials = sorted(potentials, key=lambda item: item[0])

    # Use best combination
    if len(potentials) > 1:
        # box_positions = potentials[0][2]
        for i, rot in enumerate(potentials[0][3]):
            if rot:
                box_sizes[i, :] = box_sizes[i, ::-1]

        # skyline = potentials[0][4]
        #
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot([0.,
        #          0.,
        #          bin_size[0],
        #          bin_size[0],
        #          0.],
        #         [0.,
        #          bin_size[1],
        #          bin_size[1],
        #          0.,
        #          0.], label='BIN')
        # for i in range(box_sizes.shape[0]):
        #     ax.plot([box_positions[i, 0],
        #              box_positions[i, 0],
        #              box_positions[i, 0] + box_sizes[i, 0],
        #              box_positions[i, 0] + box_sizes[i, 0],
        #              box_positions[i, 0], ],
        #             [box_positions[i, 1],
        #              box_positions[i, 1] + box_sizes[i, 1],
        #              box_positions[i, 1] + box_sizes[i, 1],
        #              box_positions[i, 1],
        #              box_positions[i, 1]], label='box ' + str(i))
        # ax.plot(skyline[:, 0], skyline[:, 1], label='SKYLINE')
        # plt.legend()
        # plt.axis('equal')
        # ax.invert_yaxis()
        # plt.show()

        # Transform paths accordingly
        paths_ = []
        attributes = []
        for i, j in enumerate(order):
            rot = int(potentials[0][3][i])
            irot = int(not potentials[0][3][i])
            rotcorrection = (bboxes[j, 3] + bboxes[j, 2]) if rot else 0.
            trans_x = potentials[0][2][i][0] + (mm2px(sheet_margin) - mm2px(file_margin) / 2.) + \
                rotcorrection + mm2px(file_margin) / 2. - bboxes[j, rot * 2]
            trans_y = potentials[0][2][i][1] + (mm2px(sheet_margin) - mm2px(file_margin) / 2.) + \
                mm2px(file_margin) / 2. - bboxes[j, irot * 2]
            am = np.array([irot, -rot, trans_x, rot, irot, trans_y, 0., 0., 1.]).reshape((3, 3))

            # Apply transformation to all elements of the paths
            def xy(p):
                return np.array([p.real, p.imag, 1.])

            def z(coords):
                return coords[0] + 1j * coords[1]

            paths_ += [Path(*[bpoints2bezier([z(np.dot(am, xy(pt))) for pt in seg.bpoints()]) for seg in path])
                       for path in paths[j]]

            attributes_ = data[j][1]
            if styles is not None:
                if type(styles) is str:
                    style = styles
                elif len(styles) != len(data):
                    style = styles[0]
                else:
                    style = styles[j]
                for k in range(len(attributes_)):
                    attributes_[k]['style'] = style
            attributes += attributes_

        return paths_, attributes


def px2mm(px):
    """
    Conversion of pixels to milli meters. Needs the global variable SVG_DPI to be configured correctly. 
    :param px: Length in pixel which is supposed to be converted into mm.
    :return: Length in mm.
    """
    return px / SVG_DPI * 25.40


def mm2px(mm):
    """
    Conversion of milli meters to pixels. Needs the global variable SVG_DPI to be configured correctly. 
    :param mm: Length in mm which is supposed to be converted into pixel.
    :return: Length in pixel.
    """
    return mm * SVG_DPI / 25.40


def calculate_idle_pathlengths(paths):
    idlepathlengths = np.zeros((len(paths) + 1,))

    # Calculate idle path length before optimization
    idlepathlengths[0] = px2mm(abs((0. + 1j * 0.) - paths[0].start))

    for i in range(len(paths) - 1):
        idlepathlengths[i + 1] = px2mm(abs(paths[i + 1].start - paths[i].end))

    idlepathlengths[-1] = px2mm(abs(paths[-1].end - (0. + 1j * 0.)))
    return list(idlepathlengths)


def optimize_path_order(paths, attributes, auto_merge_distance):
    # Define helper functions
    def get_shortest_distance_to_path(x1, y1, path2):
        i_ = mm2px(x1) + 1j * mm2px(y1)

        def _distance(t):
            while t > 1.:
                t = t - 1.
            while t < 0.:
                t = t + 1.
            return abs(path2.point(t) - i_)

        if path2.isclosed():
            # Get first estimate by sampling
            ts = np.arange(0., 1., .001)
            ys = list(map(_distance, ts))
            min_idx = np.argmin(ys)
            min_t = scipy.optimize.minimize(lambda ts_: _distance(ts_[0]),
                                            np.array([ts[min_idx]]),
                                            method='Nelder-Mead').x
            return _distance(min_t[0]), min_t[0]
        else:
            startdistance = _distance(0.)
            enddistance = _distance(1.)

            if startdistance <= enddistance:
                return startdistance, 0.
            else:
                return enddistance, 1.

    def change_start_in_path(path, t):
        # Treat special cases:
        if t == 0.:
            return path
        if t == 1.:
            return svgpathtools.path.Path(
                *[svgpathtools.path.bpoints2bezier(segment.bpoints()[::-1]) for segment in path])

        # Identify segment first
        lengths = [seg.length() for seg in path]
        pos_along_path = sum(lengths) * t
        for i_ in range(len(lengths)):
            if px2mm(abs(pos_along_path - lengths[i_])) < auto_merge_distance:
                # Split path at segment
                return svgpathtools.path.Path(*(path[i_ + 1:] + path[:i_ + 1]))
            if pos_along_path > lengths[i_]:
                pos_along_path -= lengths[i_]
            else:
                seg_t = pos_along_path / lengths[i_]
                if type(path[i_]) == svgpathtools.path.Line:
                    start = path[i_].start
                    sep_point = path[i_].point(seg_t)
                    end = path[i_].end

                    return svgpathtools.path.Path(*(
                        [svgpathtools.path.bpoints2bezier([sep_point, end])] + path[i_ + 1:] + path[:i_] + [
                            svgpathtools.path.bpoints2bezier([start, sep_point])]))

                elif type(path[i_]) == svgpathtools.path.CubicBezier:
                    points = np.zeros((4, 4), dtype=np.complex)
                    points[:, 0] = path[i_].bpoints()
                    for j in range(1, 4):
                        for k in range(4 - j):
                            points[k, j] = (1. - seg_t) * points[k, j - 1] + seg_t * points[k + 1, j - 1]

                    return svgpathtools.path.Path(*(
                        [svgpathtools.path.bpoints2bezier([points[i_, 4 - 1 - i_] for i_ in range(4)])] +
                        path[i_ + 1:] + path[:i_] + [svgpathtools.path.bpoints2bezier(points[0, :])]))

                else:
                    print('Trying to cut ' + str(path[i_]) + ' at internal t of ' + str(seg_t) + '!')
                    raise NotImplementedError

    # Start with Neares Neighbour Optimization
    # Find starting point
    startidx = 0
    shortest_distance, shortest_tval = get_shortest_distance_to_path(0., 0., paths[0])
    for i in range(1, len(paths)):
        distance, tval = get_shortest_distance_to_path(0., 0., paths[i])
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_tval = tval
            startidx = i

    paths = paths[startidx:] + paths[:startidx]
    attributes = attributes[startidx:] + attributes[:startidx]

    paths[0] = change_start_in_path(paths[0], shortest_tval)

    for i in range(len(paths) - 1):
        endx = px2mm(paths[i][-1].end.real)
        endy = px2mm(paths[i][-1].end.imag)

        # Find next best path
        distances = [[j, get_shortest_distance_to_path(endx, endy, paths[j])] for j in range(i + 1, len(paths))]
        distances = sorted(distances, key=lambda item: item[1][0])
        order = [item[0] for item in distances]
        paths[i + 1:] = [paths[j] for j in order]
        paths[i + 1] = change_start_in_path(paths[i + 1], distances[0][1][1])
        attributes[i + 1:] = [attributes[j] for j in order]
        # print(distances[0])
        # print(attributes[i+1]['id'])
        # print(getDistanceToPathStart(endx,endy,paths[i+1]))
        # print(lasertools.px2mm(distances[0][1][0]))

    return paths, attributes


def plot_paths(paths, attributes):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_it = itertools.cycle(colors)

    lhandles = [plt.Line2D((0, 1), (0, 0), lw=1, color='black', ls='--')]
    llabels = ['idle']

    for i, path in enumerate(paths):
        verts = []
        codes = []
        for j, segment in enumerate(path):
            if type(segment) == svgpathtools.path.Line:
                verts += [[px2mm(segment.start.real), px2mm(segment.start.imag)],
                          [px2mm(segment.end.real), px2mm(segment.end.imag)]]
                codes += [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO]

            elif type(segment) == svgpathtools.path.CubicBezier:
                verts += [[px2mm(segment.start.real), px2mm(segment.start.imag)],
                          [px2mm(segment.control1.real), px2mm(segment.control1.imag)],
                          [px2mm(segment.control2.real), px2mm(segment.control2.imag)],
                          [px2mm(segment.end.real), px2mm(segment.end.imag)]]
                codes += [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE4, matplotlib.path.Path.CURVE4,
                          matplotlib.path.Path.CURVE4]

            else:
                print('Problem @ Segment: ' + str(i) + '.' + str(j))
                print(segment)
                raise NotImplementedError

        if len(verts) > 0:
            color = next(colors_it)
            patch = matplotlib.patches.PathPatch(matplotlib.path.Path(np.array(verts), codes), ec=color, fill=False,
                                                 lw=2)
            ax.add_patch(patch)
            lhandles.append(plt.Line2D((0, 1), (0, 0), lw=2, color=color))
            llabels.append(str(attributes[i]['id']))

            # Plot idle path
            if i > 0:
                patch = matplotlib.patches.PathPatch(matplotlib.path.Path(
                    [[px2mm(paths[i - 1][-1].end.real), px2mm(paths[i - 1][-1].end.imag)],
                     [px2mm(path[0].start.real), px2mm(path[0].start.imag)]],
                    [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO]), ec='black', fill=False, lw=1, ls='--')
                ax.add_patch(patch)
            else:
                patch = matplotlib.patches.PathPatch(matplotlib.path.Path(
                    [[0., 0.], [px2mm(path[0].start.real), px2mm(path[0].start.imag)]],
                    [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO]), ec='black', fill=False, lw=1, ls='--')
                ax.add_patch(patch)

    patch = matplotlib.patches.PathPatch(
        matplotlib.path.Path([[px2mm(paths[-1][-1].end.real), px2mm(paths[-1][-1].end.imag)], [0., 0.]],
                             [matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO]), ec='black', fill=False, lw=1,
        ls='--')
    ax.add_patch(patch)

    plt.legend(lhandles, llabels)
    plt.axis('equal')
    ax.invert_yaxis()
    plt.show()

# def cutting_time(paths, cutting_speed):
#    # Convert cutting_speed to real units
#    cutting_speed = float(cutting_speed) / 100. * MAX_CUTTING_SPEED
#
#    # Calculate cutting path length first
#    pathlength = 0.
#    for path in paths:
#        pathlength += path.length()
#
#    pathlength = px2mm(pathlength)
#    cutting_time = pathlength / cutting_speed
#
#    idletime = 0.
#    # Calculate distance to get up to full speed and decelerate to cutting speed again
#    acclength = (MAX_CUTTING_SPEED + cutting_speed) * (MAX_CUTTING_SPEED - cutting_speed) / ACCELERATION
#    acctime = 2. * (MAX_CUTTING_SPEED - cutting_speed) / ACCELERATION
#
#    idlepathlengths = calculate_idle_pathlengths(paths)
#    for idlepathlength in idlepathlengths[1:-1]:
#        if idlepathlength >= acclength:
#            idletime += acctime
#            idletime += (idlepathlength - acclength) / MAX_CUTTING_SPEED
#        else:
#            idletime += 2. * (np.sqrt(ACCELERATION * idlepathlength + cutting_speed ** 2) - cutting_speed) / \
#                        ACCELERATION
#
#    acclength += cutting_speed ** 2 / (2. * ACCELERATION)
#    acctime += cutting_speed / ACCELERATION
#
#    for idlepathlength in [idlepathlengths[0], idlepathlengths[-1]]:
#        if idlepathlength >= acclength:
#            idletime += acctime
#            idletime += (idlepathlength - acclength) / MAX_CUTTING_SPEED
#        elif idlepathlength >= cutting_speed ** 2 / (2. * ACCELERATION):
#            idlepathlength = idlepathlength - cutting_speed ** 2 / (2. * ACCELERATION)
#            idletime += cutting_speed / ACCELERATION
#            idletime += 2. * (np.sqrt(ACCELERATION * idlepathlength + cutting_speed ** 2) - cutting_speed) / \
#                        ACCELERATION
#        else:
#            print('Not able to get up to speed before reaching first path!')
#
#    return cutting_time + idletime
