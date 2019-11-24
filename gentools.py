# Standard libraries
import csv
import copy
import os.path as osp
from collections import deque
from math import pi, sin, cos, atan2, hypot
from pathlib import Path
import pickle
import tkinter.filedialog as tkfd

# 3rd-party libraries
import numpy as np
import polarTransform as pt

########################################################################################################################


HOLDER_PATH = Path("C:\\Users\\Taylor\\Desktop\\")


########################################################################################################################


def CSV_to_matrix(csvPath, headerRows=0):
    """Load CSV file as a numpy matrix.
    Optionally return header matrix with specified number of rows."""
    with open(csvPath, 'r') as csvfile:
        reader = csv.reader(csvfile, 'unix')
        matrix = [row for row in reader]
        if headerRows:
            header = np.squeeze(np.array(matrix[:headerRows]))
            matrix = np.array(matrix[headerRows:])
            return matrix, header
        else:
            matrix = np.array(matrix)
            return matrix


def repackage_CSV_input(data, header, *fields, identifier='code'):
    """Repackages the output of CSV_to_matrix in dictionary format.
    Intended for columnar data which has a particular field (column) that will be used to identify individual records (row).
    If 'identifier' is None, it will package a flat dictionary with entries pointing to entire columns instead of individual records.
    Implicitly, 'code' column should contain unique values.
    If *fields is provided any 2-tuples, they will be evaluated as (field, type) for conversion purposes."""

    assert data.shape[1] == len(header), "'data' & 'header' should have the same number of columns."
    repackage = {}
    hInds = {e: int(i) for i, e in enumerate(header)}
    if identifier is None:  # Make dict of columns
        for f in fields:
            if isinstance(f, tuple):
                f, t = f
                column = data[:, hInds[f]]
                column = column.astype('object')
                for i, e in enumerate(column):
                    try:
                        column[i] = convert_type(e, t)
                    except ValueError:
                        column[i] = None
                repackage[f] = column
            else:
                repackage[f] = data[:, hInds[f]]
    else:  # Make dict of records, which are dicts of fields
        for r in data:
            subdict = {}
            for f in fields:
                if isinstance(f, tuple):
                    f, t = f
                    e = r[hInds[f]]
                    try:
                        subdict[f] = convert_type(e, t)
                    except ValueError:
                        subdict[f] = None
                else:
                    subdict[f] = r[hInds[f]]
            repackage[r[hInds[identifier]]] = subdict

    return repackage


def matrix_to_CSV(filePath, *args):
    """Export matrix/numpy array to CSV file.
    All elements of args must be iterables."""
    _, ext = osp.splitext(filePath)
    if ext != '.csv':
        filePath = f"{str(filePath)}.csv"
    with open(filePath, 'w') as csvfile:
        writer = csv.writer(csvfile, 'unix', quoting=csv.QUOTE_MINIMAL)
        for e in args:
            twoD = np.array(e, dtype='object') if isinstance(e, list) else e  # Generalize with 2D numpy arrays
            twoD = np.expand_dims(twoD, 0) if twoD.ndim == 1 else twoD
            writer.writerows(twoD)


def load_pickle_file(path=None):
    """Load a pickled file for examination."""
    if path is None:
        path = Path(tkfd.askopenfilename(title="Select OLAY", initialdir=HOLDER_PATH))
    with open(path, 'rb') as fileStream:
        package = pickle.load(fileStream)

    return package


########################################################################################################################


def flood_fill(arr, start:tuple, paintColor, wraparound=False):
    """Generic handler for flood fill requests."""

    start = tuple(start)
    if arr.ndim == 2:
        if wraparound:
            arr = _flood_fill_grey_wraparound_y(arr, start, paintColor)
        else:
            arr = _flood_fill_grey(arr, start, paintColor)
    elif arr.ndim == 3:
        assert arr.shape[2] == 3, "Input array must be MxNx3 RGB numpy array."
        assert len(paintColor) == 3, "Input color must be RGB."
        encodedArr = arr[:, :, 0]*255**2 + arr[:, :, 1]*255 + arr[:, :, 2]  # Encodes RGB for efficient value comparison
        encodedColor = paintColor[0]*255**2 + paintColor[1]*255 + paintColor[2]
        if wraparound:
            encodedArr = _flood_fill_grey_wraparound_y(encodedArr, start, encodedColor)
        else:
            encodedArr = _flood_fill_grey(encodedArr, start, encodedColor)
        arr[:, :, 0], r = divmod(encodedArr, 255**2)
        arr[:, :, 1], r = divmod(r, 255)
        arr[:, :, 2] = r

    return arr


def _flood_fill_grey(arr, start:tuple, paintColor):
    """Four-direction, line-by-line flood fill algorithm for MxN array.
    Outer loop is vertical, inner is horizontal."""

    targetColor = arr[start[0], start[1]]
    if targetColor == paintColor:
        return arr
    callStack = deque([start])

    while len(callStack):
        y, x = callStack.pop()

        previousWallAboveFlag = True
        previousWallBelowFlag = True

        # Zip leftward to a wall.
        while x > 0:
            px = arr[y,x]
            if px == targetColor:
                x -= 1
            else:
                x += 1
                break

        # Fill rightward to a wall.
        while x < arr.shape[1]:
            px = arr[y,x]
            if px == targetColor:
                arr[y,x] = paintColor
            else:
                break

            # Watch for new segments to fill below.
            if y < arr.shape[0]-1:
                belowPx = arr[y+1,x]
                if belowPx != targetColor:
                    previousWallBelowFlag = True
                else:
                    if previousWallBelowFlag:
                        callStack.append((y+1, x))  # Add the start (left) of a below new segment to the stack.
                    previousWallBelowFlag = False

            # Watch for new segments to fill above.
            if y > 0:
                abovePx = arr[y-1,x]
                if abovePx != targetColor:
                    previousWallAboveFlag = True
                else:
                    if previousWallAboveFlag:
                        callStack.append((y-1, x))  # Add the start (left) of an above new segment to the stack.
                    previousWallAboveFlag = False

            x += 1

    return arr


def _flood_fill_grey_wraparound_x(arr, start:tuple, paintColor):
    """Four-direction, line-by-line flood fill algorithm for MxN array.
    Outer loop is vertical, inner is horizontal.
    Wraps around the image in the horizontal dimension."""

    targetColor = arr[start[0], start[1]]
    lowest = arr.shape[0] - 1
    rightest = arr.shape[1] - 1
    if targetColor == paintColor:
        return arr
    callStack = deque([start])

    while len(callStack):
        y, x = callStack.pop()

        previousWallAboveFlag = True
        previousWallBelowFlag = True

        crossedLeftward = False
        crossedRightward = False

        # Zip leftward to a wall.
        while x > 0:
            px = arr[y,x]
            if px == targetColor:
                x -= 1
                if x == 0 and not crossedLeftward:  # Wraparound leftward once
                    x = rightest
                    crossedLeftward = True
            else:
                x += 1
                break

        # Fill rightward to a wall.
        while x < arr.shape[1]:
            px = arr[y,x]
            if px == targetColor:
                arr[y,x] = paintColor
            else:
                break

            # Watch for new segments to fill below.
            if y < lowest:
                belowPx = arr[y+1,x]
                if belowPx != targetColor:
                    previousWallBelowFlag = True
                else:
                    if previousWallBelowFlag:
                        callStack.append((y+1, x))  # Add the start (left) of a below new segment to the stack.
                    previousWallBelowFlag = False

            # Watch for new segments to fill above.
            if y > 0:
                abovePx = arr[y-1,x]
                if abovePx != targetColor:
                    previousWallAboveFlag = True
                else:
                    if previousWallAboveFlag:
                        callStack.append((y-1, x))  # Add the start (left) of an above new segment to the stack.
                    previousWallAboveFlag = False

            x += 1
            if x == arr.shape[1] and not crossedRightward:  # Wraparound rightward once
                x = 0
                crossedRightward = True

    return arr


def _flood_fill_grey_wraparound_y(arr, start:tuple, paintColor):
    """Four-direction, line-by-line flood fill algorithm for MxN array.
    Outer loop is vertical, inner is horizontal.
    Wraps around in the image in the vertical dimension."""

    targetColor = arr[start[0], start[1]]
    lowest = arr.shape[0] - 1
    rightest = arr.shape[1] - 1
    if targetColor == paintColor:
        return arr
    callStack = deque([start])

    while len(callStack):
        y, x = callStack.pop()

        previousWallAboveFlag = True
        previousWallBelowFlag = True

        # Zip leftward to a wall.
        while x > 0:
            px = arr[y,x]
            if px == targetColor:
                x -= 1
            else:
                x += 1
                break

        # Fill rightward to a wall.
        while x < arr.shape[1]:
            px = arr[y,x]
            if px == targetColor:
                arr[y,x] = paintColor
            else:
                break

            # Watch for new segments to fill below.
            Y = y+1 if y < lowest else 0
            belowPx = arr[Y,x]
            if belowPx != targetColor:
                previousWallBelowFlag = True
            else:
                if previousWallBelowFlag:
                    callStack.append((Y, x))  # Add the start (left) of a below new segment to the stack.
                previousWallBelowFlag = False

            # Watch for new segments to fill above.
            Y = y-1 if y > 0 else lowest
            abovePx = arr[Y,x]
            if abovePx != targetColor:
                previousWallAboveFlag = True
            else:
                if previousWallAboveFlag:
                    callStack.append((Y, x))  # Add the start (left) of an above new segment to the stack.
                previousWallAboveFlag = False

            x += 1

    return arr


def flood_smallest_box(arr, start:tuple):
    """Four-direction, line-by-line flood fill algorithm for MxN RGB array.
    Instead of painting, finds the smallest-box-fit center of a region."""

    assert arr.ndim == 3 and arr.shape[2] == 3, "Input array must be MxNx3 RGB numpy array."
    start = tuple(start)

    x, y = start
    xmin = x
    xmax = x
    ymin = y
    ymax = y
    count = 0
    targetColor = tuple(arr[y, x])
    callStack = [start]

    while len(callStack):
        if len(callStack) == 0:
            return
        x, y = callStack.pop()

        curr = x
        newAboveSegmentFlag = True
        newBelowSegmentFlag = True
        ymin = min(ymin, y)
        ymax = max(ymax, y)

        # Zip leftward to a wall.
        while(curr > 0):
            px = tuple(arr[y,curr])
            if px == targetColor:
                curr -= 1
            else:
                curr += 1
                xmin = min(xmin, curr)
                break

        # Fill rightward to a wall.
        while(curr < arr.shape[1]):
            px = tuple(arr[y,curr])
            if px == targetColor:
                arr[y,curr] = [256, 256, 256]  # Nonexistent color in place of a "visited" boolean
                count += 1
            else:
                xmax = max(xmax, curr-1)
                break

            # Watch for new segments to fill below.
            if y < arr.shape[0]-1:
                belowPx = tuple(arr[y+1,curr])
                if belowPx != targetColor:
                    newBelowSegmentFlag = True
                else:
                    if newBelowSegmentFlag:
                        callStack.append((curr, y+1))  # Add the start (left) of a below new segment to the stack.
                    newBelowSegmentFlag = False

            # Watch for new segments to fill above.
            if y > 0:
                abovePx = tuple(arr[y-1,curr])
                if abovePx != targetColor:
                    newAboveSegmentFlag = True
                else:
                    if newAboveSegmentFlag:
                        callStack.append((curr, y-1))  # Add the start (left) of an above new segment to the stack.
                    newAboveSegmentFlag = False

            curr += 1

    return xmin, xmax, ymin, ymax, count


def flood_fill_3D(arr, start:tuple, paintColor:tuple=(0,0,0), sixDir=True):
    """Inefficient, 3D flood-fill algorithm."""

    assert arr.ndim == 4, "Array must be 3D RGB volume."
    assert 3 <= arr.shape[-1] <= 4, "Last dimension should be RGB."
    arr = copy.deepcopy(arr)

    if arr.shape[-1] == 4:
        arr = arr[:, :, :, :3]  # Strip alpha if present

    targetColor = tuple(arr[start[0], start[1], start[2]])
    callStack = []
    callStack.append(start)

    if sixDir:
        xis = [-1, 1, 0, 0, 0, 0]
        yis = [0, 0, -1, 1, 0, 0]
        zis = [0, 0, 0, 0, -1, 1]
    else:
        xis = [-1, 0, 1]*9
        yis = [[-1]*3 + [0]*3 + [1]*3]*3
        zis = [-1]*9 + [0]*9 + [1]*9
        xis[13:14] = []  # Delete center pixel from 27-block
        yis[13:14] = []
        zis[13:14] = []
    while len(callStack):
        x, y, z = callStack.pop()
        arr[x,y,z] = list(paintColor)
        for i in range(len(xis)):
            xi = xis[i]
            yi = yis[i]
            zi = zis[i]
            if tuple(arr[xi,yi,zi]) == targetColor:
                callStack.append((xi,yi,zi))

    return arr


########################################################################################################################


def findNth(searchString, substring, nth):
    """Returns the index of the Nth occurence of substring in the searchString.
    Includes overlapping instances.
    Returns -1 if substring not found."""
    ind = -1
    for n in range(nth):
        ind += 1
        ind = searchString.find(substring, ind)
        if ind == -1:
            return ind

    return ind


def convert_type(arg, type):
    """Convert to Python built-in type."""

    if type is str:
        return str(arg)
    if type is int:
        return int(arg)
    if type is float:
        return float(arg)
    if type is bool:
        return bool(arg)

    print("Type not preprogrammed.")


def scale(mat, oMin=0, oMax=1, optIMin=None, optIMax=None, truncateRange=True):
    """Scale matrix."""
    #TODO Raise warning if mat is unary but either optIMin and/or optIMax are not given
    if optIMin is None:
        iMin = np.min(mat)
    else:
        iMin = optIMin
    if optIMax is None:
        iMax = np.max(mat)
    else:
        iMax = optIMax
    if oMin == iMin and oMax == iMax:  # No scaling necessary
        return mat

    range_ = iMax - iMin
    scaled = (mat - iMin) / range_ * oMax + oMin
    if truncateRange:
        if isinstance(mat, np.ndarray):
            scaled[scaled < oMin] = oMin
            scaled[scaled > oMax] = oMax
        else:
            scaled = max(scaled, oMin)
            scaled = min(scaled, oMax)

    return scaled


def rgb_to_hsl(input_):
    """Convert an RGB dataset to HSL.
    Interprets 1-2D input as set of values; interprets 3D input as image.
    Last dimension should be RGB.
    RGB ranges [0,255].
    Hue ranges (0,6]; 0 is grey.
    Saturation and lightness range [0,1]."""
    assert 3 <= input_.shape[-1] <= 4, "RGB dimension should be last."
    if input_.shape[-1] == 4:
       input_ = input_[:, :, :3]  # Strip the alpha, if present

    input_ = scale(input_, optIMin=0, optIMax=255)
    max_ = np.max(input_, axis=-1)
    maxInds = np.argmax(input_, axis=-1)
    min_ = np.min(input_, axis=-1)
    delta = max_ - min_
    lightness = (max_ + min_) / 2
    chroma = delta / (1 - np.abs(2 * lightness - 1))
    chroma[np.isnan(chroma)] = 0  # Black/white
    hue = np.zeros_like(chroma)

    if input_.ndim == 3:  # Process as image
        for y in range(input_.shape[0]):
            for x in range(input_.shape[1]):
                if delta[y, x] == 0:
                    hue[y, x] = 0
                elif maxInds[y, x] == 0:
                    hue[y, x] = ((input_[y, x, 1] - input_[y, x, 2]) / delta[y, x]) % 6
                elif maxInds[y, x] == 1:
                    hue[y, x] = (input_[y, x, 2] - input_[y, x, 0]) / delta[y, x] + 2
                elif maxInds[y, x] == 2:
                    hue[y, x] = (input_[y, x, 0] - input_[y, x, 1]) / delta[y, x] + 4

    elif input_.ndim == 2:  # Process as set
        for i in range(len(input_)):
            if delta[i] == 0:
                hue[i] = 0
            elif maxInds[i] == 0:
                hue[i] = ((input_[i, 1] - input_[i, 2]) / delta[i]) % 6
            elif maxInds[i] == 1:
                hue[i] = (input_[i, 2] - input_[i, 0]) / delta[i] + 2
            elif maxInds[i] == 2:
                hue[i] = (input_[i, 0] - input_[i, 1]) / delta[i] + 4

    elif input_.ndim == 1:  # Process as individual
        if delta == 0:
            hue = 0
        elif maxInds == 0:
            hue = ((input_[1] - input_[2]) / delta) % 6
        elif maxInds == 1:
            hue = (input_[2] - input_[0]) / delta + 2
        elif maxInds == 2:
            hue = (input_[0] - input_[1]) / delta + 4

    hsl = np.stack((hue, chroma, lightness), axis=-1)
    return hsl


def hsl_to_rgb(input_):
    """Convert an HSL dataset to RGB.
    Interprets 1-2D input as set of values; interprets 3D input as image.
    Last dimension should be HSL.
    RGB ranges [0,255].
    Hue ranges (0,6]; 0 is grey.
    Saturation and lightness range [0,1]."""

    assert input_.shape[-1] == 3, "Last dimension should be the HSL."

    if input_.ndim == 3:  # Interpret as image
        hue = input_[:, :, 0]
        sat = input_[:, :, 1]
        lig = input_[:, :, 2]
        delta = sat * (1 - np.abs(2*lig - 1))
        out = delta * (1 - np.abs(hue % 2 - 1))
        floor = lig - delta/2
        rgb = np.zeros_like(input_)
        for y in range(input_.shape[0]):
            for x in range(input_.shape[1]):
                if hue[y,x] < 1:
                    rgb[y,x,:] = [delta[y,x], out[y,x], 0]
                elif hue[y,x] < 2:
                    rgb[y,x,:] = [out[y,x], delta[y,x], 0]
                elif hue[y,x] < 3:
                    rgb[y,x,:] = [0, delta[y,x], out[y,x]]
                elif hue[y,x] < 4:
                    rgb[y,x,:] = [0, out[y,x], delta[y,x]]
                elif hue[y,x] < 5:
                    rgb[y,x,:] = [out[y,x], 0, delta[y,x]]
                elif hue[y,x] < 6:
                    rgb[y,x,:] = [delta[y,x], 0, out[y,x]]
                rgb[y,x,:] += floor[y,x]
        rgb = (rgb*255).astype('uint8')

    if input_.ndim == 2:  # Interpret as set
        hue = input_[:, 0]
        sat = input_[:, 1]
        lig = input_[:, 2]
        delta = sat * (1 - np.abs(2*lig - 1))
        out = delta * (1 - np.abs(hue % 2 - 1))
        floor = lig - delta/2
        rgb = np.zeros_like(input_)
        for i in range(len(input_)):
            if 0 <= hue[i] < 1:
                rgb[i,:] = [delta[i], out[i], 0]
            elif hue[i] < 2:
                rgb[i,:] = [out[i], delta[i], 0]
            elif hue[i] < 3:
                rgb[i,:] = [0, delta[i], out[i]]
            elif hue[i] < 4:
                rgb[i,:] = [0, out[i], delta[i]]
            elif hue[i] < 5:
                rgb[i,:] = [out[i], 0, delta[i]]
            elif hue[i] < 6:
                rgb[i,:] = [delta[i], 0, out[i]]
            rgb[i,:] += floor[i]
        rgb = (rgb*255).astype('uint8')

    if input_.ndim == 1:  # Interpret as individual
        hue = input_[0]
        sat = input_[1]
        lig = input_[2]
        delta = sat * (1 - np.abs(2 * lig - 1))
        out = delta * (1 - np.abs(hue % 2 - 1))
        floor = lig - delta / 2
        if hue < 1:
            rgb = [delta, out, 0]
        elif hue < 2:
            rgb = [out, delta, 0]
        elif hue < 3:
            rgb = [0, delta, out]
        elif hue < 4:
            rgb = [0, out, delta]
        elif hue < 5:
            rgb = [out, 0, delta]
        elif hue < 6:
            rgb = [delta, 0, out]
        rgb += floor
        rgb = (rgb * 255).astype('uint8')

    return rgb


def hsl_to_cart(hsl):
    """Converts HSL coordinates into Cartesian domain.
    Last dimension should be HSL.
    Hue ranges (0,6]; 0 is grey.
    Saturation and lightness range [0,1].
    X/Y:= [-1,1]; Z:= [0,1]."""

    hsl = np.array(hsl)
    assert hsl.ndim < 3 and hsl.shape[-1] == 3, "'hsl' should be 1-2D array with HSL in the last dimension."
    if hsl.ndim == 1:  # Expand to 2D to generalize for individual inputs
        hsl = np.expand_dims(hsl, 0)

    hue = hsl[:, 0]*pi/3  # Convert from 6-sectors to radians
    sat = hsl[:, 1]
    lig = hsl[:, 2]
    x = sat*cos(hue)
    y = sat*sin(hue)
    z = lig
    cart = np.stack((x,y,z), axis=-1)
    cart = np.squeeze(cart)

    return cart


def cart_to_hsl(cart):
    """Converts Cartesian coordinates into HSL domain.
    Last dimension should be XYZ.
    Hue ranges (0,6]; 0 is grey.
    Saturation and lightness range [0,1].
    X/Y:= [-1,1]; Z:= [0,1]."""

    cart = np.array(cart)
    assert cart.ndim < 3 and cart.shape[-1] == 3, "'cart' should be 1-2D array with XYZ in the last dimension."
    if cart.ndim == 1:  # Expand to 2D to generalize for individual inputs
        cart = np.expand_dims(cart, 0)

    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    hue = atan2(y, x)*3/pi
    sat = hypot(x, y)
    lig = z
    hsl = np.stack((hue, sat, lig), axis=-1)

    return hsl


# TODO Allow 'originCenter' to be True
def build_polar_cartesian_luts(width, height, wrapCounterClockwise=True, originCenter=False, downscale=2):
    """Create lookup table from Cartesian to polar domains.
    Wrap starts pointing east.
    If 'originCenter' is False, then Cartesian coordinates are located in the 1st quadrant.
    'downscale' is the ratio of the polar radius over the Cartesian radius.
    p2clut[theta, r, [x, y]] ; c2plut[x, y, [theta, r]]."""
    theta = np.linspace(0, 2 * pi, width, endpoint=False)
    if wrapCounterClockwise:
        theta = np.flip(theta)
    rs = list(range(height)) * width
    ts = []
    for i in range(width):
        ts += [theta[i]] * height
    coords = np.stack((rs, ts), axis=-1)
    coords = pt.getCartesianPoints(coords, (height, height))
    coords = (coords / downscale)
    p2clut = np.reshape(coords, (width, height, 2))
    xs = list(range(height)) * height
    ys = []
    for i in range(height):
        ys += [i] * height
    coords = np.stack((xs, ys), axis=-1)
    coords = pt.getPolarPoints(coords, (height / downscale, height / downscale))
    coords = np.reshape(coords, (height, height, 2))
    if wrapCounterClockwise:
        coords = np.flip(coords, axis=-1)
    coords[:, :, 0] = coords[:, :, 0] * width / 2 / pi
    coords[:, :, 0] = np.rot90(coords[:, :, 0], k=-1)
    coords[:, :, 0] = (width - 1) - coords[:, :, 0]
    coords[:, :, 1] = (height - 1) - coords[:, :, 1] * downscale
    c2plut = coords

    return p2clut, c2plut


def identify_segments(binary):
    """Detect the boundaries of segments in a 1D binary array."""
    assert binary.ndim == 1, "'binary' should be a 1D binary array."
    segments = []
    onSegmentFlag = False
    firstB = len(binary) - 1
    secondB = 0
    for i2 in range(len(binary)):
        if binary[i2] and not onSegmentFlag:  # Moving onto foreground segment
            firstB = i2
            onSegmentFlag = True
        elif onSegmentFlag and not binary[i2]:  # Moving off foreground segment
            secondB = i2
            onSegmentFlag = False
            segments.append([firstB, secondB])

    if firstB >= secondB:  # Close dangling foreground segment
        if len(segments):  # Segment straddles the frame
            segments[0] = [firstB, segments[0][1]]
        elif firstB == secondB == 0:  # Segment encompasses whole frame
            segments.append(firstB, len(binary))
        elif firstB == len(binary)-1 and secondB == 0:  # No foreground in the frame
            pass
        else:  # Edge case: segment exactly touches right edge without crossing
            segments.append([firstB, secondB])

    return segments
