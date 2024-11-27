import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pdb import set_trace as ST


# Function to convert a box from (x, y, w, h, angle) to its 4 corners
def get_corners(bbox):
    xc, yc, w, h, angle = bbox
    theta = np.radians(angle)

    '''
    # Compute the offsets w.r.t. the center
    wx, wy = w / 2 * np.cos(theta), w / 2 * np.sin(theta)
    hx, hy = -h / 2 * np.sin(theta), h / 2 * np.cos(theta)

    # Compute the vertices of the oriented bounding box
    p1 = (xc - wx - hx, yc - wy - hy)
    p2 = (xc + wx - hx, yc + wy - hy)
    p3 = (xc + wx + hx, yc + wy + hy)
    p4 = (xc - wx + hx, yc - wy + hy)
    '''

    half_w, half_h = w / 2, h / 2

    # Define box corner points relative to the center (along the horizontal axis)
    top_left     = np.array([-half_w, -half_h])
    top_right    = np.array([half_w,  -half_h])
    bottom_right = np.array([half_w,  half_h])
    bottom_left  = np.array([-half_w, half_h])
    corners = np.array([top_left, top_right, bottom_right, bottom_left])

    # Construct the rotation matrix
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Translate by the box's center
    rotated_corners = np.dot(corners, R.T) + np.array([xc, yc])

    return rotated_corners


'''
Functions for handling the out-of-bounadry issue
'''
def is_obb_out_of_bounds(corners, img_width=512, img_height=512):
    # Check if any corner of the oriented bounding box (OBB) extends beyond the image boundaries.
    for corner in corners:
        x, y = corner
        if x < 0 or x > img_width or y < 0 or y > img_height:
            return True  # Out of bounds
    return False


def adjust_obb_within_bounds(bbox, img_width=512, img_height=512):
    # Adjust the OBB to ensure it stays within the image boundaries.
    # Moves the box if necessary and optionally adjusts size.
    
    cx, cy, w, h, theta = bbox
    corners = get_corners(bbox)
    
    if is_obb_out_of_bounds(corners, img_width, img_height):
        # Center the OBB within the image bounds if needed (i.e., ensure the center locates within the image boundary)
        bbox[0] = np.clip(cx, w / 2, img_width - w / 2) # update cx
        bbox[1] = np.clip(cy, h / 2, img_height - h / 2) # update cy
        
        # Recompute corners and check again after adjusting the center
        corners = get_corners(bbox)
        
        # Resize the OBB to fit within the image if it is still out of bounds
        while is_obb_out_of_bounds(corners, img_width, img_height) and w > 0 and h > 0:
            bbox[2] *= 0.9  # Shrink width by 10%
            bbox[3] *= 0.9  # Shrink height by 10%
            corners = get_corners(bbox)
        
    return bbox


'''
Functions for resolving the overlap among bounding boxes
'''
def project_onto_axis(corners, axis):
    # Project the corners of a bounding box onto a given axis and return the min and max projections.
    projections = np.dot(corners, axis)
    return projections.min(), projections.max()


def are_obb_overlapping(box1, box2):
    # Check if two oriented bounding boxes (OBBs) are overlapping using the Separating Axis Theorem (SAT).
    corners1 = get_corners(box1)
    corners2 = get_corners(box2)

    # Compute edges of both boxes
    edges1 = [corners1[i] - corners1[(i + 1) % 4] for i in range(4)]
    edges2 = [corners2[i] - corners2[(i + 1) % 4] for i in range(4)]

    # Axes to test (normals of the edges)
    axes = [np.array([-edge[1], edge[0]]) for edge in edges1 + edges2]

    # Check for overlap along each axis
    for axis in axes:
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)

        # If there is no overlap on this axis, the boxes do not overlap
        if max1 < min2 or max2 < min1:
            return False

    # If we didn't find a separating axis, the boxes must overlap
    return True


def compute_overlap_vector(corners1, corners2):
    # Compute the Minimum Translation Vector (MTV) to separate two overlapping oriented bounding boxes.
    edges1 = [corners1[i] - corners1[(i + 1) % 4] for i in range(4)]
    edges2 = [corners2[i] - corners2[(i + 1) % 4] for i in range(4)]

    axes = [np.array([-edge[1], edge[0]]) for edge in edges1 + edges2]

    min_overlap = float('inf')
    mtv_axis = None  # Minimum Translation Vector (MTV) axis

    for axis in axes:
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)

        overlap = min(max1 - min2, max2 - min1)
        if overlap < min_overlap:
            min_overlap = overlap
            mtv_axis = axis

    # The minimum translation vector is along the axis with the least overlap
    mtv = mtv_axis * min_overlap
    return mtv


def resolve_overlap_and_boundary(boxes, img_width=512, img_height=512, max_iterations=1000, tolerance=1e-4):
    # Resolve overlaps between oriented bounding boxes (OBBs) and ensure they stay within the image boundaries.
    
    moved_boxes = boxes.copy()  # Copy of boxes to move without modifying the original
    has_overlap = True
    overlap_detected = False
    iteration = 0

    while has_overlap and iteration < max_iterations:
        has_overlap = False
        iteration += 1
        for i in range(len(moved_boxes)):
            for j in range(i + 1, len(moved_boxes)):
                box1, box2 = moved_boxes[i], moved_boxes[j]

                # Check if boxes overlap
                if are_obb_overlapping(box1, box2):
                    overlap_detected = True

                    # Compute the minimum translation vector (MTV) to separate the boxes
                    corners1 = get_corners(box1)
                    corners2 = get_corners(box2)
                    mtv = compute_overlap_vector(corners1, corners2)

                    # Skip further adjustments if the movement is smaller than the tolerance
                    if np.linalg.norm(mtv) < tolerance:
                        continue

                    # Split the translation between the two boxes (move both boxes half of the MTV)
                    moved_boxes[i] = [box1[0] - mtv[0] / 2, box1[1] - mtv[1] / 2, box1[2], box1[3], box1[4]]
                    moved_boxes[j] = [box2[0] + mtv[0] / 2, box2[1] + mtv[1] / 2, box2[2], box2[3], box2[4]]

                    has_overlap = True  # Continue checking for overlaps

        # After resolving overlaps, check if any boxes are out of bounds
        for k in range(len(moved_boxes)):
            moved_boxes[k] = adjust_obb_within_bounds(moved_boxes[k], img_width, img_height)

    if iteration >= max_iterations:
        print('Max iterations reached, the layout may still contain overlaps.')

    return overlap_detected, moved_boxes


'''
Helper function for layout visualization
'''
# Fuction for painting bounding boxes onto the given image
def draw_box_desc(img, obj_names, bboxes):
    draw = ImageDraw.Draw(img)
        
    # font_folder = os.path.dirname(os.path.dirname(__file__))
    font_path = os.path.join('Rainbow-Party-2.ttf')
    font = ImageFont.truetype(font_path, 30)

    for obj_name, bbox in zip(obj_names, bboxes):
        rotated_corners = get_corners(bbox)
        p1 = (rotated_corners[0,0].item(), rotated_corners[0,1].item())
        p2 = (rotated_corners[1,0].item(), rotated_corners[1,1].item())
        p3 = (rotated_corners[2,0].item(), rotated_corners[2,1].item())
        p4 = (rotated_corners[3,0].item(), rotated_corners[3,1].item())
        
        # Draw the bbox
        draw.polygon(
            [p1, p2, p3, p4],
            outline='black',
            width=4,
        )

        # Annotate the class label
        x_min, y_min = p1
        label_pos = (int(x_min), int(y_min))
        draw.text(label_pos, obj_name, fill='black', font=font)

    return img


# visualize the layout for a single image
def visualize_layout(img_path, obj_names, bboxes, save_path):
    assert os.path.exists(img_path), f'Invalid image path ({img_path})!'

    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512))
    vis = draw_box_desc(img, obj_names, bboxes)
    vis.save(save_path)


# visualize the layout for retrieved examplars
def visualize_examplars(img_name_list, img_data_dict, save_path):
    # Create the canvas for visualization
    n_img = len(img_name_list)
    img_w, img_h = (512, 512)
    vis_res = Image.new('RGB', (img_w * n_img, img_h))

    # Add the probing image
    query_img_path = os.path.join('./images', img_name_list[0])
    query_img = Image.open(query_img_path).convert('RGB')
    query_img = query_img.resize((512, 512))
    vis_res.paste(query_img, (0, 0))
    
    for img_ind, img_name in enumerate(img_name_list[1:]):
        # load the examplar image
        img_path = os.path.join('./images', img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512))

        # obtain the visualization result
        # img_obj_names = img_data_dict[img_name]['obj_names']
        # img_bboxes = img_data_dict[img_name]['bboxes']
        # vis_img = draw_box_desc(img, img_obj_names, img_bboxes)
        
        vis_res.paste(img, ((img_ind+1) * img_w, 0))
    
    vis_res.save(save_path)
    
def poly2hbb(polys):
    """Convert polygons to horizontal bboxes.

    Args:
        polys (np.array): Polygons with shape (N, 8)

    Returns:
        np.array: Horizontal bboxes.
    """
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)

def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 1 or h < 1:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    a = a / np.pi * 180
    return x, y, w, h, a

def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4], axis=-1)
    return polys