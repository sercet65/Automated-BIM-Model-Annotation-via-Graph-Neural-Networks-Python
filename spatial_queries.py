
from distance3d import random, plotting, gjk, colliders
from pytransform3d.transform_manager import TransformManager
from distance3d.broad_phase import BoundingVolumeHierarchy
from distance3d.urdf_utils import fast_transform_manager_initialization
from distance3d.colliders import Box
import numpy as np
import pytransform3d.visualizer as pv


def braod_search(formatted_shapes: dict) -> list:
    # if reduction_criteria: we we don't check all we start off with a subselectioon of all shapes. e.g. we only comapre each wall to all spaces. but don't comapare walls to walls
    # we do this by setting the reduction_criteria to a list of lists. each list contains the indices of the shapes that should be compared to each other
    # e.g. reduction_criteria = [[0,1,2,3], [4,5,6,7], [8,9,10,11]] would compare the first 4 shapes to each other, the next 4 shapes to each other and the last 4 shapes to each other

    tm = TransformManager(check=False)

    bvh = BoundingVolumeHierarchy(tm, "base")

    fast_transform_manager_initialization(tm, list(formatted_shapes.keys()), "base")

    for i, formatted_shape in formatted_shapes.items():
        # make box2origin and size from bbx
        box2origin = np.eye(4)
        size = formatted_shape[3:] - formatted_shape[:3]
        size = size.astype(np.float64)  # Convert size to float64
        box2origin[:3, 3] = formatted_shape[0:3] + size / 2

        collider_box = Box(box2origin, size)

        # collider_box.make_artist(c=(0, 1, 0))

        #tm.add_transform(i, "base", box2origin)
        bvh.add_collider(i, collider_box)

    pairs = bvh.aabb_overlapping_with_self()
    # remove duplicates, because order is not important
    pairs = list(set([tuple(sorted(pair)) for pair in pairs]))
    return pairs, bvh


def close_search_distances(pairs, formatted_shapes, TOLERANCE=0.1, geometry_to_use="box"):

    pairs_close, dists, closest_points = [], [], []

    for (frame1, collider1), (frame2, collider2) in pairs:

        # dist, p1, p2, _ = gjk.gjk(colliders.ConvexHullVertices(formatted_shapes[frame1][0]),
        #                           colliders.ConvexHullVertices(formatted_shapes[frame2][0]))
        if geometry_to_use == "box":
            box2origin1 = np.eye(4)
            size1 = formatted_shapes[frame1][3:] - formatted_shapes[frame1][:3]
            size1 = size1.astype(np.float64)
            box2origin1[:3, 3] = formatted_shapes[frame1][0:3] + size1 / 2
            collider_box = Box(box2origin1, size1)

            box2origin2 = np.eye(4)
            size2 = formatted_shapes[frame2][3:] - formatted_shapes[frame2][:3]
            size2 = size2.astype(np.float64)  # Convert size to float64
            box2origin2[:3, 3] = formatted_shapes[frame2][0:3] + size2 / 2
            collider_box2 = Box(box2origin2, size2)


            dist, p1, p2, _ = gjk.gjk(collider_box, collider_box2)


        if dist < TOLERANCE:
            pairs_close.append((frame1, frame2))
            dists.append(dist)
            closest_points.append((p1, p2))

    return pairs_close, dists, closest_points


def get_adjacencies(shapes, distance_tolerance, visualise=False):
    # store distances between each pair. store in a way that is easily filterable afterwards
    # make one d array of all the distances
    # measure time
    broad_pairs, bvh = braod_search(shapes)
    pairs_close, distances, closest_points = close_search_distances(broad_pairs, shapes, distance_tolerance)

    if visualise:
        for frame1, frame2 in pairs_close:
            bvh.colliders_[frame1].artist_.geometries[0].paint_uniform_color((1, 0, 0))
        fig = pv.figure()
        for artist in bvh.get_artists():
            artist.add_artist(fig)

        if "__file__" in globals():
            fig.show()
        else:
            fig.save_image("__open3d_rendered_image.jpg")

    return pairs_close, distances, closest_points

def test():
    pass