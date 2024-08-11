from __future__ import annotations
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Callable
from shapely import Polygon, Point
import glob
from pathlib import PureWindowsPath


def get_all_png_files(root: str) -> list[str]:
    result = glob.glob(f"{root}/**/*.png", recursive=True)
    return [PureWindowsPath(x).as_posix() for x in result]


def get_boundary_iou(image: np.array, boundary: np.ndarray) -> float:
    """
    Get the IOU of the boundary with the image.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv.fillPoly(mask, [boundary.astype(np.int32)], 1)
    intersection = np.logical_and(image, mask)
    union = np.logical_or(image, mask)
    if image.sum() == 0:
        return 1
    return intersection.sum() / union.sum()


def get_gray_image(file_name: str, image_size=224) -> np.ndarray:
    """
    Get the gray image.
    """
    image = cv.imread(file_name)
    image = cv.resize(image, (image_size, image_size))
    def get_white_part(image: np.ndarray) -> np.ndarray:
        white_mask = np.zeros_like(image) + 255
        black_mask = np.zeros_like(image)
        return np.where((image == [255, 255, 255]).all(axis=-1)[..., None], white_mask, black_mask)
    image = get_white_part(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def get_video_path() -> dict[str, list[str]]:
    """
    Get the video path of the DAVIS dataset.
    """

    split_file_path = "./ImageSets/480p/trainval.txt"
    with open(split_file_path, "r") as f:
        file_names = f.readlines()
    file_names = [x.strip() for x in file_names]
    file_names = [x.split(" ") for x in file_names]
    file_names = [["." + y for y in x] for x in file_names]

    def get_video_name(file_name: str) -> str:
        return file_name.split("/")[-2]

    video_names = {}
    for file_name in file_names:
        video_name = get_video_name(file_name[0])
        if video_name not in video_names:
            video_names[video_name] = []
        video_names[video_name].append(file_name)
    for k, v in video_names.items():
        video_names[k] = sorted(v)

    remove_video_names = []
    for video_name, file_names in video_names.items():
        for file_name in file_names:
            image = get_gray_image(file_name[1])
            if image.sum() == 0:
                remove_video_names.append(video_name)
                break

    for video_name in remove_video_names:
        del video_names[video_name]

    return video_names


def get_boundary_points(image: np.array) -> np.ndarray:
    """
    Get the boundary points of the largest component in the image.
    """

    # get the boundaries
    boundaries, contours = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )
    boundaries = [np.squeeze(boundary) for boundary in boundaries]

    # get max-area boundary
    max_area = 0
    max_area_boundary = boundaries[0]
    for boundary in boundaries:
        if len(boundary) < 3:
            continue
        area = cv.contourArea(boundary)
        if area > max_area:
            max_area = area
            max_area_boundary = boundary

    return max_area_boundary


def uniform_sample_points(boundary: np.array, num_points: int) -> np.ndarray:
    """
    Sample the boundary points.
    """
    def get_fit_boundary_points(boundary_point:np.array, num_points:int):
        fit_boundary_points = []
        for i in range(num_points):
            if i == num_points - 1:
                fit_boundary_points.append(boundary_point[-1])
                break
            idx = int(i * len(boundary_point) / num_points)
            fit_boundary_points.append(boundary_point[idx])
        return np.array(fit_boundary_points)
    if len(boundary) > num_points:
        num_points = min(num_points, len(boundary))
        indices = np.linspace(0, len(boundary) - 1, num_points, dtype=int)
        return boundary[indices]
    else:
        return get_fit_boundary_points(boundary, num_points)


def test_uniform_sample_num(
    num_points: int,
    video_names: dict[str, list[list[str]]],
) -> float:
    """
    Test the sample number.
    """
    total_iou = 0
    total_len = 0
    for video_name in video_names:
        total_len += len(video_names[video_name])
        for file_name in video_names[video_name]:
            # print(file_name[1])
            image = get_gray_image(file_name[1])
            boundary = get_boundary_points(image)
            boundary = uniform_sample_points(boundary, num_points)
            iou = get_boundary_iou(image, boundary)
            total_iou += iou
    return total_iou / total_len


def plot_test_results(test_results: dict, label: str = None):
    """
    Plot the test results.
    """
    # plt.plot(list(test_results.keys())[150:], list(test_results.values())[150:], label=label)
    # plt.plot(sorted(list(test_results.keys())), sorted(list(test_results.values())), label=label)
    x = sorted([int(i) for i in test_results.keys()])
    y = [float(test_results[str(i)]) for i in x]
    plt.plot(x, y, label=label)
    plt.xlabel("Number of Sample Points")
    plt.ylabel("Average IoU")
    plt.xticks(np.arange(0, 1000, 100))
    plt.grid()
    plt.legend()
    plt.gca().invert_xaxis()
    plt.title("Average IoU vs Number of Sample Points")


class Vertex:
    def __init__(
        self,
        coord: tuple[int, int],
    ) -> None:
        self.point = Point(coord)
        self.pre_vertex = None
        self.succ_vertex = None
        self.triangle_area = None

    def update_area(self) -> None:
        if self.pre_vertex is not None and self.succ_vertex is not None:
            self.triangle_area = Polygon(
                [self.point, self.pre_vertex.point, self.succ_vertex.point]
            ).area

    def add_neighbors(self, pre_vertex: Vertex, succ_vertex: Vertex) -> None:
        self.pre_vertex = pre_vertex
        self.succ_vertex = succ_vertex
        self.update_area()


class Boundary_points:
    def __init__(self, boundary: np.ndarray) -> None:
        self.vertices = [Vertex(coord) for coord in boundary]
        for i in range(len(self.vertices)):
            self.vertices[i].add_neighbors(
                self.vertices[i - 1], self.vertices[(i + 1) % len(self.vertices)]
            )
        self.sort_vertices()

    def sort_vertices(self) -> None:
        self.vertices = sorted(self.vertices, key=lambda x: x.triangle_area)

    def remove_one_vertex(self) -> None:
        removed_vertex = self.vertices.pop(0)
        removed_vertex.pre_vertex.succ_vertex = removed_vertex.succ_vertex
        removed_vertex.pre_vertex.update_area()
        removed_vertex.succ_vertex.pre_vertex = removed_vertex.pre_vertex
        removed_vertex.succ_vertex.update_area()
        self.sort_vertices()

    def get_boundary_coods(self) -> np.ndarray:
        boundary_coods = []
        current_vertex = self.vertices[0]
        while current_vertex.succ_vertex != self.vertices[0]:
            boundary_coods.append([current_vertex.point.x, current_vertex.point.y])
            current_vertex = current_vertex.succ_vertex
        boundary_coods.append([current_vertex.point.x, current_vertex.point.y])
        return np.array(boundary_coods)

    def simplify_to_num(self, num: int) -> np.ndarray:
        offset = len(self.vertices) - num
        if offset <= 0:
            return self.get_boundary_coods()
        for i in range(offset):
            self.remove_one_vertex()
        return self.get_boundary_coods()
