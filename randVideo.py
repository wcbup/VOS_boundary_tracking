import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle, Arrow, FancyBboxPatch
import random
import os


class VideoObject:
    def __init__(self) -> None:
        self.name = ""
        self.x = 0
        self.y = 0

    def random_move(self):
        pass

    def draw(self, ax: plt.Axes):
        pass


class MyEllipse(VideoObject):
    def __init__(self, x, y, width, height, angle):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.name = "ellipse"

    def random_move(self):
        self.x += random.uniform(-2, 2)
        self.y += random.uniform(-2, 2)
        self.width += random.uniform(-2, 2)
        self.height += random.uniform(-2, 2)
        self.angle += random.uniform(-10, 10)
        min_length = 2
        max_length = 12
        if self.width < min_length:
            self.width = min_length
        if self.height < min_length:
            self.height = min_length
        if self.width > max_length:
            self.width = max_length
        if self.height > max_length:
            self.height = max_length
        x_min = -8
        x_max = 8
        y_min = -8
        y_max = 8
        if self.x < x_min:
            self.x = x_min
        if self.x > x_max:
            self.x = x_max
        if self.y < y_min:
            self.y = y_min
        if self.y > y_max:
            self.y = y_max

    def draw(self, ax: plt.Axes):
        e = Ellipse(
            xy=(self.x, self.y), width=self.width, height=self.height, angle=self.angle
        )
        ax.add_patch(e)
        e.set_facecolor("w")


class MyTriangle(VideoObject):
    def __init__(
        self,
        x_list: list[int],
        y_list: list[int],
    ):
        self.x_list = x_list
        self.y_list = y_list
        self.name = "triangle"

    def random_move(self):
        x_min = -8
        x_max = 8
        y_min = -8
        y_max = 8
        for i in range(3):
            self.x_list[i] += random.uniform(-2, 2)
            self.y_list[i] += random.uniform(-2, 2)
            if self.x_list[i] < x_min:
                self.x_list[i] = x_min
            if self.x_list[i] > x_max:
                self.x_list[i] = x_max
            if self.y_list[i] < y_min:
                self.y_list[i] = y_min
            if self.y_list[i] > y_max:
                self.y_list[i] = y_max
        self.x = sum(self.x_list) / 3
        self.y = sum(self.y_list) / 3

    def draw(self, ax: plt.Axes):
        p = Polygon(xy=list(zip(self.x_list, self.y_list)))
        ax.add_patch(p)
        p.set_facecolor("w")


class MyRectangle(VideoObject):
    def __init__(
        self,
        x,
        y,
        width,
        height,
        angle,
        max_length=12,
        min_length=2,
        color="w",
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
        self.max_length = max_length
        self.min_length = min_length
        self.color = color
        self.name = "rectangle"

    def random_move(self):
        self.x += random.uniform(-2, 2)
        self.y += random.uniform(-2, 2)
        self.width += random.uniform(-2, 2)
        self.height += random.uniform(-2, 2)
        self.angle += random.uniform(-10, 10)
        min_length = self.min_length
        max_length = self.max_length
        if self.width < min_length:
            self.width = min_length
        if self.height < min_length:
            self.height = min_length
        if self.width > max_length:
            self.width = max_length
        if self.height > max_length:
            self.height = max_length
        x_min = -8
        x_max = 8
        y_min = -8
        y_max = 8
        if self.x < x_min:
            self.x = x_min
        if self.x > x_max:
            self.x = x_max
        if self.y < y_min:
            self.y = y_min
        if self.y > y_max:
            self.y = y_max

    def draw(self, ax: plt.Axes):
        r = Rectangle(
            xy=(self.x, self.y),
            width=self.width,
            height=self.height,
            angle=self.angle,
        )
        ax.add_patch(r)
        r.set_facecolor(self.color)


class MyArrow(VideoObject):
    def __init__(self, x, y, dx, dy, width=2.0):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.width = width
        self.name = "arrow"

    def random_move(self):
        self.x += random.uniform(-2, 2)
        self.y += random.uniform(-2, 2)
        self.dx += random.uniform(-2, 2)
        self.dy += random.uniform(-2, 2)
        self.width += random.uniform(-1, 1)
        x_min = -8
        x_max = 8
        y_min = -8
        y_max = 8
        width_min = 2
        width_max = 10
        if self.width < width_min:
            self.width = width_min
        if self.width > width_max:
            self.width = width_max
        if self.x < x_min:
            self.x = x_min
        if self.x > x_max:
            self.x = x_max
        if self.y < y_min:
            self.y = y_min
        if self.y > y_max:
            self.y = y_max
        if self.dx < x_min:
            self.dx = x_min
        if self.dx > x_max:
            self.dx = x_max
        if self.dy < y_min:
            self.dy = y_min
        if self.dy > y_max:
            self.dy = y_max

    def draw(self, ax: plt.Axes):
        a = Arrow(
            x=self.x,
            y=self.y,
            dx=self.dx,
            dy=self.dy,
            width=self.width,
        )
        ax.add_patch(a)
        a.set_facecolor("w")


class MyFancyBbox(VideoObject):
    def __init__(
        self,
        x,
        y,
        width,
        height,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = "fancybbox"

    def random_move(self):
        min_length = 2
        max_length = 12
        x_min = -8
        x_max = 8
        y_min = -8
        y_max = 8
        self.x += random.uniform(-2, 2)
        self.y += random.uniform(-2, 2)
        self.width += random.uniform(-2, 2)
        self.height += random.uniform(-2, 2)
        if self.width < min_length:
            self.width = min_length
        if self.height < min_length:
            self.height = min_length
        if self.width > max_length:
            self.width = max_length
        if self.height > max_length:
            self.height = max_length
        if self.x < x_min:
            self.x = x_min
        if self.x > x_max:
            self.x = x_max
        if self.y < y_min:
            self.y = y_min
        if self.y > y_max:
            self.y = y_max

    def draw(self, ax: plt.Axes):
        f = FancyBboxPatch(xy=(self.x, self.y), width=self.width, height=self.height)
        ax.add_patch(f)
        f.set_facecolor("w")


class VideoRenderer:
    def __init__(self, root="model_name") -> None:
        self.root = root
        self.objects: list[VideoObject] = []
        self.video_path_dict: dict[str, list[str]] = {}
        self.objects.append(
            MyEllipse(
                x=0,
                y=0,
                width=4,
                height=4,
                angle=20,
            )
        )
        self.objects.append(
            MyTriangle(
                x_list=[0, 4, -4],
                y_list=[0, 4, 4],
            )
        )
        self.objects.append(
            MyRectangle(
                x=0,
                y=0,
                width=4,
                height=4,
                angle=20,
            )
        )
        self.objects.append(
            MyArrow(
                x=0,
                y=0,
                dx=4,
                dy=4,
                width=8,
            )
        )
        self.objects.append(
            MyFancyBbox(
                x=0,
                y=0,
                width=4,
                height=4,
            )
        )
        self.occlusion_object = MyRectangle(
            x=-1,
            y=-6,
            width=0.8,
            height=0.8,
            angle=0,
            max_length=2,
            min_length=0.8,
            color="b",
        )

    def render_one_video(
        self,
        video_object: VideoObject,
        occulusion: bool = False,
    ):
        if occulusion:
            video_name = video_object.name + "_occlusion"
            self.occlusion_object.x = video_object.x
            self.occlusion_object.y = video_object.y
        else:
            video_name = video_object.name
        path = f"./tmp_videos/{self.root}/{video_name}"
        self.video_path_dict[video_name] = []

        os.makedirs(path, exist_ok=True)
        for i in range(20):
            plt.figure()
            ax = plt.gca()
            plt.style.use("dark_background")
            video_object.random_move()
            video_object.draw(ax)
            if occulusion:
                self.occlusion_object.random_move()
                self.occlusion_object.draw(ax)
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.axis("off")
            save_path = f"{path}/{i:02d}.png"
            self.video_path_dict[video_name].append(save_path)
            plt.savefig(save_path)
            plt.close()

    def render_all_video(self):
        for video_object in self.objects:
            self.render_one_video(video_object, occulusion=False)
            self.render_one_video(video_object, occulusion=True)
