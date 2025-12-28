import numpy as np
import cv2

class OpenCVRenderer:
    name = "opencv"

    def _ensure_canvas(self, image, size):
        if image is None:
            return np.zeros((*size, 3), dtype=np.uint8)
        return image

    def draw_point(
        self,
        point,
        image=None,
        radius=3,
        color=(0, 255, 0),
        thickness=-1,
        canvas_size=(512, 512),
        **_,
    ):
        image = self._ensure_canvas(image, canvas_size)
        cv2.circle(image, (int(point.x), int(point.y)),
                   radius, color, thickness)
        return image

    def draw_line(
        self,
        line,
        image=None,
        color=(255, 0, 0),
        thickness=2,
        canvas_size=(512, 512),
        **_,
    ):
        image = self._ensure_canvas(image, canvas_size)
        p1, p2 = line.start, line.end
        cv2.line(
            image,
            (int(p1.x), int(p1.y)),
            (int(p2.x), int(p2.y)),
            color,
            thickness,
        )
        return image

    def draw_polygon(
        self,
        polygon,
        image=None,
        color=(0, 255, 0),
        thickness=2,
        fill=False,
        canvas_size=(512, 512),
        **_,
    ):
        image = self._ensure_canvas(image, canvas_size)
        pts = np.array(
            [[int(p.x), int(p.y)] for p in polygon.points],
            np.int32,
        )

        if fill:
            cv2.fillPoly(image, [pts], color)
        else:
            cv2.polylines(image, [pts], True, color, thickness)

        return image
