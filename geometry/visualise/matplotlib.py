import matplotlib.pyplot as plt

class MatplotlibRenderer:
    name = "matplotlib"

    def draw_point(self, point, ax=None, color="red", size=50, **_):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(point.x, point.y, c=color, s=size)
        ax.set_aspect("equal")
        return ax

    def draw_line(self, line, ax=None, color="blue", linewidth=2, **_):
        if ax is None:
            fig, ax = plt.subplots()
        xs = [line.start.x, line.end.x]
        ys = [line.start.y, line.end.y]
        ax.plot(xs, ys, color=color, linewidth=linewidth)
        ax.set_aspect("equal")
        return ax

    def draw_polygon(self, polygon, ax=None, color="green", fill=False, **_):
        if ax is None:
            fig, ax = plt.subplots()
        xs = [p.x for p in polygon.points] + [polygon.points[0].x]
        ys = [p.y for p in polygon.points] + [polygon.points[0].y]

        if fill:
            ax.fill(xs, ys, color=color, alpha=0.3)
        else:
            ax.plot(xs, ys, color=color)

        ax.set_aspect("equal")
        return ax
