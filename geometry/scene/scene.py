class Scene:
    def __init__(self):
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)
        return self

    def visualise(self, backend="opencv", **style):
        canvas = style.pop("image", None)
        for obj in self.objects:
            canvas = obj.visualise(
                backend=backend,
                image=canvas,
                **style,
            )
        return canvas
