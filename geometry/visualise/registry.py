class VisualisationRegistry:
    _renderers = {}

    @classmethod
    def register(cls, renderer):
        cls._renderers[renderer.name] = renderer

    @classmethod
    def get(cls, name: str):
        if name not in cls._renderers:
            raise ValueError(f"Renderer '{name}' not registered.")
        return cls._renderers[name]
