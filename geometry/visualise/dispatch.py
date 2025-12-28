from .registry import VisualisationRegistry

def visualise(obj, backend: str = "opencv", **style):
    renderer = VisualisationRegistry.get(backend)
    geom_type = obj.geometry_type()

    method_name = f"draw_{geom_type}"
    if not hasattr(renderer, method_name):
        raise NotImplementedError(
            f"{backend} does not support '{geom_type}'"
        )

    return getattr(renderer, method_name)(obj, **style)
