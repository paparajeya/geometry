from .registry import VisualisationRegistry

try:
    from .opencv import OpenCVRenderer
    VisualisationRegistry.register(OpenCVRenderer())
except ModuleNotFoundError:
    pass

try:
    from .matplotlib import MatplotlibRenderer
    VisualisationRegistry.register(MatplotlibRenderer())
except ModuleNotFoundError:
    pass