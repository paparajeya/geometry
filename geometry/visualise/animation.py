def animate(
    obj,
    transforms,
    frames=60,
    backend="opencv",
    **style,
):
    """
    Generator yielding rendered frames.
    """
    current = obj
    for t in transforms:
        current = t(current)
        yield current.visualise(backend=backend, **style)
