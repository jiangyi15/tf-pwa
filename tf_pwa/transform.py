from .config import create_config

set_trans, get_trans, register_trans = create_config()

T = "Tensor"


class BaseTransform:
    def __call__(self, x: T) -> T:
        return self.call(x)

    def call(self, x: T) -> T:
        raise NotImplementedError()

    def inverse(self, y: T) -> T:
        raise NotImplementedError()


def create_trans(item: dict) -> BaseTransform:
    model = item.pop("model", "default")
    cls = get_trans(model)
    obj = cls(**item)
    obj._model_name = model
    return obj


@register_trans("default")
@register_trans("linear")
class LinearTrans(BaseTransform):
    def __init__(self, k: float = 1.0, b: float = 0.0, **kwargs):
        self.k = k
        self.b = b

    def call(self, x: T) -> T:
        return self.k * x + self.b

    def inverse(self, x: T) -> T:
        return (x - self.b) / self.k
