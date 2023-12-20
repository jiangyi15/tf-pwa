from .config import create_config

set_trans, get_trans, register_trans = create_config()

T = "Tensor"


class BaseTransform:
    def __init__(self, x: "list | str", **kwargs):
        self.x = x

    def __call__(self, dic: dict) -> T:
        x = self.read(dic)
        return self.call(x)

    def read(self, x: dict) -> T:
        if isinstance(self.x, (list, tuple)):
            return [x[i] for i in self.x]
        elif isinstance(self.x, str):
            return x[self.x]
        else:
            raise TypeError("only str of list of str is supported for x")

    def call(self, x: T) -> T:
        raise NotImplementedError()

    def inverse(self, y: T) -> T:
        return None


def create_trans(item: dict) -> BaseTransform:
    model = item.pop("model", "default")
    cls = get_trans(model)
    obj = cls(**item)
    obj._model_name = model
    return obj


@register_trans("default")
@register_trans("linear")
class LinearTrans(BaseTransform):
    def __init__(
        self, x: "list | str", k: float = 1.0, b: float = 0.0, **kwargs
    ):
        super().__init__(x)
        self.k = k
        self.b = b

    def call(self, x) -> T:
        return self.k * x + self.b

    def inverse(self, x: T) -> T:
        return (x - self.b) / self.k
