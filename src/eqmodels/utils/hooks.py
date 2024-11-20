from dataclasses import dataclass, fields, make_dataclass

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ObservationHook(nn.Module):
    """Class to pass intermediate results to Hook.

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self):
        super().__init__()

    def forward(self, **kwargs) -> None:
        """Hooks will catch kwargs.

        Returns
        -------
        None
        """
        del kwargs


class InterventionHook(nn.Module):
    def __init__(self):
        super().__init__()

        def default_function(**kwargs):
            return kwargs["before"]

        self.function = default_function

    def set_function(self, function):
        self.function = function

    def forward(self, **kwargs):
        return self.function(**kwargs)


class Hook:
    """Base class for hooks."""

    hook: RemovableHandle

    def __init__(self, module: nn.Module, result_class) -> None:
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)
        self.result = None
        self.result_class = result_class

    def hook_fn(self, module, args, kwargs, output) -> None:
        """Hook function to catch attention weights."""
        del module, args, output
        self.result = self.result_class(**kwargs)

    def remove(self):
        """Remove the hook."""
        self.hook.remove()


@dataclass
class AbstractResult:
    def __repr__(self):
        msg = self.__class__.__name__ + ":\n"
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                msg += f"\t{k}: {v.shape}\n"
            elif isinstance(v, AbstractResult):
                msg += f"\t{k}: {v.__class__.__name__}\n"
            else:
                msg += f"\t{k}: {v}\n"
        return msg

    def __init__(self, **kwargs):
        # Get the field names from the dataclass
        field_names = {f.name for f in fields(self.__class__)}
        for key, value in kwargs.items():
            if key in field_names:
                setattr(self, key, value)
        # Handle ignored/unexpected keys
        ignored_keys = set(kwargs) - field_names
        if ignored_keys:
            print(f"Ignored unexpected keys: {ignored_keys}")

    @classmethod
    def init_all(cls, **kwargs):
        for key, value in kwargs.items():
            setattr(cls, key, value)


@dataclass(repr=False, init=False)
class AbstractBatchResult(AbstractResult):
    def unbatch(self) -> list[AbstractResult]:
        for k, v in self.__dict__.items():
            if isinstance(v, AbstractBatchResult):
                setattr(self, k, v.unbatch())
            elif isinstance(v, (torch.Tensor | list)):
                continue
            else:
                raise ValueError(f"Unexpected type: {type(v)}")

        results = []
        new_class_name = self.__class__.__name__.replace("Batch", "")
        new_class_fields = [(f.name, f.type) for f in fields(self)]
        new_class = make_dataclass(
            new_class_name,
            fields=new_class_fields,
            bases=(AbstractResult,),
            repr=False,
            init=False,
        )

        for i in range(self.get_batch_size()):
            results.append(new_class(**{k: v[i] for k, v in self.__dict__.items()}))

        return results

    def get_batch_size(self) -> int:
        """Get the batch size."""
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
            elif isinstance(v, AbstractBatchResult):
                return v.get_batch_size()
