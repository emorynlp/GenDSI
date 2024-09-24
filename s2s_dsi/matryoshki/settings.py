
import typing as T
import inspect


F = T.TypeVar('F', bound=T.Callable[..., T.Any])


def no_leading_underscore(s): return not s.startswith('_')
def all_parameters(s): return True


def settings(
    signature: inspect.Signature = None,
    filter=no_leading_underscore,
    /, **parameters_on_or_off
):
    return SettingsDecorator(signature, filter, **parameters_on_or_off)


class SettingsDecorator:

    def __init__(self,
        signature: inspect.Signature = None,
        filter=no_leading_underscore,
        /, **parameters_on_or_off
    ):
        self.filter = filter
        self.signature = signature
        self.parameters_on_or_off = parameters_on_or_off

    def __call__(self, fn: F) -> F:
        return SettingsDecorated(
            fn, self.signature, self.filter, self.parameters_on_or_off
        ) # noqa


class SettingsDecorated:

    def __init__(self, fn, signature, filter, parameters_on_or_off):
        self.fn = fn
        self.filter = filter or all_parameters
        self.parameters_on_or_off = parameters_on_or_off
        self.signature = inspect.signature(fn) if signature is None else signature
        self.defaults = {}
        for name, parameter in self.signature.parameters.items():
            if parameter.default is not inspect.Parameter.empty and name != 'settings':
                if self.parameters_on_or_off.get(name):
                    self.defaults[name] = parameter.default
                elif self.filter(name) and self.parameters_on_or_off.get(name, True):
                    self.defaults[name] = parameter.default
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                self.defaults[name] = {}
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                self.defaults[name] = ()

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        specified = {k: v for k, v in bound.arguments.items()}
        settings_index = None
        if 'settings' in specified:
            specified.pop('settings')
            if 'settings' in kwargs:
                kwargs.pop('settings')
            else:
                settings_index = 0
                for name in self.signature.parameters:
                    if name == 'settings':
                        break
                    else:
                        settings_index += 1
                args = args[:settings_index] + args[settings_index + 1:]
        defaults = self.defaults
        bound.apply_defaults()
        arguments = bound.arguments
        settings = SettingsDict()
        settings.defaults.update(defaults)
        settings.args = bound.args
        if settings_index is not None:
            settings.args = settings.args[:settings_index] + settings.args[settings_index + 1:]
        settings.kwargs = bound.kwargs
        if 'settings' in settings.kwargs:
            settings.kwargs.pop('settings')
        for name, value in arguments.items():
            if self.parameters_on_or_off.get(name):
                settings[name] = value
                if name in specified:
                    settings.specified[name] = value
            elif self.filter(name) and self.parameters_on_or_off.get(name, True):
                settings[name] = value
                if name in specified:
                    settings.specified[name] = value
            else:
                if name in settings.kwargs:
                    settings.kwargs.pop(name)
        return self.fn(*args, settings=settings, **kwargs)


class SettingsDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defaults = {}
        self.specified = {}
        self.args = ()
        self.kwargs = {}
        if args and isinstance(args[0], SettingsDict):
            self.defaults = args[0].defaults
            self.specified = args[0].specified
            self.args = args[0].args
            self.kwargs = args[0].kwargs

    def __str__(self):
        return f'{self.__class__.__name__}({str(dict(self))})'

    def __repr__(self):
        return str(self)



if __name__ == '__main__':

    @settings(b=False)
    def foo(a, b, c=3, _d=4, settings=None, **additional):
        print(f'{settings=}')
        print(f'{settings.__class__.__bases__=}')
        print(f'{dict(settings)=}')
        print(f'{settings.specified=}')
        print(f'{settings.defaults=}')
        print(f'{settings.args=}')
        print(f'{settings.kwargs=}')
        return settings

    settings = foo(1, 2, x=8, y=9)
    print('\nAnother call using settings.args and settings.kwargs...')
    another = foo(*settings.args, **settings.kwargs)

