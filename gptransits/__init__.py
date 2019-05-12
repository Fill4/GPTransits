
__version__ = 0.3

from .model import Model
from.settings import Settings
from .helper import analyse

# Analysis is still not working with the default settings
# Should be a class that recieves a model and analyses all the data
__all__ = [Model, Settings, analyse]