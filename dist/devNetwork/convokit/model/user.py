from .speaker import Speaker
from convokit.util import deprecation


class User(Speaker):
    def __init__(self, *args, **kwargs):
        deprecation("The User class", "the Speaker class")
        super().__init__(*args, **kwargs)