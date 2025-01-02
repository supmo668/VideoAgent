"""Video player app with frame counter."""

import reflex as rx
from .state import State
from .pages.index import index

app = rx.App()
app.add_page(index)
