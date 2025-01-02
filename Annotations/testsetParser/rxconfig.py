import reflex as rx
import sys, os
from _PATH import EXTRA_PATHS

# Add extra paths to sys.path
sys.path.extend(EXTRA_PATHS)
print("App root:",  os.getcwd())
config = rx.Config(
    app_name="testsetparser",
    tailwind={
        "theme": {
            "extend": {},
        },
        "plugins": ["@tailwindcss/typography"],
    },
    frontend_packages=[
        "@fontsource/inter",
    ],
)