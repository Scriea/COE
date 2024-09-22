import os

ROOT = os.path.abspath(
    os.path.join((os.path.dirname(os.path.relpath(__file__))), "../../")
)
TEMP_FILE = os.path.join(ROOT, "temp")
print(TEMP_FILE)
