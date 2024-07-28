import os
import subprocess

# Set the environment variable temporarily
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"

# Install submodlib
subprocess.run(["pip", "install", "-i", "https://test.pypi.org/simple/", "--extra-index-url", "https://pypi.org/simple/", "submodlib"])