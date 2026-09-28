"""Package that holds the compiled `exllamav3_ext` HIP extension module.

`exllamav3_ext` (the .so built by setup.py) is a direct submodule of this package, so
`from exllamav3.ext import exllamav3_ext` resolves to it. No re-export is needed: Python's import
machinery finds the extension module by path.
"""
