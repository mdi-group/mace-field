project = "MACE-Field"
author = "MACE-Field contributors"
copyright = "2026, MACE-Field contributors"
release = "1.1.0"

extensions = ["myst_parser", "sphinx.ext.mathjax"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

html_theme = "sphinx_rtd_theme"
html_title = "MACE-Field documentation"
html_logo = "../macefield_logo.png"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
