project = "MACE-Field"
copyright = "2026, MACE-Field contributors"
author = "MACE-Field contributors"
release = "1.0.2"

extensions = ["sphinx.ext.mathjax"]
root_doc = "index"
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "MACE-Field documentation"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}
