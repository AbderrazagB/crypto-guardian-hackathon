"""Export static HTML while preserving the original executed notebook.

Some upstream progress bars leave incomplete widget-state metadata. Static
reports omit widget views but retain tables, text, and scientific figures.
"""

from copy import deepcopy
from nbconvert import HTMLExporter


def export_html(notebook):
    static = deepcopy(notebook)
    static.metadata.pop("widgets", None)
    for cell in static.cells:
        if cell.cell_type != "code":
            continue
        retained = []
        for output in cell.outputs:
            if "data" in output:
                output.data.pop("application/vnd.jupyter.widget-view+json", None)
                if not output.data:
                    continue
            retained.append(output)
        cell.outputs = retained
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(static)
    return html
