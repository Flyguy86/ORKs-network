import random
from math import ceil
from textwrap import shorten

from dash import Dash, html, dcc, Input, Output, State
import dash_cytoscape as cyto

# Generate test graph: up to 7 layers deep. Top layer has 10 nodes.
def generate_test_graph(layers=7, top_nodes=10, min_branch=3, max_branch=5, max_nodes=1000, seed=42):
    random.seed(seed)
    nodes = []
    edges = []

    node_counter = 0
    # create top layer
    current_layer_nodes = []
    for i in range(top_nodes):
        nid = f"L0-{i}"
        title = f"TopNode {i} - layer 0"
        full = f"{title} - detailed text. " + ("More details. " * 20)
        nodes.append({
            "data": {
                "id": nid,
                "label": shorten(title, width=200, placeholder="..."),
                "full_text": full,
                "layer": 0
            }
        })
        current_layer_nodes.append(nid)
        node_counter += 1

    # build subsequent layers
    for layer in range(1, layers):
        next_layer_nodes = []
        for parent in current_layer_nodes:
            # generate between min_branch and max_branch children but never exceed max_nodes
            if node_counter >= max_nodes:
                break
            n_children = random.randint(min_branch, max_branch)
            for c in range(n_children):
                if node_counter >= max_nodes:
                    break
                nid = f"L{layer}-{len(next_layer_nodes)}-{node_counter}"
                title = f"Node {nid} (layer {layer})"
                full = f"{title} - long detailed description. " + ("Extra info. " * 30)
                nodes.append({
                    "data": {
                        "id": nid,
                        "label": shorten(title, width=200, placeholder="..."),
                        "full_text": full,
                        "layer": layer
                    }
                })
                edges.append({
                    "data": {"source": parent, "target": nid}
                })
                next_layer_nodes.append(nid)
                node_counter += 1
        if not next_layer_nodes:
            break
        current_layer_nodes = next_layer_nodes

    # also add any edges between top nodes randomly (to show network edges)
    # connect some top nodes together
    top_ids = [n["data"]["id"] for n in nodes if n["data"]["layer"] == 0]
    for i in range(min(len(top_ids) - 1, 10)):
        if random.random() < 0.3:
            edges.append({"data": {"source": top_ids[i], "target": top_ids[(i + 1) % len(top_ids)]}})

    elements = nodes + edges
    return elements, max([n["data"]["layer"] for n in nodes])

elements_all, max_layer = generate_test_graph(
    layers=7, top_nodes=10, min_branch=3, max_branch=5, max_nodes=1000, seed=1234
)

# color palette for layers (cycle if needed)
palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-wrap": "wrap",
            "text-max-width": 150,
            "text-valign": "center",
            "text-halign": "center",
            "font-size": 10,
            "width": "label",
            "height": "label",
            "padding": "8px",
            "background-color": "#97C2FC",
            "shape": "roundrectangle",
            "border-width": 1,
            "border-color": "#666"
        }
    },
    {
        "selector": "edge",
        "style": {
            "width": 1,
            "line-color": "#999",
            "target-arrow-color": "#999",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier"
        }
    }
]

# add layer-specific styling
for layer in range(max_layer + 1):
    stylesheet.append({
        "selector": f'[layer = "{layer}"]',
        "style": {
            "background-color": palette[layer % len(palette)],
            "border-color": palette[layer % len(palette)]
        }
    })

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3("Zoomable Layered Network (click nodes for details)"),
        html.Div(id="zoom-info", style={"marginBottom": "6px"}),
    ]),
    # Cytoscape container: relative so we can overlay the floating button
    html.Div([
        cyto.Cytoscape(
            id="cyt",
            elements=elements_all,
            stylesheet=stylesheet,
            style={"width": "100%", "height": "700px", "border": "1px solid #ccc"},
            layout={"name": "cose", "animate": True, "randomize": True},
            zoom=1,
        ),
        # Floating button over the canvas (top-right)
        html.Button(
            "Show all layers",
            id="show-all-btn",
            n_clicks=0,
            style={
                "position": "absolute",
                "top": "12px",
                "right": "12px",
                "zIndex": 9999,
                "padding": "8px 12px",
                "backgroundColor": "#fff",
                "border": "1px solid #444",
                "borderRadius": "4px",
                "cursor": "pointer"
            }
        ),
    ], style={"width": "75%", "display": "inline-block", "verticalAlign": "top", "position": "relative"}),
    html.Div([
        html.H4("Node details"),
        html.Div(id="node-details", style={"whiteSpace": "pre-wrap", "fontSize": 14}),
        html.H4("Controls"),
        html.Div([
            html.Label("Max nodes (regenerate)"),
            dcc.Input(id="max-nodes", type="number", value=1000, min=100, step=100),
            html.Button("Regenerate", id="regen", n_clicks=0)
        ], style={"marginTop": "12px"})
    ], style={"width": "24%", "display": "inline-block", "paddingLeft": "12px", "verticalAlign": "top"}),
    # store full elements so we can filter without losing metadata
    dcc.Store(id="all-elements-store", data=elements_all),
    dcc.Store(id="max-layer-store", data=max_layer),
    # store for 'show all' toggle
    dcc.Store(id="show-all-store", data=False),
])


def zoom_to_visible_layer(zoom, max_layer):
    # Map zoom to how many layers to reveal.
    # heuristic: zoom ranges typically 0.2 - 3.0; scale into [0..max_layer]
    if zoom is None:
        return 0
    z = float(zoom)
    # Normalize and clamp
    min_z, max_z = 0.2, 3.0
    z = max(min_z, min(max_z, z))
    frac = (z - min_z) / (max_z - min_z)  # 0..1
    layers_shown = int(round(frac * (max_layer)))  # 0..max_layer
    return layers_shown


@app.callback(
    Output("cyt", "elements"),
    Output("zoom-info", "children"),
    Input("cyt", "viewport"),
    Input("show-all-store", "data"),
    State("all-elements-store", "data"),
    State("max-layer-store", "data"),
)
def filter_elements(viewport, show_all, all_elements, max_layer_store):
    # viewport: dict with 'zoom' and 'pan'
    # show_all: boolean from the floating button toggle (when True, ignore zoom)
    if show_all:
        # show everything
        node_ids = set()
        filtered_nodes = []
        filtered_edges = []
        for el in all_elements:
            if "source" in el.get("data", {}):
                filtered_edges.append(el)
            else:
                filtered_nodes.append(el)
                node_ids.add(el["data"]["id"])
        msg = f"Show-all ON — showing all layers (nodes: {len(filtered_nodes)}, edges: {len(filtered_edges)})"
        return filtered_nodes + filtered_edges, msg

    zoom = viewport.get("zoom") if viewport else None
    visible_layer = zoom_to_visible_layer(zoom, max_layer_store)
    # keep nodes with layer <= visible_layer
    node_ids = set()
    filtered_nodes = []
    for el in all_elements:
        if "source" in el.get("data", {}):
            continue
        layer = el["data"].get("layer", 0)
        if layer <= visible_layer:
            filtered_nodes.append(el)
            node_ids.add(el["data"]["id"])
    # keep edges where both nodes visible
    filtered_edges = []
    for el in all_elements:
        if "source" in el.get("data", {}):
            s = el["data"]["source"]
            t = el["data"]["target"]
            if s in node_ids and t in node_ids:
                filtered_edges.append(el)
    msg = f"Zoom: {zoom:.2f} — showing layers 0..{visible_layer} (nodes: {len(filtered_nodes)}, edges: {len(filtered_edges)})"
    return filtered_nodes + filtered_edges, msg


@app.callback(
    Output("node-details", "children"),
    Input("cyt", "tapNodeData")
)
def show_node_details(data):
    if not data:
        return "Click a node to see full details."
    title = data.get("label", "")
    full = data.get("full_text", "")
    layer = data.get("layer", "")
    return f"ID: {data.get('id')}\nLayer: {layer}\nTitle: {title}\n\n{full}"


# toggle callback: clicking the floating button toggles the show-all state and updates label
@app.callback(
    Output("show-all-store", "data"),
    Output("show-all-btn", "children"),
    Input("show-all-btn", "n_clicks"),
    State("show-all-store", "data"),
)
def toggle_show_all(n_clicks, current):
    if not n_clicks:
        # initial state
        return False, "Show all layers"
    on = (n_clicks % 2 == 1)
    label = "Showing all layers (ON)" if on else "Show all layers"
    return on, label


@app.callback(
    Output("all-elements-store", "data"),
    Output("max-layer-store", "data"),
    Input("regen", "n_clicks"),
    State("max-nodes", "value")
)
def regenerate(n, max_nodes):
    # regenerate test graph with new max_nodes cap
    elements, max_layer = generate_test_graph(
        layers=7, top_nodes=10, min_branch=3, max_branch=5, max_nodes= int(max_nodes or 1000), seed=random.randint(0,9999)
    )
    return elements, max_layer


if __name__ == "__main__":
    print("Starting Dash app on http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050, host="0.0.0.0")