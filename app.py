import random
from math import ceil
from textwrap import shorten
import json

import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_cytoscape as cyto

# Generate test graph: up to 7 layers deep. Top layer has 10 nodes.
def generate_test_graph(layers=7, top_nodes=10, min_branch=3, max_branch=5, max_nodes=1000, seed=42, max_columns=6):
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
    top_ids = [n["data"]["id"] for n in nodes if n["data"]["layer"] == 0]
    for i in range(min(len(top_ids) - 1, 10)):
        if random.random() < 0.3:
            edges.append({"data": {"source": top_ids[i], "target": top_ids[(i + 1) % len(top_ids)]}})

    # assign preset positions per-layer: keep tree structure (parent centroids) and compact deeper layers
    layers_map = {}
    for n in nodes:
        layer = n["data"].get("layer", 0)
        layers_map.setdefault(layer, []).append(n)

    spacing_y = 140   # vertical spacing between layers (tune)
    stagger_amount = 20  # small vertical stagger for alternating nodes

    # make layer 0 wider by a multiplier and keep it un-staggered
    top_spacing_multiplier = 2.0  # larger horizontal spacing for layer 0

    # compute a spacing_x based on the maximum nodes in any non-top layer to keep total width bounded
    max_count = max((len(lst) for lst in layers_map.values()), default=1)
    desired_total_width = 1200  # target width for the widest row (pixels)
    spacing_x = max(60, desired_total_width / max_count)  # clamp minimum spacing

    # build parent map from edges for tree-aware layout
    parent_map = {}
    for e in edges:
        src = e["data"]["source"]; tgt = e["data"]["target"]
        parent_map.setdefault(tgt, []).append(src)

    # helper map for quick id->node element
    node_map = {n["data"]["id"]: n for n in nodes}

    # 1) First position layer-0 nodes evenly (they will later be adjusted by adjust_layer0_positions using descendants)
    top_nodes = [n for n in layers_map.get(0, [])]
    if top_nodes:
        cols = len(top_nodes)
        spacing_x_layer = spacing_x * top_spacing_multiplier
        width = (cols - 1) * spacing_x_layer
        x_offset = -width / 2
        base_y = 0 * spacing_y
        for idx, node in enumerate(sorted(top_nodes, key=lambda e: e["data"]["id"])):
            x = x_offset + idx * spacing_x_layer
            node["position"] = {"x": x, "y": base_y}

    # 2) For each deeper layer, position each node under the centroid of its parents (if available).
    #    When many siblings collapse to the same x, we no longer force extra spacing or greedy nudges.
    for layer in sorted([l for l in layers_map.keys() if l > 0]):
        row_nodes = layers_map.get(layer, [])
        if not row_nodes:
            continue

        # for fallback distribution (when parent positions are missing), compute a centered band
        count = len(row_nodes)
        band_width = max(spacing_x * count * 0.8, spacing_x * count)
        start_x = -band_width / 2.0

        # compute initial x aiming to cluster siblings of same parent near parent's x
        x_assigned = {}

        # build parent -> children in this layer mapping (only include parents with positions)
        parent_children = {}
        orphan_candidates = []  # nodes without any positioned parent
        for node in sorted(row_nodes, key=lambda e: e["data"]["id"]):
            nid = node["data"]["id"]
            parents = [p for p in parent_map.get(nid, []) if p in node_map and node_map[p].get("position")]
            if not parents:
                orphan_candidates.append(nid)
            else:
                # record under each positioned parent so we can cluster siblings
                for p in parents:
                    parent_children.setdefault(p, []).append(nid)
                # initial x = centroid of parent positions (useful for multi-parent cases)
                xs = [node_map[p]["position"]["x"] for p in parents]
                x_assigned[nid] = sum(xs) / len(xs)

        # For parents that have children in this row, layout their children contiguously centered on parent's x
        for p, kids in parent_children.items():
            kids_unique = sorted(dict.fromkeys(kids))  # stable unique list
            parent_x = node_map[p]["position"]["x"]
            # simple contiguous placement with small constant spacing between siblings (no aggressive widening)
            delta = spacing_x * 0.5
            start_group_x = parent_x - (delta * (len(kids_unique) - 1)) / 2.0
            for i, k in enumerate(kids_unique):
                x_assigned[k] = start_group_x + i * delta

        # assign nodes without positioned parents (orphans) evenly across the band
        if orphan_candidates:
            for i, nid in enumerate(orphan_candidates):
                if len(orphan_candidates) > 1:
                    x_assigned[nid] = start_x + (i * (band_width / max(1, len(orphan_candidates) - 1)))
                else:
                    x_assigned[nid] = 0.0

        # ensure every node has an x (fallback)
        for idx, node in enumerate(sorted(row_nodes, key=lambda e: e["data"]["id"])):
            nid = node["data"]["id"]
            if nid not in x_assigned:
                if count > 1:
                    x_assigned[nid] = start_x + (idx * (band_width / max(1, count - 1)))
                else:
                    x_assigned[nid] = 0.0

        # assign final positions (no extra spreading / greedy overlap enforcement)
        ordered_nodes = sorted(row_nodes, key=lambda e: e["data"]["id"])

        # prepare 4-way horizontal stagger pattern (multipliers)
        stagger_multipliers = [-1.5, -0.5, 0.5, 1.5]
        # small base stagger distance relative to spacing_x so offsets remain subtle
        base_stagger = spacing_x * 0.12

        for i, node in enumerate(ordered_nodes):
            nid = node["data"]["id"]
            x = x_assigned.get(nid, start_x + i * (band_width / max(1, count - 1)))
            # apply 4-way horizontal stagger pattern
            x += stagger_multipliers[i % 4] * base_stagger
            y = layer * spacing_y
            # keep a small alternating vertical nudge for readability, but no horizontal forcing
            y += (stagger_amount if (i % 2 == 0) else -stagger_amount)
            node["position"] = {"x": x, "y": y}

    # --- new: bottom-up pass to center parents over their grouped children (only for layers >= 1) ---
    max_layer = max((n["data"].get("layer", 0) for n in nodes), default=0)
    # iterate from deepest non-top layer up to layer 1 (do not re-center layer 0 here)
    for layer in range(max_layer - 1, 0, -1):
        layer_nodes = layers_map.get(layer, [])
        if not layer_nodes:
            continue

        # set parent x to centroid of direct children's x positions (only consider children in deeper layers)
        for parent_node in layer_nodes:
            pid = parent_node["data"]["id"]
            # direct children (from edges)
            child_ids = [e["data"]["target"] for e in edges if e["data"]["source"] == pid]

            # only consider positioned children that are in a deeper layer than the parent
            positioned_children = []
            for cid in child_ids:
                child = node_map.get(cid)
                if not child:
                    continue
                child_layer = child["data"].get("layer", 0)
                # require child_layer > parent_layer to avoid using siblings or same-layer nodes
                if child_layer <= layer:
                    continue
                if child.get("position"):
                    positioned_children.append(cid)

            if positioned_children:
                xs = [node_map[cid]["position"]["x"] for cid in positioned_children]
                centroid_x = sum(xs) / len(xs)
                # preserve parent's y if set, otherwise set based on layer
                py = parent_node.get("position", {}).get("y", layer * spacing_y)
                parent_node["position"] = {"x": centroid_x, "y": py}

    # NOTE: we intentionally do NOT run a greedy non-overlap pass here.
    # This keeps the centroid-based tree structure intact and avoids adding extra spacing.

    elements = nodes + edges
    max_layer = max([n["data"]["layer"] for n in nodes])
    return elements, max_layer

elements_all, max_layer = generate_test_graph(
    layers=7, top_nodes=10, min_branch=3, max_branch=5, max_nodes=1000, seed=1234, max_columns=5
)

# compute top/root ids for tree-like layout
top_ids = [n["data"]["id"] for n in elements_all if n.get("data", {}).get("layer") == 0]

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
            "text-max-width": 120,
            "text-valign": "center",
            "text-halign": "center",
            "font-size": 9,
            "width": "label",
            "height": "label",
            "padding": "4px",
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
    layer_style = {
        "background-color": palette[layer % len(palette)],
        "border-color": palette[layer % len(palette)]
    }
    # make layer 0 visually prominent: larger font + padding
    if layer == 0:
        layer_style.update({
            "font-size": "12px",
            "padding": "6px",
            "width": "label",
            "height": "label"
        })

    stylesheet.append({
        "selector": f'[layer = "{layer}"]',
        "style": layer_style
    })

app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H3("Zoomable Layered Network (click nodes for details)"),
        html.Div(id="zoom-info", style={"marginBottom": "6px"}),
    ]),
    # Main row: center graph and right details (left title column removed)
    html.Div([
        # Cytoscape container: relative so we can overlay the floating button and manual zoom controls
        html.Div([
            cyto.Cytoscape(
                id="cyt",
                elements=elements_all,
                stylesheet=stylesheet,
                style={"width": "100%", "height": "700px", "border": "1px solid #ccc"},
                # use preset layout so we use positions computed above (keeps width limited)
                layout={
                    "name": "preset",
                    "padding": 10
                },
                zoom=1,
                pan={"x": 0, "y": 0},
            ),
            # Floating "Show all layers" button (top-right)
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

            # Clear focus button (below Show all)
            html.Button(
                "Clear focus",
                id="clear-focus-btn",
                n_clicks=0,
                title="Clear node focus and show all (or previous) nodes",
                style={
                    "position": "absolute",
                    "top": "54px",
                    "right": "12px",
                    "zIndex": 9999,
                    "padding": "6px 10px",
                    "backgroundColor": "#fff",
                    "border": "1px solid #444",
                    "borderRadius": "4px",
                    "cursor": "pointer"
                }
            ),

            # Manual zoom controls (right-center)
            html.Div([
                html.Button("+", id="zoom-in-btn", n_clicks=0, title="Increase visible layers",
                            style={"width": "40px", "height": "40px", "fontSize": "18px", "marginBottom": "6px"}),
                html.Button("-", id="zoom-out-btn", n_clicks=0, title="Decrease visible layers",
                            style={"width": "40px", "height": "40px", "fontSize": "18px", "marginBottom": "6px"}),
                html.Button("Auto", id="auto-zoom-btn", n_clicks=0, title="Return to auto zoom-based layers",
                            style={"width": "60px", "height": "30px", "fontSize": "12px"}),
                html.Div(id="manual-layer-display", style={"marginTop": "8px", "textAlign": "center", "fontSize": "12px"})
            ], style={
                "position": "absolute",
                "right": "12px",
                "top": "50%",
                "transform": "translateY(-50%)",
                "zIndex": 9998,
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "background": "rgba(255,255,255,0.9)",
                "padding": "6px",
                "borderRadius": "6px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.15)"
            }),
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
    ], style={"display": "flex", "alignItems": "flex-start"}),
    # store full elements so we can filter without losing metadata
    dcc.Store(id="all-elements-store", data=elements_all),
    dcc.Store(id="max-layer-store", data=max_layer),
    # store for 'show all' toggle
    dcc.Store(id="show-all-store", data=False),
    # store to hold manual layer override (None => auto / zoom-based)
    dcc.Store(id="manual-layer-store", data=None),
    # store to hold current node focus id (None => no focus)
    dcc.Store(id="focus-store", data=None),
])


def zoom_to_visible_layer(zoom, max_layer):
    if zoom is None:
        return 0
    z = float(zoom)
    min_z, max_z = 0.2, 3.0
    z = max(min_z, min(max_z, z))
    frac = (z - min_z) / (max_z - min_z)
    layers_shown = int(round(frac * (max_layer)))
    return layers_shown


def adjust_layer0_positions(nodes_list, all_elements, desired_total_width=1200, top_spacing_multiplier=1.8, min_spacing=60):
    """
    Position layer-0 nodes horizontally to align with the centroid of their visible descendants.
    Only descendants with layer > 0 are considered for centroid computation (ignore same-layer nodes).
    Fallback: when centroid can't be computed, distribute remaining top nodes proportionally
    according to descendant counts. Enforce min_spacing and center the entire row around x=0.
    Modifies nodes_list in-place (updates node["position"]).
    """
    # map node id -> node element for available nodes (only node elements, not edges)
    node_map = {n["data"]["id"]: n for n in nodes_list if "source" not in n.get("data", {})}

    # build children adjacency from full graph (all_elements)
    children = {}
    layer_map = {}
    for el in all_elements:
        d = el.get("data", {})
        if "source" in d:
            children.setdefault(d["source"], []).append(d["target"])
        else:
            # record layer for every node id
            layer_map[d["id"]] = d.get("layer", 0)

    # collect preset positions from all_elements (if present)
    pos_map = {}
    for el in all_elements:
        d = el.get("data", {})
        if "source" not in d:
            p = el.get("position")
            if isinstance(p, dict) and "x" in p and "y" in p:
                pos_map[d["id"]] = p

    # find top layer nodes that exist in the current filtered node set
    top_nodes = [nid for nid, n in node_map.items() if n["data"].get("layer") == 0]
    if not top_nodes:
        return

    included_ids = set(node_map.keys())

    # collect all reachable descendants (DFS)
    def descendants(root):
        res = set()
        stack = list(children.get(root, []))
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            res.add(cur)
            stack.extend(children.get(cur, []))
        return res

    # compute centroid x for each top node using only included descendants that have positions and are higher-layer (>0)
    centroids = []
    weights = []
    for tn in top_nodes:
        descs = descendants(tn) & included_ids
        # consider only descendants with layer > 0 for centroid and weight
        xs = [pos_map[d]["x"] for d in descs if d in pos_map and layer_map.get(d, 0) > 0]
        centroid = (sum(xs) / len(xs)) if xs else None
        centroids.append(centroid)
        # weight used for fallback distribution: count only higher-layer descendants
        higher_count = sum(1 for d in descs if layer_map.get(d, 0) > 0)
        weights.append(max(1, higher_count))

    # total width to allocate
    total_width = max(desired_total_width, min_spacing * len(top_nodes)) * top_spacing_multiplier

    # create initial x list: use centroid where available, mark None for others
    xs_initial = list(centroids)

    # For nodes without centroid, allocate positions using proportional weights across remaining span
    missing_idx = [i for i, x in enumerate(xs_initial) if x is None]
    if missing_idx:
        # compute remaining span allocation: we will distribute along [-W/2, W/2]
        start = -total_width / 2.0
        # compute proportional slots by weight among missing
        missing_weights = [weights[i] for i in missing_idx]
        total_missing_weight = sum(missing_weights) or 1
        cursor = start
        for i, idx in enumerate(missing_idx):
            w = missing_weights[i]
            slot = (w / total_missing_weight) * total_width
            # center of allocated slot
            x_center = cursor + slot / 2.0
            xs_initial[idx] = x_center
            cursor += slot

    # For any remaining nodes (shouldn't be), fallback to evenly spaced
    for i, x in enumerate(xs_initial):
        if x is None:
            xs_initial[i] = (i - (len(xs_initial) - 1) / 2.0) * max(min_spacing, total_width / max(1, len(xs_initial)))

    # Now enforce min_spacing between adjacent top nodes while preserving order
    pairs = sorted([(x, i) for i, x in enumerate(xs_initial)], key=lambda t: t[0])
    for k in range(1, len(pairs)):
        prev_x, prev_i = pairs[k - 1]
        cur_x, cur_i = pairs[k]
        if cur_x < prev_x + min_spacing:
            cur_x = prev_x + min_spacing
            pairs[k] = (cur_x, cur_i)
    final_xs = [0.0] * len(pairs)
    for x, i in pairs:
        final_xs[i] = x

    # center the whole row around x=0
    mean_x = sum(final_xs) / len(final_xs)
    final_xs = [x - mean_x for x in final_xs]

    # apply positions (preserve existing y if present)
    for tn, x in zip(top_nodes, final_xs):
        el = node_map.get(tn)
        if not el:
            continue
        y = el.get("position", {}).get("y") if el.get("position") else 0
        el["position"] = {"x": x, "y": y}


# new helper: lay out the currently visible focused nodes in a vertical tree (focus node at y=0)
def layout_focus_vertical(nodes_list, focus_id, spacing_y=140, min_spacing_x=80):
    """
    Update nodes_list (in-place) node["position"] for a vertical tree layout centered on focus_id.
    Parents (smaller layer numbers) are placed above, children below.
    """
    # map id -> node element (node-only elements)
    node_map = {n["data"]["id"]: n for n in nodes_list if "source" not in n.get("data", {})}

    if focus_id not in node_map:
        # nothing to layout if focus not visible in current filtered set
        return

    focus_layer = node_map[focus_id]["data"].get("layer", 0)

    # group nodes by relative layer (layer - focus_layer)
    groups = {}
    for nid, el in node_map.items():
        layer = el["data"].get("layer", 0)
        rel = layer - focus_layer
        groups.setdefault(rel, []).append(el)

    # sort relative layers so vertical order is deterministic (parents above -> negative rel, then focus rel 0, then children)
    rel_layers = sorted(groups.keys())

    # compute positions for each relative layer row
    for rel in rel_layers:
        row = groups[rel]
        count = len(row)
        # estimate label widths to avoid overlaps (approximate)
        labels = [n["data"].get("label", "") for n in row]
        char_w = 7  # approx pixels per character
        padding = 20
        widths = [max(40, len(lbl) * char_w + padding) for lbl in labels] if labels else [min_spacing_x]
        max_width = max(widths)

        # dynamic spacing: ensure spacing >= min_spacing_x and large enough to fit widest label
        spacing_x = max(min_spacing_x, max_width + 8, 120)

        # if there are many nodes, slightly increase spacing to reduce collisions
        if count > 8:
            spacing_x *= 1.0 + (count - 8) * 0.04

        width = (count - 1) * spacing_x
        x_start = -width / 2.0
        y_pos = rel * spacing_y
        # stabilize order by node id to avoid jumping
        row_sorted = sorted(row, key=lambda e: e["data"].get("id"))
        for i, node in enumerate(row_sorted):
            x = x_start + i * spacing_x
            node["position"] = {"x": x, "y": y_pos}


# replace the filter_elements callback with focus-aware filtering
@app.callback(
    Output("cyt", "elements"),
    Output("zoom-info", "children"),
    Input("cyt", "zoom"),
    Input("show-all-store", "data"),
    Input("manual-layer-store", "data"),
    Input("focus-store", "data"),
    State("all-elements-store", "data"),
    State("max-layer-store", "data"),
)
def filter_elements(zoom, show_all, manual_layer, focus_node_id, all_elements, max_layer_store):
    # If show-all is on, ignore zoom/manual/focus and show everything
    if show_all:
        node_ids = set()
        filtered_nodes = []
        filtered_edges = []
        for el in all_elements:
            if "source" in el.get("data", {}):
                filtered_edges.append(el)
            else:
                filtered_nodes.append(el)
                node_ids.add(el["data"]["id"])
        # adjust layer0 spacing based on visible descendants
        adjust_layer0_positions(filtered_nodes, all_elements)
        msg = f"Show-all ON — showing all layers (nodes: {len(filtered_nodes)}, edges: {len(filtered_edges)})"
        return filtered_nodes + filtered_edges, msg

    # If focus request present: show only grandparents, parents, self, children, grandchildren
    if focus_node_id:
        # build adjacency maps
        parents = {}
        children = {}
        for el in all_elements:
            data = el.get("data", {})
            if "source" in data:
                s = data["source"]; t = data["target"]
                children.setdefault(s, set()).add(t)
                parents.setdefault(t, set()).add(s)
            else:
                if data["id"] == focus_node_id:
                    # for the focus node, ensure it has a defined position (in case it's a new addition)
                    children.setdefault(None, set()).add(data["id"])

        included = set()
        included.add(focus_node_id)
        # parents and children
        p1 = parents.get(focus_node_id, set())
        c1 = children.get(focus_node_id, set())
        included.update(p1)
        included.update(c1)
        # grandparents (parents of parents)
        for p in list(p1):
            included.update(parents.get(p, set()))
        # grandchildren (children of children)
        for c in list(c1):
            included.update(children.get(c, set()))

        # collect node and edge elements for included set
        filtered_nodes = []
        filtered_edges = []
        for el in all_elements:
            data = el.get("data", {})
            if "source" in data:
                if data["source"] in included and data["target"] in included:
                    filtered_edges.append(el)
            else:
                if data["id"] in included:
                    filtered_nodes.append(el)

        # layout focused subset vertically (focus node centered at y=0)
        layout_focus_vertical(filtered_nodes, focus_node_id, spacing_y=120, min_spacing_x=100)

        msg = f"Focus: {focus_node_id} — showing related nodes (total nodes: {len(filtered_nodes)})"
        return filtered_nodes + filtered_edges, msg

    # Otherwise: normal layer-based filtering (manual or auto)
    if manual_layer is not None:
        visible_layer = int(max(0, min(max_layer_store, manual_layer)))
    else:
        visible_layer = zoom_to_visible_layer(zoom, max_layer_store)

    node_ids = set()
    filtered_nodes = []
    for el in all_elements:
        if "source" in el.get("data", {}):
            continue
        layer = el["data"].get("layer", 0)
        if layer <= visible_layer:
            filtered_nodes.append(el)
            node_ids.add(el["data"]["id"])
    filtered_edges = []
    for el in all_elements:
        if "source" in el.get("data", {}):
            s = el["data"]["source"]
            t = el["data"]["target"]
            if s in node_ids and t in node_ids:
                filtered_edges.append(el)

    # adjust layer0 spacing based on currently visible downstream nodes
    adjust_layer0_positions(filtered_nodes, all_elements)

    zoom_display = f"{float(zoom):.2f}" if zoom is not None else "N/A"
    mode = f"manual(max layer={manual_layer})" if manual_layer is not None else "auto"
    msg = f"Mode: {mode} — Zoom: {zoom_display} — showing layers 0..{visible_layer} (nodes: {len(filtered_nodes)}, edges: {len(filtered_edges)})"
    return filtered_nodes + filtered_edges, msg


# new callback: adjust viewport (zoom & pan) when focus changes so focused nodes are centered & scaled
@app.callback(
    Output("cyt", "zoom"),
    Output("cyt", "pan"),
    Input("focus-store", "data"),
    State("all-elements-store", "data"),
)
def adjust_viewport_for_focus(focus_node_id, all_elements):
    if not focus_node_id:
        # don't change viewport when focus cleared
        return dash.no_update, dash.no_update

    # build adjacency and preset positions map
    parents = {}
    children = {}
    pos = {}
    node_elements = {}
    for el in all_elements:
        data = el.get("data", {})
        if "source" in data:
            s = data["source"]; t = data["target"]
            children.setdefault(s, set()).add(t)
            parents.setdefault(t, set()).add(s)
        else:
            node_elements[data.get("id")] = el
            p = el.get("position")
            if isinstance(p, dict) and "x" in p and "y" in p:
                pos[data["id"]] = p

    included = set()
    included.add(focus_node_id)
    p1 = parents.get(focus_node_id, set()); c1 = children.get(focus_node_id, set())
    included.update(p1); included.update(c1)
    for p in list(p1):
        included.update(parents.get(p, set()))
    for c in list(c1):
        included.update(children.get(c, set()))

    xs = []; ys = []
    missing = []
    for node_id in included:
        if node_id in pos:
            xs.append(pos[node_id]["x"]); ys.append(pos[node_id]["y"])
        else:
            missing.append(node_id)

    if missing:
        # compute temporary vertical layout for included nodes so viewport can center/zoom correctly
        temp_nodes = []
        for nid in included:
            el = node_elements.get(nid)
            layer = el.get("data", {}).get("layer", 0) if el else 0
            temp_nodes.append({"data": {"id": nid, "layer": layer}})
        layout_focus_vertical(temp_nodes, focus_node_id, spacing_y=120, min_spacing_x=100)
        for el in temp_nodes:
            p = el.get("position")
            if p:
                xs.append(p["x"]); ys.append(p["y"])

    if not xs or not ys:
        return dash.no_update, dash.no_update

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bbox_w = max(10, maxx - minx)
    bbox_h = max(10, maxy - miny)

    view_w = 900.0
    view_h = 600.0
    zoom_x = view_w / (bbox_w + 200)
    zoom_y = view_h / (bbox_h + 200)
    new_zoom = float(max(0.2, min(3.0, min(zoom_x, zoom_y))))

    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    pan = {"x": -center_x * new_zoom + view_w / 2.0, "y": -center_y * new_zoom + view_h / 2.0}

    return new_zoom, pan


# replace node details callback to include full lineage (all ancestor paths up to top layer)
@app.callback(
    Output("node-details", "children"),
    Input("focus-store", "data"),
    State("all-elements-store", "data"),
)
def show_node_details(focus_node_id, all_elements):
    if not focus_node_id:
        return "Click a node to see full details."

    # find the full node element in the stored full-elements list
    node_data = None
    for el in all_elements:
        d = el.get("data", {})
        if d.get("id") == focus_node_id:
            node_data = d
            break

    if not node_data:
        return f"Node {focus_node_id} not found."

    node_id = node_data.get("id")
    title = node_data.get("label", "")
    full = node_data.get("full_text", "")
    layer = node_data.get("layer", "")

    # build parents map from the full element list
    parents = {}
    for el in all_elements:
        d = el.get("data", {})
        if "source" in d:
            s = d["source"]; t = d["target"]
            parents.setdefault(t, []).append(s)

    # recursive function to build ancestor paths (from root -> ... -> node)
    def ancestor_paths(nid):
        ps = parents.get(nid, [])
        if not ps:
            return [[nid]]
        paths = []
        for p in ps:
            for path in ancestor_paths(p):
                paths.append(path + [nid])
        return paths

    paths = ancestor_paths(node_id)

    # build a map id -> label for nodes for faster lookup
    id_to_label = {}
    for el in all_elements:
        d = el.get("data", {})
        if "source" not in d:
            id_to_label[d.get("id")] = d.get("label", d.get("id"))

    formatted = []
    for p in paths:
        labels = [id_to_label.get(nid, nid) for nid in p]
        formatted.append(" -> ".join(labels))

    lineage_text = "\n".join(formatted) if formatted else node_id

    details = f"ID: {node_id}\nLayer: {layer}\nTitle: {title}\n\nLineage:\n{lineage_text}\n\nFull text:\n{full}"
    return details


@app.callback(
    Output("show-all-store", "data"),
    Output("show-all-btn", "children"),
    Input("show-all-btn", "n_clicks"),
    State("show-all-store", "data"),
)
def toggle_show_all(n_clicks, current):
    if not n_clicks:
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
    elements, max_layer = generate_test_graph(
        layers=7, top_nodes=10, min_branch=3, max_branch=5, max_nodes=int(max_nodes or 1000), seed=random.randint(0,9999)
    )
    return elements, max_layer


@app.callback(
    Output("manual-layer-store", "data"),
    Output("manual-layer-display", "children"),
    Input("zoom-in-btn", "n_clicks"),
    Input("zoom-out-btn", "n_clicks"),
    Input("auto-zoom-btn", "n_clicks"),
    State("manual-layer-store", "data"),
    State("max-layer-store", "data"),
)
def adjust_manual_layer(inc_clicks, dec_clicks, auto_clicks, manual_layer, max_layer_store):
    ctx = callback_context
    if not ctx.triggered:
        # no change
        display = f"Auto (zoom)"
        if manual_layer is not None:
            display = f"Manual: max layer = {manual_layer}"
        return manual_layer, display

    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    cur = manual_layer if manual_layer is not None else 0

    if trig == "zoom-in-btn":
        new = min(max_layer_store, cur + 1)
    elif trig == "zoom-out-btn":
        new = max(0, cur - 1)
    elif trig == "auto-zoom-btn":
        new = None
    else:
        new = manual_layer

    display = f"Manual: max layer = {new}" if new is not None else "Auto (zoom)"
    return new, display


# Replace the two separate focus callbacks with one combined callback to avoid duplicate outputs.
@app.callback(
    Output("focus-store", "data"),
    Input("cyt", "tapNodeData"),
    Input("clear-focus-btn", "n_clicks"),
)
def handle_focus(tap_node_data, clear_n):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    if trig == "clear-focus-btn":
        if not clear_n:
            return dash.no_update
        return None

    if trig == "cyt":
        if not tap_node_data:
            return dash.no_update
        return tap_node_data.get("id")

    return dash.no_update


# Example input structure (nodes + edges). Format this way if you want to feed/serialize the network.
EXAMPLE_INPUT = {
    "nodes": [
        {"id": "L0-0", "label": "TopNode 0", "layer": 0, "full_text": "Detailed text..."},
        {"id": "L1-0-1", "label": "Child A", "layer": 1, "full_text": "Child detailed..."},
        {"id": "L1-0-2", "label": "Child B", "layer": 1, "full_text": "Child detailed..."}
    ],
    "edges": [
        {"source": "L0-0", "target": "L1-0-1"},
        {"source": "L0-0", "target": "L1-0-2"}
    ]
}


if __name__ == "__main__":
    print("Starting Dash app on http://127.0.0.1:8050")
    app.run(debug=True, port=8050, host="0.0.0.0")