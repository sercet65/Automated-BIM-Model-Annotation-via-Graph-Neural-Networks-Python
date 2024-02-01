import os
from neo4j import GraphDatabase
import numpy as np
from spatial_queries import get_adjacencies
import networkx as nx
import pandas as pd
import re

def extract_coordinates(coord_str):
    # This pattern matches the coordinates in the format [(x1, y1, z1), (x2, y2, z2)]
    pattern = r'\[\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\),\s*\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)\]'
    matches = re.search(pattern, coord_str)
    if matches:
        coords = [float(num) for num in matches.groups()]
    else:
        # If the pattern is not found, return an empty list
        coords = []
    return coords

def generate_graph_from_text_file(file_path):
    nodes_data = []
    node_id = 0  # Initialize a counter for node IDs

    with open(file_path, 'r') as file:
        for line in file:
            elements = re.split(r', (?=[A-Z])', line.strip())
            node_data = {"node_id": node_id}  # Add node ID as the first element in node_data

            for element in elements:
                key_value = element.split(': ', 1)

                if len(key_value) == 2:
                    key, value = key_value
                    key_formatted = key.lower().replace(' ', '_')

                    if 'bounding_box' in key_formatted:
                        coords = extract_coordinates(value)
                        if coords:
                            bbox_keys = ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']
                            node_data.update({f'{key_formatted}_{k}': v for k, v in zip(bbox_keys, coords)})
                    else:
                        if key == "Label Type" or key == "Embedded Doors":
                            if value.startswith('['):  # It's a list
                                value = value.strip('[]').split(', ')
                            else:
                                value = int(value)
                        node_data[key_formatted] = value

            if node_data:
                nodes_data.append(node_data)
                node_id += 1  # Increment node ID for the next node



    return nodes_data

def generate_edges_data(nodes_data):
    # Prepare formatted_shapes for get_adjacencies
    formatted_shapes = {}
    for node in nodes_data:
        base_key = f"{node['element_type'].lower()}_bounding_box_"
        formatted_shapes[node['guid']] = np.array([
            node.get(base_key + 'xmin', 0),
            node.get(base_key + 'ymin', 0),
            node.get(base_key + 'zmin', 0),
            node.get(base_key + 'xmax', 0),
            node.get(base_key + 'ymax', 0),
            node.get(base_key + 'zmax', 0)])
    
    # Get adjacencies with a specified distance tolerance
    distance_tolerance = 0.1
    adjacencies, distances, _ = get_adjacencies(formatted_shapes, distance_tolerance)

    # Map node names to node IDs
    node_name_to_id = {node['guid']: node['node_id'] for node in nodes_data}

    # Create edges_data based on adjacencies
    edges_data = {
        "edge_id": [i for i in range(len(adjacencies))],
        "node1_id": [node_name_to_id[node_pair[0]] for node_pair in adjacencies],
        "node2_id": [node_name_to_id[node_pair[1]] for node_pair in adjacencies],
        "distance": distances
    }
    return edges_data

# Call the modified function with the path to your text file
nodes_data = generate_graph_from_text_file(r'C:\Users\serve\OneDrive\Desktop\Phythin\Elementinfo.txt')
edges_data = generate_edges_data(nodes_data)

# Convert nodes_data list to a DataFrame
nodes_df = pd.DataFrame(nodes_data)
edges_df = pd.DataFrame(edges_data)

# Specify the CSV file path where you want to save the data
nodes_csv_file_path = r'C:\Users\serve\OneDrive\Desktop\Phythin\nodes_data.csv'
edges_csv_file_path = r'C:\Users\serve\OneDrive\Desktop\Phythin\edges_data.csv'

# Save the DataFrame to a CSV file
nodes_df.to_csv(nodes_csv_file_path, index=False)
edges_df.to_csv(edges_csv_file_path, index=False)

# Print the list of dictionaries (each dictionary represents a node)
for node in nodes_data:
    print(node)

def save_graph_to_graphml(nodes_df, edges_df, output_graphml):
    G = nx.Graph()

 # Convert complex or unsupported data types to strings
    for _, row in nodes_df.iterrows():
        node_attributes = {k: str(v) if isinstance(v, (list, dict, type)) else v 
                           for k, v in row.items() if not pd.isna(v)}
        G.add_node(row['node_id'], **node_attributes)

    # Add edges to the graph
    if edges_df is not None:
        for _, row in edges_df.iterrows():
            G.add_edge(row['node1_id'], row['node2_id'], distance=row.get('distance', 0.0))

    # Save the graph to a GraphML file
    nx.write_graphml(G, output_graphml)
    print(f"Graph saved to {output_graphml}")

# Specify the path for the output GraphML file
output_graphml_path = r'C:\Users\serve\OneDrive\Desktop\Phythin\output_graph.graphml'

# Call the function to save the graph
save_graph_to_graphml(nodes_df, edges_df, output_graphml_path)  # Ensure this line is executed

def write_to_neo4j(node_csv="nodes_data.csv", edge_csv="edges_data.csv"):
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    if os.path.exists(node_csv) and os.path.exists(edge_csv):
        nodes_df = pd.read_csv(node_csv)
        edges_df = pd.read_csv(edge_csv)

        # Convert dataframes to list of dictionaries
        nodes_dict_list = nodes_df.to_dict('records')
        edges_dict_list = edges_df.to_dict('records')

        def add_data_to_neo4j(tx):
            # Add nodes to Neo4j
            tx.run("""
            MERGE (n:Node {node_id: $node_id})
            SET n.element_type = $element_type,
                n.guid = $guid,
                n.wall_bounding_box_xmin = $wall_bounding_box_xmin,
                n.wall_bounding_box_ymin = $wall_bounding_box_ymin,
                n.wall_bounding_box_zmin = $wall_bounding_box_zmin,
                n.wall_bounding_box_xmax = $wall_bounding_box_xmax,
                n.wall_bounding_box_ymax = $wall_bounding_box_ymax,
                n.wall_bounding_box_zmax = $wall_bounding_box_zmax,
                n.door_bounding_box_xmin = $door_bounding_box_xmin,
                n.door_bounding_box_ymin = $door_bounding_box_ymin,
                n.door_bounding_box_zmin = $door_bounding_box_zmin,
                n.door_bounding_box_xmax = $door_bounding_box_xmax,
                n.door_bounding_box_ymax = $door_bounding_box_ymax,
                n.door_bounding_box_zmax = $door_bounding_box_zmax,
                n.zone_bounding_box_xmin = $zone_bounding_box_xmin,
                n.zone_bounding_box_ymin = $zone_bounding_box_ymin,
                n.zone_bounding_box_zmin = $zone_bounding_box_zmin,
                n.zone_bounding_box_xmax = $zone_bounding_box_xmax,
                n.zone_bounding_box_ymax = $zone_bounding_box_ymax,
                n.zone_bounding_box_zmax = $zone_bounding_box_zmax
                -- Include other attributes as needed
        """, node)
            
            # Add edges to Neo4j
            tx.run("""
                UNWIND $edges as edge
                MATCH (a:Node {node_id: edge.node1_id}), (b:Node {node_id: edge.node2_id})
                MERGE (a)-[r:CONNECTS_TO {edge_id: edge.edge_id, distance: edge.distance}]->(b)
            """, edges=edges_dict_list)

        # Execute the function to add data to Neo4j
        with driver.session() as session:
            session.write_transaction(add_data_to_neo4j)

        # Close the driver connection
        driver.close()

if __name__ == "__main__":
	write_to_neo4j(node_csv="nodes_data.csv", edge_csv="edges_data.csv")
