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

def extract_position(pos_str):
    # This pattern matches coordinates in the format (x, y)
    pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    matches = re.search(pattern, pos_str)
    if matches:
        x, y = matches.groups()
        return float(x), float(y)
    else:
        # If the pattern is not found, return None values
        return None, None
    
def generate_graph_from_text_file(file_path):
    nodes_data = []
    node_id = 0  # Initialize a counter for node IDs

    with open(file_path, 'r') as file:
        for line in file:
            elements = re.split(r', (?=[A-Z])', line.strip())
            node_data = {"node_id": node_id}  # Add node ID as the first element in node_data

            # Initialize consolidated bounding box keys with default values
            consolidated_bbox = {'bb_xmin': 0.0, 'bb_ymin': 0.0, 'bb_zmin': 0.0, 
                                 'bb_xmax': 0.0, 'bb_ymax': 0.0, 'bb_zmax': 0.0}

            for element in elements:
                key_value = element.split(': ', 1)
                
                if len(key_value) == 2:
                    key, value = key_value
                    key_formatted = key.lower().replace(' ', '_')

                    # Check if the key is for a bounding box of interest
                    if 'bounding_box' in key_formatted:
                        coords = extract_coordinates(value)
                        # Update the consolidated bounding box keys
                        consolidated_bbox.update({'bb_xmin': coords[0], 'bb_ymin': coords[1], 'bb_zmin': coords[2],
                                                  'bb_xmax': coords[3], 'bb_ymax': coords[4], 'bb_zmax': coords[5]})
                    elif key == "Label Type":
                        node_data['label_type'] = int(value)
                    elif key == "Embedded Doors":
                        node_data['embedded_doors'] = value.strip('[]').split(', ') if value.startswith('[') else []

                    elif 'position' in key_formatted:
                         pos_x, pos_y = extract_position(value)
                         if pos_x is not None and pos_y is not None:
                            node_data['pos_x'] = pos_x
                            node_data['pos_y'] = pos_y
    
                    else:
                        node_data[key_formatted] = value

            # Merge the consolidated bounding box information with the rest of the node data
            node_data.update(consolidated_bbox)

            if node_data:  # Check if any data was extracted before adding it to the list
                nodes_data.append(node_data)
                node_id += 1  # Increment node ID for the next node

    return nodes_data

import numpy as np


def generate_edges_data(nodes_data, distance_tolerance=0.1):
    # Prepare formatted_shapes for spatial adjacency calculation, ensuring correct bounding box handling.
    formatted_shapes = {}
    for node in nodes_data:
        bbox = np.array([
            node.get('bb_xmin', 0),
            node.get('bb_ymin', 0),
            node.get('bb_zmin', 0),
            node.get('bb_xmax', 0),
            node.get('bb_ymax', 0),
            node.get('bb_zmax', 0)
        ])
        # Ensure the bounding box is correctly formatted and represented.
        formatted_shapes[node['node_id']] = bbox

    # Determine adjacency between nodes using the specified distance tolerance.
    adjacencies, distances, _ = get_adjacencies(formatted_shapes, distance_tolerance)

    # Map node IDs to bounding box coordinates for easy access
    node_id_to_bbox = {node['node_id']: formatted_shapes[node['node_id']] for node in nodes_data}

    # Initialize the edges data dictionary
    edges_data = {
        "edge_id": [],
        "node1_id": [],
        "node2_id": [],
        "distance": []
    }

    # Generate edges from adjacency information
    for (node_id1, node_id2), distance in zip(adjacencies, distances):

          # Check if either of the nodes is of type "Dim Text" and if the text field has a value of 0
        is_dim_text_0 = (
            (nodes_data[node_id1]['element_type'] == 'DimText' and 
             str(nodes_data[node_id1].get('text', '')) == '0') or
            (nodes_data[node_id2]['element_type'] == 'DimText' and 
             str(nodes_data[node_id2].get('text', '')) == '0')
        )
        
        if is_dim_text_0:
            continue  # Skip creating edge for this pair of nodes


        # Add an edge between node1 and node2 if their distance is within tolerance
        edges_data["edge_id"].append(len(edges_data["edge_id"]) + 1)  # Unique edge ID
        edges_data["node1_id"].append(node_id1)
        edges_data["node2_id"].append(node_id2)
        edges_data["distance"].append(distance)

    return edges_data

 
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

def clean_database(driver, database="neo4j4"):
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")

def write_to_neo4j(node_csv="nodes_data.csv", edge_csv="edges_data.csv", database="neo4j4"):
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    if os.path.exists(node_csv) and os.path.exists(edge_csv):
        elements_df = pd.read_csv(node_csv)
        edges_df = pd.read_csv(edge_csv)

        # Convert dataframes to list of dictionaries
        nodes_dict_list = elements_df.to_dict('records')
        edges_dict_list = edges_df.to_dict('records')

        def add_data_to_neo4j(tx, nodes, edges):
            # Add nodes to Neo4j
            for node in nodes:
                tx.run("""
                MERGE (n:Node {node_id: $node_id})
                SET n.element_type = $element_type,
                    n.guid = $guid,
                    n.info_string = $info_string,
                    n.bb_xmin = $bb_xmin,
                    n.bb_ymin = $bb_ymin,
                    n.bb_zmin = $bb_zmin,
                    n.bb_xmax = $bb_xmax,
                    n.bb_ymax = $bb_ymax,
                    n.bb_zmax = $bb_zmax,
                    n.pos_x = $pos_x,
                    n.pos_y = $pos_y,
                    n.label_type = $label_type
                """, parameters=node)
            
            # Add edges to Neo4j
            tx.run("""
                UNWIND $edges AS edge
                MATCH (a:Node {node_id: edge.node1_id}), (b:Node {node_id: edge.node2_id})
                MERGE (a)-[r:CONNECTS_TO]->(b)
                ON CREATE SET r.edge_id = edge.edge_id, r.distance = edge.distance
                ON MATCH SET r.distance = edge.distance
            """, parameters={'edges': edges_dict_list})

        # Execute the function to add data to Neo4j
        with driver.session(database="neo4j4") as session:
            session.write_transaction(add_data_to_neo4j, nodes_dict_list, edges_dict_list)

        # Close the driver connection
        driver.close()

def main():

    uri = "neo4j://localhost:7687"
    auth = ("neo4j", "password")  # Replace with your actual credentials
    database_name = "neo4j4"  # Specify your target database name

    driver = GraphDatabase.driver(uri, auth=auth)

    # Confirmation prompt to prevent accidental data loss
    confirm = input("This will delete all data in the database '{}'. Are you sure? (yes/no): ".format(database_name))
    if confirm.lower() == 'yes':
        clean_database(driver, database=database_name)
        print("Database has been cleaned.")
    else:
        print("Operation cancelled.")
        driver.close()
        return

    # Paths to your files
    text_file_path = r'C:\Users\serve\OneDrive\Desktop\Phython\Elementinfo.txt'
    elements_csv_file_path = r'C:\Users\serve\OneDrive\Desktop\Phython\elements_data.csv'
    annotations_csv_file_path = r'C:\Users\serve\OneDrive\Desktop\Phython\annotations_data.csv'
    nodes_csv_file_path = r'C:\Users\serve\OneDrive\Desktop\Phython\nodes_data.csv'
    edges_csv_file_path = r'C:\Users\serve\OneDrive\Desktop\Phython\edges_data.csv'
    output_graphml_path = r'C:\Users\serve\OneDrive\Desktop\Phython\output_graph.graphml'

    # Generate nodes and edges data
    nodes_data = generate_graph_from_text_file(text_file_path)
    edges_data = generate_edges_data(nodes_data)

    # Filter nodes_data into two separate lists: one for elements and one for annotations
    elements_data = [node for node in nodes_data if node['element_type'] in ['Wall', 'Door', 'Zone', 'Slab']]
    annotations_data = [node for node in nodes_data if node['element_type'] not in ['Wall', 'Door', 'Zone', 'Slab']]

    # Convert to data frames
    elements_df = pd.DataFrame(elements_data)
    annotations_df = pd.DataFrame(annotations_data)
    edges_df = pd.DataFrame(edges_data)
    nodes_df = pd.DataFrame(nodes_data)

    # Save to CSV files
    elements_df.to_csv(elements_csv_file_path, index=False ) # add this if you see merged items in csv file sep=';'
    annotations_df.to_csv(annotations_csv_file_path, index=False) # add this if you see merged items in csv file sep=';'
    edges_df.to_csv(edges_csv_file_path, index=False) # add this if you see merged items in csv file sep=';'
    nodes_df.to_csv(nodes_csv_file_path, index=False) # add this if you see merged items in csv file sep=';'


    # Save to GraphML
    save_graph_to_graphml(pd.concat([elements_df, annotations_df]), edges_df, output_graphml_path)

    # Optional: Print data for verification
    #print("Elements:")
    #print(elements_df.head())
    #print("\nAnnotations:")
    #print(annotations_df.head())

    write_to_neo4j(node_csv="nodes_data.csv", edge_csv="edges_data.csv", database="neo4j4")

if __name__ == "__main__":
    main()
