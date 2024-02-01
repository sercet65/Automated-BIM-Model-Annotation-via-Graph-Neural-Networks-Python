# Automated-BIM-Model-Annotation-via-Graph-Neural-Networks-Python

# Graph Generation and Neo4j Integration

This project includes a set of Python scripts for generating graph data from a text file, saving it in various formats (CSV, GraphML), and writing the data to a Neo4j graph database.

## Description

The `generate_graph_from_text_file` function processes a text file to create node data, while `generate_edges_data` generates edge data based on spatial adjacencies. The data can be saved as CSV or GraphML using `save_graph_to_graphml`, and can be written to a Neo4j database using `write_to_neo4j`.

## Installation

Before running the scripts, ensure you have the following dependencies installed:

- Neo4j
- Python 3
- NetworkX
- Pandas
- Numpy

You can install the necessary Python packages using pip:

```bash
pip install neo4j pandas numpy networkx

## Configuration
Before running the script, make sure to configure the following:

Neo4j URI, user, and password in the write_to_neo4j function.
The path to your text file in the generate_graph_from_text_file function call.
The CSV file paths in the main section of the script.
