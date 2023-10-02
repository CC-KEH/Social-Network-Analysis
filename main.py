import networkx as nx
import matplotlib.pyplot as plt

def txt_to_graph(file_path,G):   
    # Open the file and read each line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into tokens (assuming space-separated values)
            tokens = line.strip().split()

            # Extract the nodes and weight (if available)
            if len(tokens) >= 2:
                node1 = tokens[0]
                node2 = tokens[1]

                # Check if a weight is provided
                if len(tokens) == 3:
                    weight = float(tokens[2])
                else:
                    weight = 1.0  # Default weight is 1.0

                # Add the edge with the specified weight
                G.add_edge(node1, node2, weight=weight)

    # You can now perform operations on the graph G
    for u, v, data in G.edges(data=True):
        print(f"Edge ({u}, {v}) has weight {data['weight']}")

def visualize(G):
    # Create a layout for the graph
    pos = nx.spring_layout(G)

    # Extract edge weights to be used as labels
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_weight='bold')
    
    # Show the graph
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Create an empty weighted graph
    G = nx.Graph()

    # Define the path to your dataset .txt file
    file_path = 'dataset.txt'
    txt_to_graph(file_path,G)
    visualize(G)

