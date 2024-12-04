import networkx as nx
import spacy
from community import community_louvain
import matplotlib.pyplot as plt
import streamlit as st
import pdfplumber  # For PDF text extraction


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file."""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def process_document(self, text):
        """Processes the document to extract entities and relationships."""
        doc = self.nlp(text)
        for sent in doc.sents:
            entities = [ent for ent in sent.ents]
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i + 1:]:
                    self.graph.add_node(ent1.text, label=ent1.label_)
                    self.graph.add_node(ent2.text, label=ent2.label_)
                    # Add or update edge weights
                    if self.graph.has_edge(ent1.text, ent2.text):
                        self.graph[ent1.text][ent2.text]['weight'] += 1
                    else:
                        self.graph.add_edge(ent1.text, ent2.text, weight=1)

    def filter_graph_for_all_links(self, target_entity):
        """Filters the graph to include all edges connected to the target entity."""
        if target_entity not in self.graph:
            st.warning(f"The target entity '{target_entity}' is not found in the graph.")
            return nx.Graph()  # Return an empty graph

        # Create a subgraph with all nodes and edges connected to the target entity
        all_links_graph = nx.Graph()
        for node1, node2, data in self.graph.edges(data=True):
            if node1 == target_entity or node2 == target_entity:
                all_links_graph.add_node(node1, label=self.graph.nodes[node1].get("label", ""))
                all_links_graph.add_node(node2, label=self.graph.nodes[node2].get("label", ""))
                all_links_graph.add_edge(node1, node2, weight=data["weight"])

        return all_links_graph

    def visualize_graph(self, filtered_graph, target_entity):
        """Visualizes the filtered knowledge graph."""
        if not filtered_graph.nodes:
            st.warning("No connections found for the specified target entity.")
            return

        # Assign cluster-based colors
        partition = community_louvain.best_partition(filtered_graph)
        cluster_colors = plt.cm.tab20.colors
        node_colors = [
            cluster_colors[partition[node] % len(cluster_colors)] for node in filtered_graph
        ]

        # Layout and visualization
        pos = nx.spring_layout(filtered_graph, seed=42)
        plt.figure(figsize=(14, 8))
        nx.draw_networkx_nodes(
            filtered_graph, pos, node_size=800, node_color=node_colors, edgecolors="black"
        )
        nx.draw_networkx_edges(filtered_graph, pos, alpha=0.5, edge_color="gray", width=1.5)
        nx.draw_networkx_labels(filtered_graph, pos, font_size=10, font_color="black")

        plt.title(f"Knowledge Graph for '{target_entity}' and Connected Entities", fontsize=18)
        plt.axis("off")
        st.pyplot(plt)
        plt.close()


