import streamlit as st
from knowledge_graph import KnowledgeGraphBuilder

# Streamlit Interface
def main():
    st.title("Knowledge Graph Builder")

    uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
    target_entity = st.text_input("Enter the target entity:")

    if uploaded_file and st.button("Build Knowledge Graph"):
        kg_builder = KnowledgeGraphBuilder()
        
        # Extract text from the uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            text = kg_builder.extract_text_from_pdf(uploaded_file)
        
        # Process the document
        with st.spinner("Processing document..."):
            kg_builder.process_document(text)

        # Filter graph for all links
        with st.spinner("Filtering graph for target entity..."):
            all_links_graph = kg_builder.filter_graph_for_all_links(target_entity)

        # Visualize the graph
        kg_builder.visualize_graph(all_links_graph, target_entity)


if __name__ == "__main__":
    main()
