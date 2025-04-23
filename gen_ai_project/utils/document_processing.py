# utils/document_processing.py

import os
from typing import List

# --- LangChain Components ---
# Use UnstructuredFileLoader for broad file type support
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Rename to avoid potential future naming conflicts within this file
from langchain.schema import Document as LangchainDocument

def load_and_split_documents(
    source_path: str,
    workspace_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[LangchainDocument]:
    """
    Loads documents from a file or directory and splits them into chunks.
    Supports various file types via UnstructuredFileLoader.

    Args:
        source_path (str): The path to the document file (e.g., 'docs/policy.pdf')
                           or directory (e.g., 'knowledge_files/') relative to the workspace,
                           or an absolute path.
        workspace_dir (str): The absolute path to the agent's workspace directory.
                             Used to resolve relative source_paths.
        chunk_size (int): The target size for each document chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        List[LangchainDocument]: A list of split document chunks, or an empty list if loading fails.
    """
    full_path = source_path
    # Resolve relative paths against the workspace directory
    if not os.path.isabs(source_path):
        full_path = os.path.join(workspace_dir, source_path)

    if not os.path.exists(full_path):
        print(f"Error [Document Processing]: Source path not found: {full_path}")
        return []

    documents = []
    print(f"Info [Document Processing]: Attempting to load documents from: {full_path}")
    try:
        if os.path.isdir(full_path):
            # Load all supported files from the directory using UnstructuredFileLoader
            # show_progress=True can be helpful for large directories
            loader = DirectoryLoader(
                full_path,
                glob="**/*.*", # Load all files recursively
                loader_cls=UnstructuredFileLoader,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True # Prevent loader from crashing on a single bad file
            )
            documents = loader.load()
            if not documents:
                 print(f"Warning [Document Processing]: Directory loaded, but no documents found/extracted in {full_path}")

        elif os.path.isfile(full_path):
            # Use UnstructuredFileLoader for broad single-file support
            # Handle potential errors during loading of a single file
            try:
                loader = UnstructuredFileLoader(full_path)
                documents = loader.load()
            except Exception as e:
                 print(f"Error [Document Processing]: Failed to load single file {full_path} with UnstructuredFileLoader: {e}")
                 return [] # Return empty list if single file load fails
        else:
            # This case should ideally not be reached due to the initial os.path.exists check
            print(f"Error [Document Processing]: Path is neither a file nor a directory: {full_path}")
            return []

    except Exception as e:
        # Catch potential errors during DirectoryLoader initialization or loading
        print(f"Error [Document Processing]: Failed during document loading process for {full_path}: {e}")
        return []


    if not documents:
        print(f"Info [Document Processing]: No documents were successfully loaded from {full_path}.")
        return []

    print(f"Info [Document Processing]: Successfully loaded {len(documents)} document(s) from source.")

    # Split documents into manageable chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len, # Use standard length function
            add_start_index=True, # Helps potentially with locating context later
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Info [Document Processing]: Split {len(documents)} document(s) into {len(split_docs)} chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")
        return split_docs
    except Exception as e:
         print(f"Error [Document Processing]: Failed during text splitting: {e}")
         # Decide whether to return original documents or empty list on splitting error
         # Returning empty might be safer for RAG consistency
         return []

# Example of how this might be tested (optional, usually done in separate test files)
# if __name__ == "__main__":
#     print("Testing document processing...")
#     # Create dummy workspace and file for testing
#     test_workspace = "./temp_test_workspace"
#     os.makedirs(test_workspace, exist_ok=True)
#     test_file_path = os.path.join(test_workspace, "test_doc.txt")
#     with open(test_file_path, "w") as f:
#         f.write("This is the first sentence.\n")
#         f.write("This is the second sentence, which is a bit longer.\n")
#         f.write("Finally, the third sentence concludes this short document.")
#
#     print(f"Created test file: {test_file_path}")
#
#     # Test loading the single file
#     split_documents = load_and_split_documents(
#         source_path="test_doc.txt", # Relative path
#         workspace_dir=os.path.abspath(test_workspace),
#         chunk_size=50,
#         chunk_overlap=10
#     )
#
#     if split_documents:
#         print(f"\nSuccessfully loaded and split into {len(split_documents)} chunks:")
#         for i, doc in enumerate(split_documents):
#             print(f"--- Chunk {i+1} ---")
#             print(f"Content: '{doc.page_content}'")
#             print(f"Metadata: {doc.metadata}")
#     else:
#         print("\nFailed to load or split documents.")
#
#     # Clean up dummy file and directory
#     # import shutil
#     # shutil.rmtree(test_workspace)
#     # print(f"\nCleaned up test workspace: {test_workspace}")

