# handlers/knowledge_handler.py

import logging
import asyncio
import os
import uuid
from typing import Dict, Any, Optional, List

# --- LangChain ---
# Import Document type hint if needed, though not strictly required for implementation
# from langchain.schema import Document as LangchainDocument

# --- Project Imports ---
from ..core.core_services import CoreAgentServices
from ..utils import config
from ..utils.document_processing import load_and_split_documents # Document loading/splitting utility

class KnowledgeHandler:
    """
    Manages the knowledge base (vector store) interactions, including
    adding new sources and retrieving relevant context.
    """
    def __init__(self, services: CoreAgentServices):
        """
        Initializes the KnowledgeHandler.

        Args:
            services: The shared CoreAgentServices instance, providing access
                      to the vector store collection, embedding function,
                      workspace directory, and async utilities.
        """
        self.services = services
        self.logger = logging.getLogger(__name__)
        if not self.services.collection:
             # This check depends on CoreAgentServices initialization succeeding
             self.logger.critical("KnowledgeHandler cannot initialize: Vector store collection not available in CoreAgentServices.")
             raise ValueError("Vector store collection not available.")
        self.logger.info("KnowledgeHandler initialized.")

    async def add_source(self, source_path: str) -> Dict[str, Any]:
        """
        Loads, splits, embeds, and adds documents from a source path to the knowledge base.

        Args:
            source_path: Path relative to workspace or absolute path to the document or directory.

        Returns:
            A dictionary indicating the status ('success' or 'error') and a message.
        """
        self.logger.info(f"Attempting to add knowledge source: {source_path}")

        # 1. Load and Split Documents (Sync operation in thread)
        try:
            self.logger.debug(f"Loading and splitting documents from '{source_path}'...")
            # Use the utility function via the sync runner from services
            documents = await self.services._run_sync_in_thread(
                load_and_split_documents,
                source_path=source_path,
                workspace_dir=self.services.workspace_dir
                # Pass chunk_size/overlap from config if desired
            )
            if not documents:
                msg = f"No documents were loaded or split from source: {source_path}"
                self.logger.warning(msg)
                return {"status": "error", "message": msg}
            self.logger.info(f"Successfully loaded and split {len(documents)} chunks from '{source_path}'.")

        except Exception as e:
            msg = f"Failed during document loading/splitting for '{source_path}': {e}"
            self.logger.error(msg, exc_info=self.services.verbose)
            return {"status": "error", "message": msg}

        # 2. Prepare for Vector Store Addition
        ids = [str(uuid.uuid4()) for _ in documents]
        contents = [doc.page_content for doc in documents]
        # Ensure metadata is serializable (convert complex objects if necessary)
        metadatas = []
        for doc in documents:
             serializable_meta = {}
             for k, v in doc.metadata.items():
                  try:
                       # Attempt basic check - more robust serialization might be needed
                       if isinstance(v, (str, int, float, bool, list, dict)):
                            serializable_meta[k] = v
                       else:
                            serializable_meta[k] = str(v) # Convert unknown types to string
                  except Exception:
                       serializable_meta[k] = f"Error serializing metadata key '{k}'"
             metadatas.append(serializable_meta)


        # 3. Add to Vector Store (Sync operation in thread)
        try:
            self.logger.info(f"Adding {len(contents)} chunks to vector store collection '{config.COLLECTION_NAME}'...")
            await self.services._run_sync_in_thread(
                self.services.collection.add,
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            msg = f"Successfully added {len(contents)} chunks from '{source_path}' to the knowledge base."
            self.logger.info(msg)

            # 4. Log addition to memory (Optional but good practice)
            try:
                sources_str = ", ".join(list(set(m.get('source', 'unknown') for m in metadatas)))
                mem_text = f"Ingested {len(contents)} chunks into knowledge base from sources: {sources_str}"
                await self.services.add_memory(text=mem_text, user_id="system_knowledge", agent_id="knowledge_handler")
            except Exception as mem_e:
                 self.logger.warning(f"Failed to log knowledge addition to memory: {mem_e}")

            return {"status": "success", "message": msg}

        except Exception as e:
            msg = f"Failed to add documents from '{source_path}' to vector store: {e}"
            self.logger.error(msg, exc_info=self.services.verbose)
            return {"status": "error", "message": msg}


    async def retrieve(self, query: str, n_results: Optional[int] = None) -> List[str]:
        """
        Retrieves relevant document chunks from the knowledge base for a given query.

        Args:
            query: The search query string.
            n_results: The number of results to retrieve. Defaults to config.RAG_RESULTS_COUNT.

        Returns:
            A list of relevant document content strings, or an empty list if none found or error occurs.
        """
        if n_results is None:
            n_results = config.RAG_RESULTS_COUNT # Use default from config

        self.logger.debug(f"Retrieving {n_results} knowledge chunks for query: '{query[:100]}...'")

        try:
            # Query Vector Store (Sync operation in thread)
            results = await self.services._run_sync_in_thread(
                self.services.collection.query,
                query_texts=[query],
                n_results=n_results,
                include=['documents'] # Only fetch the document content
            )

            if results and results.get('documents') and results['documents'][0]:
                retrieved_docs = results['documents'][0]
                self.logger.info(f"Retrieved {len(retrieved_docs)} relevant context chunks.")
                return retrieved_docs
            else:
                self.logger.info("No relevant context found in knowledge base for the query.")
                return []
        except Exception as e:
            self.logger.error(f"Error querying vector store: {e}", exc_info=self.services.verbose)
            return [] # Return empty list on error

    async def stop(self):
        """Placeholder stop method if needed for handler cleanup."""
        self.logger.info("KnowledgeHandler stopping (no specific cleanup actions).")
        # Add cleanup if the handler manages background tasks (unlikely for knowledge)

