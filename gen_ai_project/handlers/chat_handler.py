# handlers/chat_handler.py

import logging
import asyncio # Added for create_task
from typing import Dict, Any

# --- LangChain ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage, MessagesPlaceholder # Added MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

# --- Project Imports ---
# Assuming this file is in handlers/, core is in core/
from ..core.core_services import CoreAgentServices

class ChatHandler:
    """Handles interactive chat requests using CoreAgentServices."""
    def __init__(self, services: CoreAgentServices):
        self.services = services
        self.logger = logging.getLogger(__name__)
        # Define the core prompt structure for the agent executor used in chat
        # This prompt now lives within the handler that uses the executor for chat
        system_prompt = """You are a helpful and comprehensive assistant.
You have access to tools for database queries, visualization, modeling, forecasting, reporting, file management, and knowledge management.
You also use an internal knowledge base (retrieved context) and conversation memory (past interactions).
Prioritize using tools for specific tasks (SQL, plotting, modeling, forecasting, files, reporting, adding knowledge).
Use your knowledge base and memory for general questions or information retrieval.
Synthesize information from retrieved context, memory, and tool outputs into a clear, final answer for the user.
Indicate when using knowledge/memory. State if you cannot fulfill a request."""

        self.chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history", optional=True), # Placeholder for memory
                MessagesPlaceholder("retrieved_context", optional=True), # Placeholder for RAG context
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"), # For agent intermediate steps
            ]
        )
        self.logger.info("ChatHandler initialized.")

    async def handle_chat(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """
        Processes a user chat message, performs RAG/Memory retrieval,
        invokes the agent executor, synthesizes the final response, and updates memory.
        """
        self.logger.info(f"ChatHandler processing request for user {user_id}")
        self.logger.debug(f"User Input: {user_input}")

        agent_response_dict = {}
        final_answer = "An error occurred during processing."
        run_config = RunnableConfig(configurable={"user_id": user_id}) # Pass user_id if needed

        try:
            # 1. Retrieve context and memory concurrently
            # --- Use KnowledgeHandler for context retrieval ---
            context_task = asyncio.create_task(self.services.knowledge_handler.retrieve(user_input))
            memory_task = asyncio.create_task(self.services.retrieve_memory(user_input, user_id))

            # Await retrieval results
            retrieved_context_chunks = await context_task
            relevant_memories = await memory_task

            # Format context and memory for the prompt
            formatted_context = ""
            if retrieved_context_chunks:
                 formatted_context = "Retrieved Context:\n" + "\n\n".join(retrieved_context_chunks)

            formatted_memory = [] # LangChain expects list of BaseMessage for chat_history
            if relevant_memories:
                 # Convert Mem0 results to LangChain message format (approximate)
                 # This might need refinement based on how you want memory presented
                 for mem in reversed(relevant_memories): # Show recent first?
                      # Simple conversion, assumes 'text' contains 'User: ... Assistant: ...'
                      if "User:" in mem['text'] and "Assistant:" in mem['text']:
                           parts = mem['text'].split("Assistant:", 1)
                           user_part = parts[0].replace("User:", "").strip()
                           assistant_part = parts[1].strip()
                           formatted_memory.append(HumanMessage(content=user_part))
                           formatted_memory.append(SystemMessage(content=assistant_part)) # Use SystemMessage or AIMessage
                      else: # Fallback for other memory formats
                           formatted_memory.append(SystemMessage(content=f"[Past interaction]: {mem['text']}"))


            # 2. Prepare input for the agent executor, including context/memory
            agent_input = {
                "input": user_input,
                "chat_history": formatted_memory,
                "retrieved_context": formatted_context # Pass context directly if prompt supports it
            }

            # 3. Run the main agent executor (using shared executor)
            if not self.services.agent_executor:
                 raise RuntimeError("Core Agent Executor not available.")

            # Invoke the agent - it will use tools based on input, history, and context
            # The agent's internal prompt should guide it on how to use these pieces
            agent_response_dict = await self.services.agent_executor.ainvoke(agent_input, config=run_config)
            final_answer = agent_response_dict.get("output", "Agent did not produce a final output.")
            self.logger.debug(f"Agent Executor Output: {final_answer[:200]}...")

            # 4. Add final interaction to Mem0 (async)
            # Use the actual final answer generated by the agent executor
            await self.services.add_memory(
                text=f"User: {user_input}\nAssistant: {final_answer}",
                user_id=user_id,
                agent_id="chat_handler" # Identify the source
            )

            # Prepare final response structure
            agent_response_dict['final_output'] = final_answer
            agent_response_dict['error'] = None # Explicitly set error to None on success

        except Exception as e:
            self.logger.error(f"Error during chat handling for user {user_id}: {e}", exc_info=self.services.verbose)
            final_answer = f"Sorry, an error occurred while processing your request."
            # Optionally include error details in debug logs or a separate field
            # final_answer = f"Sorry, an error occurred: {e}" # Less user-friendly
            agent_response_dict['error'] = str(e)
            agent_response_dict['final_output'] = final_answer

        return agent_response_dict

