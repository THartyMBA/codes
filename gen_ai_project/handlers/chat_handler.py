# handlers/chat_handler.py

import logging
from typing import Dict, Any

# --- LangChain ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage
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
        run_config = RunnableConfig(configurable={"user_id": user_id}) # Pass user_id if needed by tools/memory

        try:
            # 1. Retrieve context and memory concurrently
            context_task = asyncio.create_task(self.services.retrieve_context(user_input))
            memory_task = asyncio.create_task(self.services.retrieve_memory(user_input, user_id))

            # 2. Run the main agent executor (using shared executor)
            if not self.services.agent_executor:
                 raise RuntimeError("Core Agent Executor not available.")

            agent_response_task = asyncio.create_task(
                self.services.agent_executor.ainvoke({"input": user_input}, config=run_config)
            )

            # Await all concurrent tasks
            agent_response_dict = await agent_response_task
            initial_output = agent_response_dict.get("output", "Agent did not produce a direct output.")
            self.logger.debug(f"Initial Agent Output: {initial_output[:150]}...")

            retrieved_context_chunks = await context_task
            retrieved_context = "\n\n".join(retrieved_context_chunks) if retrieved_context_chunks else "No relevant context found."

            relevant_memories = await memory_task
            mem0_context = ""
            if relevant_memories:
                mem0_context = "\n\nRelevant past interactions:\n" + "\n".join([f"- {m['text']}" for m in relevant_memories])

            # 3. Synthesize Final Answer using LLM with all context
            synthesis_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Synthesize the final user response based on the query, initial assistant output, retrieved context, and past interactions. Prioritize context, summarize tool use, incorporate memory. Be concise."),
                HumanMessage(content=f"User Query:\n{user_input}\n\nInitial Assistant Output:\n{initial_output}\n\nRetrieved Context:\n{retrieved_context}\n\nRelevant Past Interactions:\n{mem0_context}\n\nSynthesized Final Answer:")
            ])
            synthesis_chain = synthesis_prompt | self.services.llm | StrOutputParser()
            self.logger.debug("Synthesizing final answer...")
            final_answer = await synthesis_chain.ainvoke({})
            self.logger.debug(f"Synthesized Answer: {final_answer[:150]}...")

            # 4. Add final interaction to Mem0 (async)
            await self.services.add_memory(
                text=f"User: {user_input}\nAssistant: {final_answer}",
                user_id=user_id,
                agent_id="chat_handler_final" # Identify the source
            )

            # Prepare final response structure
            agent_response_dict['final_output'] = final_answer
            agent_response_dict['error'] = None # Explicitly set error to None on success

        except Exception as e:
            self.logger.error(f"Error during chat handling for user {user_id}: {e}", exc_info=self.services.verbose)
            final_answer = f"Sorry, an error occurred: {e}"
            agent_response_dict['error'] = str(e)
            agent_response_dict['final_output'] = final_answer

        return agent_response_dict

