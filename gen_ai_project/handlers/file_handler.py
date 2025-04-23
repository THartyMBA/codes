#add later  if:

#You find multiple handlers (not the LLM agent) needing complex, programmatic file manipulation logic that goes beyond simple path joining or existence checks.
#You want to strictly enforce policies or logging for all file system access originating from any part of the system (including specialized agents).
#You find that running sync file I/O in threads (_run_sync_in_thread) becomes a bottleneck and want dedicated async file operations using aiofiles wrapped in a handler.