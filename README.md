diff --git a/README.md b/README.md
index cefc123eb7e9f7662b4ef0d822b9d9631603047d..d5fd4f1c3a7c87f72d625046e28c1eeb1a814fbb 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,75 @@
-# appraiser-app
-Real estate appraisal submission app
+# Creator's Companion (Local Setup)
+
+If your setup only has `server.mjs` and the HTML file, that is totally fine.
+
+## Where to put `README.md`
+
+Put `README.md` in the **same root folder** as your server file.
+
+Example:
+
+```txt
+your-project/
+├─ server.mjs
+├─ node-chatbot.html
+└─ README.md
+```
+
+## Memory behavior (important)
+
+The server is now tuned for RPG usage:
+
+- **Normal conversation should NOT be auto-saved** unless it looks like a clear world fact.
+- **Explicit memory commands always save** and return a deterministic confirmation.
+- **Memory questions return deterministic memory readouts** so stored facts are shown reliably.
+
+Use explicit commands like:
+
+- `remember that Solven is the main character`
+- `save this: The Ashen Order betrayed the crown`
+- `store this: The party starts in Blackreach`
+- `note that Mira is the healer`
+
+For memory recall, ask:
+
+- `what is in memory right now`
+- `what do you remember about the rpg`
+- `/lore` and `/characters`
+
+Debug endpoint to verify model↔memory link:
+
+- `GET /memory/debug-link?q=what do you remember`
+
+## Is `README.md` required?
+
+No. The app runs without it.
+
+## Minimal run checklist
+
+1. Install dependencies in your project folder.
+2. Keep `server.mjs` and `node-chatbot.html` together in the same folder.
+3. Run the server from that same folder.
+4. Open:
+   - `http://localhost:3000/` (redirects to UI), or
+   - `http://localhost:3000/node-chatbot.html`
+
+## Memory pipeline order (yes, mostly like your diagram)
+
+Current server flow is:
+
+1. User message received (`POST /chat`)
+2. Load memory from disk (`loadMemory`)
+3. Optional save step (if message looks like a memory command/fact):
+   - extract candidates
+   - score candidates
+   - save accepted entries
+4. Branch:
+   - explicit memory command (`remember/save/store/note`) → deterministic “Saved to memory” reply
+   - memory question (“what do you remember…”) → deterministic memory readout
+   - normal chat → continue below
+5. Select relevant memory (`retrieveMemories`)
+6. Inject memory into system prompt (`buildSystemPrompt`)
+7. Model generates response
+8. Persist updated memory/chat files
+
+So your diagram is correct for normal chat, with two deterministic branches added for reliability.
