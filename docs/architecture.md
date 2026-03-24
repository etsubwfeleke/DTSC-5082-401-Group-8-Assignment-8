# System Architecture
## Team: Group 8
## Date: March 24, 2026
## Members and Roles:
- Corpus Architect: Akhil Sai Yalavarthi
- Pipeline Engineer: Etsub Feleke
- UX Lead: Hoda Malak
- Prompt Engineer: Sreekanth Taduru
- QA Lead: Team Effort

---

## Architecture Diagram
The diagram must show:
- [ ] How a corpus file becomes a chunk
- [ ] How a chunk becomes an embedding
- [ ] How duplicate detection fires
- [ ] How a user query flows through LangGraph to a response
- [ ] Where the hallucination guard sits in the graph
- [ ] How conversation memory is maintained across turns

<img width="562" height="1003" alt="Flow_Diagram drawio" src="https://github.com/user-attachments/assets/fb4cf821-0110-4e8c-b32f-3cdb111dd11e" />



*(replace this line with your diagram image or ASCII art)*

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/context/`
- **File formats used:** PDF file formats are used
- **Landmark papers ingested:**
  - ARTIFICIAL NEURAL NETWORKS AND THEIR APPLICATIONS - Nitin Malik
  - An Overview of Controllable Text Generation via Variational Auto-Encoders - Haoqin Tu, Yitong Li
  - Gradient-based learning applied to Document recognition - Yann LeCun, Leon Bottou
  - Generative Adversarial Nets - Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Aaron Courville, Yoshua Bengio
  - Long Short-Term memory - Sepp Hochreiter, Jürgen Schmidhuber
  - A DYNAMICAL SYSTEM VIEW ON RECURRENT NEURAL NETWORKS - Minmin Chen, Eldad Haber, Bo Chang, Ed H. Chi
  - Sequence to Sequence Learning with Neural Networks - Ilya Sutskever, Oriol Vinyals, Quoc V. Le 
  


- **Chunking strategy:**
  *Documents are split into chunks of size 512 characters with 50 characters overlapping: 50 tokens prevent concepts that span chunk boundaries, and 512 tokens balance         context richness with retrieval precision.
  *

- **Metadata schema:**
  *(list every metadata field your chunks carry and explain why each field exists)*
  | Field | Type | Purpose |
  |---|---|---|
  | topic | string | identifies the neural network topic |
  | difficulty | string | difficulty level as advanced or intermediate|
  | type | string | describes content type |
  | source | string | original PDF file name |
  | related_topics | list | supports cross-topic retrieval |
  | is_bonus | bool | True for advanced topics like GAN |

- **Duplicate detection approach:**
  *Chunk ID is generated using a content hash through the VectorStoreManager.generate_chunk_id() function. This approach is more reliable than filename-
   based detection because filenames do not guarantee unique content.*

- **Corpus coverage:**
  - [1] ANN
  - [1] CNN
  - [1] RNN
  - [1] LSTM
  - [1] Seq2Seq
  - [1] Autoencoder
  - [0] SOM *(bonus)*
  - [0] Boltzmann Machine *(bonus)*
  - [1] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB — PersistentClient
- **Local persistence path:** ./data/chroma/

- **Embedding model:** all-MiniLM-L6-v2 via sentence-transformers

- **Why this embedding model:**
  - *provides a strong balance between performance and efficiency, and generates high-quality semantic embedding, remaining lightweight enough to run
  -  locally without requiring an external API 

- **Similarity metric:**
  - *Cosine similarity because It measures the angle between embedding vectors, and makes it effective for comparing semantic similarity
  - regardless of vector magnitude.*

- **Retrieval k:**
  - *k = 4 because increasing k can improve recall but may reduce answer precision.k as 4 will be sufficient for the language model.*

- **Similarity threshold:**
  - **0.7 (or whatever your settings.similarity_threshold is — usually 0.7–0.75). This prevents low-relevance results from being used, acting as a hallucination guard

- **Metadata filtering:**
  - *yes users can filter by topic or difficulty, which is done by metadata filtering, and is implemented using ChromaDB's WHERE clause during query execution.*

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  *(describe what each node does in one sentence)*
  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | This node rewrites the user's query to improve retrieval accuracy. |
  | retrieval_node | This node fetches relevant chunks from the vector store based on the rewritten query. |
  | generation_node | This node generates a response based on the retrieved chunks and the original query. |

- **Conditional edges:**
  After `retrieval_node` runs, `should_retry_retrieval` checks `state.no_context_found`. If True (no chunks met the similarity threshold), it routes to "end" — skipping generation entirely. If False, it routes to "generate", which runs `generation_node`. The end path returns the hallucination guard message directly.

- **Hallucination guard:**
```python
"""I was unable to find relevant information in the study corpus for your query.

This may mean:
- The topic is not yet covered in the corpus (check if it is a bonus topic)
- Your query needs to be more specific (try including the exact topic name)
- The corpus needs more content on this area

Suggested next steps:
- Rephrase your query with specific deep learning terminology
- Check which topics are available using the corpus browser
- If you are the Corpus Architect, consider adding content on this topic

Topics currently available: ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder
Bonus topics (if ingested): SOM, Boltzmann Machines, GAN"""
```
- **Query rewriting:**
  *(give one example of a raw user query and how your system rewrites it)*
  - Raw query: `"how do you fix the vanishing gradient problem in cnns"`
  - Rewritten query: `"vanishing gradient problem CNNs recurrent neural networks long short-term memory exploding gradients optimization techniques"`

- **Conversation memory:**  
  History is maintained using LangGraph's MemorySaver checkpointer, keyed by thread_id stored in st.session_state. Each new query appends to the existing message list for that thread. When the context window fills up, trim_messages with strategy="last" drops the oldest messages while keeping the most recent turns and never dropping the system prompt.

- **LLM provider:**
  *(which provider did your team use — Groq, Ollama, or LM Studio? which model?)*
  - Groq — llama-3.1-8b-instant

- **Why this provider:**  
  Groq was chosen for its free tier and extremely low latency via its custom LPU (Language Processing Unit) chip. No local GPU is required, making it accessible to all team members without hardware constraints.

---

### Prompt Layer

- **System prompt summary:**  
  The agent persona is a senior ML engineer running a technical interview prep session. Key constraints: answer only from provided context (no general knowledge), always cite sources using `[SOURCE: topic | filename]` format, adjust depth to chunk difficulty metadata, acknowledge what's correct before explaining gaps, and never guess beyond what the context states. Tone is technically precise and encouraging but rigorous.

- **Question generation prompt:**
  - Inputs: `{context}` (retrieved chunk text), `{difficulty}` (from the UI dropdown)
  -  Returns a JSON object with: `question`, `difficulty`, `topic`, `model_answer`, `follow_up`, `source_citations`
  - The question must require genuine understanding (not recall), be open-ended, and connect at least two concepts if possible

- **Answer evaluation prompt:**
  - Inputs: `{question}`, `{candidate_answer}`, `{context}`
  - Returns JSON with: `score` (0–10), `what_was_correct`, `what_was_missing`, `ideal_answer`, `interview_verdict` (hire/consider/no hire), `coaching_tip`
  - Scoring rubric: 9–10 = complete and senior-ready, 7–8 = mostly correct with minor gaps, 5–6 = core concept understood but details missing, 3–4 = partial with misconceptions, 0–2 = fundamental misunderstanding

- **JSON reliability:**  
  Both `QUESTION_GENERATION_PROMPT` and `ANSWER_EVALUATION_PROMPT` end with: *"Respond with the JSON object only. No preamble or explanation."* The double-brace `{{}}` syntax escapes the curly braces so Python's `.format()` doesn't misinterpret them as placeholders.

- **Failure modes identified:**  
  - System prompt: Model occasionally adds knowledge not in the chunk. Constraint "Do not use your general knowledge" + "Do not guess, infer beyond what is stated" was added to address this.
  - Question generation: Model wraps JSON in markdown backticks despite instruction. The `"No preamble or explanation"` line addresses most cases; parsing is wrapped in try/except as a fallback.
  - Answer evaluation: Model tends toward generous scores (7–8) for vague answers. The explicit rubric with labeled bands (hire/consider/no hire) anchors the score to observable criteria.
---

### Interface Layer

- **Framework:** Streamlit 
- **Deployment platform:** Streamlit Community Cloud
- **Public URL:** *(paste your deployed app URL here once live)*

- **Ingestion panel features:**
  *Left sidebar* : Shows a file uploader accepting  `.pdf` and `.md ` files (up to 200MB). After clicking "Ingest Documents", shows a status summary (ingested, skipped duplicates, errors). Below that, lists all ingested documents with delete buttons.

- **Document viewer features:**
  *Center panel* : Dropdown to select a document, then displays each chunk with its metadata tags (topic, difficulty). Shows total chunk count at the bottom.

- **Chat panel features:**
  *Right panel* : Includes "Focus Topic" and "Difficulty Level" dropdowns to filter retrieval. Chat history displayed above. Each response shows source citations as `[CITE: topic | filename]` with an expandable "View Source Citations" section. If the hallucination guard fires, displays the no-context message instead of an answer.


- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | chat_history |  list of `{"role": "user" \ "assistant", "content": str}`|
  | ingested_documents | list of dicts from `list_documents()` |
  | selected_document | Source filename currently in viewer dropdown |
  | thread_id | UUID string used as the MemorySaver Checkpoint key for conversation memory |

- **Stretch features implemented:**  
Graph-level state streaming via `graph.stream(stream_mode="values")` — the UI polls for state updates between LangGraph nodes and renders the final answer when generation_node completes. A typing cursor (▌) is shown during processing. Full token-by-token streaming was not implemented.


---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:** Chunk size of 512 characters with 50-character overlap

   **Rationale:**
   - 512 characters ≈ 128 tokens, fitting comfortably within all-MiniLM-L6-v2's 256-token context window
   - Provides sufficient semantic context for embedding quality without dilution
   - 50-character overlap (~10%) prevents concept fragmentation at chunk boundaries
   - Trade-off analysis:
     * Smaller (256 chars): Lost contextual meaning, retrieval precision dropped
     * Larger (1024 chars): Diluted similarity scores, multiple concepts per chunk
   - Validated through manual testing with sample LSTM chunks

   **Interview answer:**
   *"We set 512 characters because our embedding model, all-MiniLM-L6-v2, has a 256-token context limit. 512 chars translates to roughly 128 tokens, leaving headroom for safety. The 50-character overlap ensures that concepts spanning boundaries—like 'LSTM forget gate mechanism'—remain intact in at least one chunk. We tested smaller and larger sizes: 256 lost too much context, while 1024 diluted similarity scores by mixing multiple concepts. 512 struck the optimal balance between semantic richness and retrieval precision."*

2. **Decision:** Cosine similarity as the distance metric for vector search

   **Rationale:**
   - Cosine measures angular similarity between embedding vectors, making it magnitude-invariant
   - Critical for text embeddings where document length shouldn't affect relevance scoring
   - all-MiniLM-L6-v2 is optimized for cosine similarity (trained with cosine loss)
   - ChromaDB converts cosine distance to similarity: `score = 1 - distance`
   - Alternative (L2/Euclidean) would penalize longer documents unfairly
   - Threshold set at 0.3 after manual calibration against sample queries

   **Interview answer:**
   *"We use cosine similarity because it measures the angle between embedding vectors, not their magnitude. This matters for text—a 200-word chunk and a 150-word chunk can have the same semantic meaning but different vector magnitudes. Cosine ignores magnitude and focuses purely on direction, which is exactly what we want for semantic similarity. Our embedding model, all-MiniLM-L6-v2, was also trained using cosine similarity loss, so it performs optimally with this metric. We set the threshold at 0.3 through manual testing—anything below that produced irrelevant results."*
3. **Decision:** Content-hash based chunk IDs for duplicate detection

   **Rationale:**
   - Chunk ID = SHA-256 hash of `{source_filename}::{chunk_text}` (truncated to 16 chars)
   - Deterministic: same content always produces same ID across uploads
   - Detects duplicates even when:
     * File is renamed
     * File is uploaded from different path
     * Upload occurs in different session
   - Alternative (timestamp/UUID) would fail to detect re-uploads of identical content
   - Hash collision probability: ~1 in 10^19 with 16-char hex (acceptable for corpus size)
   - Implemented in `VectorStoreManager.generate_chunk_id()` and enforced in `ingest()`

   **Interview answer:**
   "We generate chunk IDs by hashing the source filename and chunk text together using SHA-256. This makes IDs deterministic—uploading the same file twice produces identical IDs, allowing our duplicate detection to work reliably. The alternative would be using timestamps or UUIDs, but those would fail to catch re-uploads of the same content. We truncate the hash to 16 characters because the collision probability is negligible for our corpus size, and it keeps IDs readable in logs. This approach ensures we never ingest duplicate content, even across sessions or filename changes."

4. **Decision:** Similarity threshold of 0.3 for hallucination guard

   **Rationale:**
   - Acts as the hallucination guard: chunks scoring below 0.3 are excluded from LLM context
   - Calibrated through manual testing with diverse query types:
     * On-topic queries (e.g., "LSTM gates"): scores 0.6-0.9 → passed threshold
     * Related queries (e.g., "RNN memory"): scores 0.35-0.55 → passed threshold
     * Off-topic queries (e.g., "Roman Empire"): scores 0.05-0.15 → correctly rejected
   - Trade-off:
     * Lower threshold (0.1): would allow weakly related content, increasing noise
     * Higher threshold (0.5): would reject borderline-relevant content, reducing recall
   - Production improvement: empirical calibration using labeled relevance dataset
   - Implemented in `retrieval_node()` via score filtering before generation

   **Interview answer:**
   "Our similarity threshold of 0.3 is the core of our hallucination guard. When a query retrieves no chunks above 0.3, we return a clear 'no context found' message instead of letting the LLM answer from general knowledge. We set 0.3 through manual testing—on-topic queries consistently scored above 0.6, while off-topic queries like 'history of Rome' scored below 0.2. A lower threshold would let noise through; a higher one would reject marginal but relevant content. In production, we'd validate this empirically with a labeled test set and precision-recall analysis, but 0.3 works well for our current corpus."

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query ("How do LSTMs help RNNs") | Relevant chunks, source cited | Retrieved 3 chunks from RNN.pdf, cited [SOURCE: rnn \| RNN.pdf], accurate answer explaining cell state and gates | PASS |
| Off-topic query ("History of Rome") | No context found message | System correctly returned no results (threshold guard activated) | PASS |
| Duplicate ingestion (upload same PDF twice) | Second upload skipped | First upload: ingested=45 chunks, Second upload: skipped=45, no duplicates created | PASS |
| Empty query (submit blank input) | Graceful error, no crash | Streamlit chat_input validation prevented submission, no error thrown | PASS |
| Cross-topic query ("Does activation functions improve vanishing gradient problem") | Multi-topic retrieval | Retrieved chunks from RNN.pdf covering both activation functions and gradient problem, synthesized answer across concepts | PASS |

**Critical failures fixed before Hour 3:**
**Critical failures fixed before Hour 3:**

1. **ChromaDB Dimension Mismatch Error**
   - **Symptom:** `chromadb.errors.InvalidArgumentError: collection expects dim 3, got 384`
   - **Root cause:** Test embeddings (3-dimensional fake vectors) were mixed with production embeddings (384-dimensional all-MiniLM-L6-v2)
   - **Impact:** Complete ingestion failure; system unusable
   - **Fix:** Added `_raise_dimension_mismatch()` method in `store.py` with clear error message:
```python
     raise RuntimeError(
         "Chroma collection embedding dimension does not match the active "
         "embedding model. This usually happens when the same collection was "
         "previously populated by a different embedding backend..."
     )
```
   - **Prevention:** Clear instructions to delete/reset collection when switching embedding models

2. **Uploaded Documents Showing Temporary Filenames**
   - **Symptom:** Documents appeared in corpus as `tmpXYZ.pdf` instead of original filename (e.g., `RNN.pdf`)
   - **Root cause:** File uploader created temporary files without preserving original metadata
   - **Impact:** Source citations were meaningless (e.g., `[SOURCE: rnn | tmp3k2j5.pdf]`)
   - **Fix:** Two-part solution:
```python
     # In app.py - pass original filename as metadata override
     chunks = chunker.chunk_file(
         tmp_path,
         metadata_overrides={"source": uploaded_file.name}  # ← Preserves original name
     )
     
     # In chunker.py - use override source for metadata inference
     metadata_path = Path(source_name) if source_name else file_path
     metadata = self._infer_metadata(metadata_path, metadata_overrides)
```
   - **Verification:** Source citations now display as `[SOURCE: rnn | RNN.pdf]`

3. **LangGraph State Access Failure in Nodes**
   - **Symptom:** `AttributeError: 'dict' object has no attribute 'messages'`
   - **Root cause:** LangGraph passes state as dict during streaming but object during invoke; nodes assumed object-only access
   - **Impact:** Chat interface crashed on first query
   - **Fix:** Created dict-safe state accessor in `nodes.py`:
```python
     def _state_get(state: AgentState, key: str, default=None):
         """Read a field from either dict-like or attribute-style graph state."""
         if isinstance(state, dict):
             return state.get(key, default)
         return getattr(state, key, default)
```
   - **Applied to:** All state reads in `query_rewrite_node`, `retrieval_node`, and `generation_node`


**Known issues not fixed (and why):**
**Known issues not fixed (and why):**

1. **PDF Reference Sections Create Low-Quality Chunks**
   - **Issue:** Academic PDFs produce chunks like "References\nBengio, Y., Simard, P., & Frasconi, P. (1994)..." with no semantic value
   - **Impact:** ~15-20% of PDF-sourced chunks are citation noise, diluting retrieval precision
   - **Why not fixed:** Would require either:
     * Custom PDF parsing with section detection (complex, out of scope)
     * ML-based chunk quality classifier (training data unavailable)
   - **Workaround:** Manual corpus curation by Corpus Architect to remove low-value chunks
   - **Production fix:** Post-processing pipeline with regex-based reference section detection

2. **No Persistent Conversation Memory Across App Restarts**
   - **Issue:** LangGraph `MemorySaver` stores conversation in RAM only; all chat history lost on app restart
   - **Impact:** Users lose context mid-conversation if Streamlit reruns/restarts
   - **Why not fixed:** 
     * `MemorySaver` designed for ephemeral sessions
     * Disk persistence requires `SqliteSaver` or `RedisSaver` (out of scope for MVP)
   - **Workaround:** Document limitation in UI or add "Export Chat" button
   - **Production fix:** Replace with `SqliteSaver` for durable checkpoint storage

3. **Single Embedding Model Lock-In**
   - **Issue:** Switching from `all-MiniLM-L6-v2` to another model requires full corpus re-ingestion
   - **Impact:** Cannot A/B test different embedding models without wiping database
   - **Why not fixed:** 
     * ChromaDB collections store embedding dimension at creation time
     * No built-in migration or backward compatibility
   - **Workaround:** Use separate collection names for different embedding models
   - **Production fix:** Versioned collections (e.g., `corpus_v1_minilm`, `corpus_v2_bge`) or automated re-embedding script

4. **Query Rewriting May Miss Synonym Variations**
   - **Issue:** Single query rewrite can't capture all semantic variations (e.g., "backprop" vs "backpropagation" vs "gradient descent update rule")
   - **Impact:** Some valid queries may retrieve fewer relevant chunks than possible
   - **Why not fixed:** 
     * Query expansion (generating multiple variants) would multiply retrieval cost 3-5x
     * Requires careful deduplication logic
   - **Workaround:** Users can manually rephrase queries if first attempt yields low results
   - **Production fix:** Multi-query retrieval with Reciprocal Rank Fusion (RRF) to merge resultswh
---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

1. **Flash of Unstyled Response During Streaming**
   - **Symptom:** When asking a second question, the previous response briefly flashes/glitches before the new response streams in
   - **Root cause:** Streamlit's `st.rerun()` triggers a full script re-execution, causing chat history to re-render momentarily before streaming begins
   - **Impact:** Jarring user experience; looks unpolished during demos
   - **Technical detail:** The chat container re-renders all previous messages from `st.session_state.chat_history` before the new streaming response starts
2. **No Conversation Reset Functionality**
   - **Symptom:** Users cannot clear chat history or start fresh without refreshing the entire page
   - **Impact:** Long conversations bloat the UI; users can't quickly test different query patterns
   - **Missing feature:** "Clear Chat" or "New Session" button in the UI
   - **Workaround:** Refresh browser page (loses all state including ingested documents)

3. **No Topic/Subtopic Navigation for Focused Learning**
   - **Symptom:** Users must manually type queries; no guided exploration of corpus structure
   - **Missing feature:** Hierarchical topic browser (e.g., "RNN → LSTM → Gates → Forget Gate")
   - **Impact:** Users don't know what content is available; reduces discoverability
   - **Current workaround:** Use topic_filter dropdown, but requires knowing topics beforehand

4. **PDF Reference Sections Create Low-Quality Chunks**
   - **Issue:** Academic PDFs produce chunks like "References\nBengio, Y., Simard, P., & Frasconi, P. (1994)..." with no semantic value
   - **Impact:** ~15-20% of PDF-sourced chunks are citation noise, diluting retrieval precision
   - **Root cause:** PyPDFLoader treats all text equally; no section-aware filtering
   - **Current workaround:** Manual corpus curation to remove noisy chunks

5. **No Persistent Conversation Memory Across App Restarts**
   - **Issue:** LangGraph `MemorySaver` stores conversation in RAM only; chat history lost on restart
   - **Impact:** Users lose context mid-conversation if app crashes or is redeployed
   - **Root cause:** `MemorySaver` designed for ephemeral sessions

6. **Single Embedding Model Lock-In**
   - **Issue:** Switching from `all-MiniLM-L6-v2` to another model requires full corpus re-ingestion
   - **Impact:** Cannot A/B test different embedding models without wiping database
   - **Root cause:** ChromaDB collections store embedding dimension at creation time

---
## What We Would Do With More Time
#### 1. **Interactive Quiz Mode** 
**Problem:** Users can ask questions but can't test their knowledge systematically

**Solution:** Add a "Quiz Me" mode that:
- Generates multiple-choice or short-answer questions from ingested content
- Tracks score across questions
- Provides immediate feedback with source citations
- Adapts difficulty based on performance (spaced repetition)

**Implementation:**
```python
# New LangGraph node: quiz_generation_node
def quiz_generation_node(state: AgentState) -> dict:
    """Generate quiz question from retrieved chunk"""
    chunk = random.choice(state.retrieved_chunks)
    
    quiz_prompt = f"""
    Based on this content: {chunk.chunk_text}
    
    Generate a quiz question with:
    - 1 correct answer
    - 3 plausible distractors
    - Explanation of why correct answer is right
    
    Return JSON: {{"question": "...", "options": ["A", "B", "C", "D"], 
                   "correct": "A", "explanation": "..."}}
    """
    # LLM call + JSON parsing
```

**UI Addition:**
- " Quiz Mode" tab next to " Chat"
- Score tracker: "Correct: 8/10 | Current Streak: 3"
- Difficulty selector: Beginner / Intermediate / Advanced

**Expected impact:** Transforms passive Q&A into active learning; increases user engagement 3-5x

---

#### 2. **Code Practice Module**
**Problem:** System explains concepts but doesn't let users practice implementation

**Solution:** Interactive code editor for implementing deep learning concepts

**Implementation:**
```python
# New panel in UI
with st.expander(" Code Practice"):
    problem = st.selectbox("Challenge", [
        "Implement forward pass for LSTM",
        "Code backpropagation for simple ANN",
        "Build Conv2D layer from scratch"
    ])
    
    # Monaco code editor (via streamlit-ace or custom component)
    user_code = st_ace(language='python', theme='monokai')
    
    if st.button("Run Tests"):
        # Execute in sandboxed environment
        result = run_code_tests(user_code, problem)
        st.write(result.passed, result.failed, result.hints)
```

**Test Cases Example (LSTM Forward Pass):**
```python
test_cases = [
    {"input": np.array([[1, 2, 3]]), "expected_shape": (1, hidden_dim)},
    {"input": np.array([[0, 0, 0]]), "expected_output": "close_to_zero"},
]
```

**Safety:** Run in Docker container with resource limits + timeout

**Expected impact:** Bridges gap between theory and implementation; critical for technical interviews

---

#### 3. **Topic/Subtopic Guided Navigation** 
**Problem:** Users don't know what content is available; no guided exploration

**Solution:** Hierarchical topic browser with auto-generated queries

**Implementation:**
```python
# Sidebar navigation tree
with st.sidebar.expander(" Explore Corpus"):
    # Build topic tree from metadata
    topic_tree = store.get_topic_hierarchy()
    # Returns: {"ANN": ["Forward Pass", "Backprop", "Activation Functions"], 
    #           "LSTM": ["Gates", "Cell State", "Applications"], ...}
    
    selected_topic = st.selectbox("Topic", topic_tree.keys())
    selected_subtopic = st.selectbox("Subtopic", topic_tree[selected_topic])
    
    if st.button("Learn About This"):
        # Auto-generate contextual query
        query = f"Explain {selected_topic} {selected_subtopic} in detail with examples"
        # Populate chat input or auto-submit
```

**Metadata Enhancement Needed:**
```python
# In ChunkMetadata, add:
subtopic: str  # e.g., "Gates", "Training", "Architecture"
keywords: list[str]  # e.g., ["forget gate", "cell state", "LSTM"]
```

**Expected impact:** Reduces friction for new users; increases corpus discoverability


#### 4. **Hybrid Search (BM25 + Vector)** 
**Problem:** Vector search fails on exact keyword matches (e.g., "LSTM" vs "Long Short-Term Memory")

**Solution:** Add BM25 sparse index alongside ChromaDB, merge with Reciprocal Rank Fusion

**Implementation:**
```python
from rank_bm25 import BM25Okapi

# At ingestion time
bm25_index = BM25Okapi([chunk.chunk_text.split() for chunk in all_chunks])

# At query time
def hybrid_search(query: str, k: int = 4):
    # Get top-20 from each retriever
    vector_results = vector_store.query(query, k=20)
    bm25_scores = bm25_index.get_scores(query.split())
    bm25_results = get_top_k(bm25_scores, k=20)
    
    # Merge with RRF
    merged = reciprocal_rank_fusion([vector_results, bm25_results], k=4)
    return merged
```

**Expected improvement:** +15-20% recall on keyword-heavy queries

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:** What is the vanishing gradient problem in RNNs and how does the LSTM architecture specifically address it?   
Model answer: *In standard RNNs, gradients are multiplied through many time steps via backpropagation through time. When weights are less than 1, repeated multiplication makes gradients exponentially small the network stops learning long-range dependencies. LSTMs address this with a cell state that flows through time with only additive updates (controlled by gates), not multiplicative ones. The forget gate learns what to discard, the input gate learns what to add, and the output gate controls what to expose to the next hidden state. The additive cell state pathway preserves gradient signal across long sequences.*

**Question 2:** How does a Seq2Seq model use an encoder-decoder architecture with an LSTM, and what is the bottleneck problem it introduces?  
Model answer: *The encoder reads the input sequence and compresses it into a fixed-size context vector (the final hidden state). The decoder uses this vector as its initial state to generate the output sequence token by token. The bottleneck problem is that the entire input sequence regardless of length must be represented in this single fixed-size vector. Long sequences lose information. This motivated the attention mechanism, which allows the decoder to attend directly to all encoder hidden states rather than just the final one.*

**Question 3:** Compare CNNs and ANNs for image classification why does a fully-connected ANN fail where a CNN succeeds?  
Model answer: *A fully-connected ANN treats every pixel as an independent input feature, which means it learns no spatial structure and has an enormous number of parameters (for a 256×256 image: 65,536 inputs per neuron). It also has no translation invariance a cat in the top-left corner and a cat in the bottom-right corner look like completely different inputs. CNNs solve both problems: convolutional filters learn local spatial patterns (edges, textures, shapes) and share weights across the image, drastically reducing parameters. Max pooling adds translation invariance. The hierarchical structure means early layers detect edges and later layers detect complex shapes.*

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
- **LangGraph's explicit state management** — Coming from black-box agent frameworks, seeing every state transition as a named node made debugging dramatically easier. When retrieval failed, we knew exactly which node to inspect.
- **Separation of concerns paid off** — Having distinct layers (corpus → chunker → vector store → agent → UI) meant bugs were isolated. When PDF chunking broke, we didn't have to touch the retrieval logic.
- **Streaming responses elevate perceived quality** — The difference between blocking for 3 seconds vs streaming tokens is night-and-day for UX, even though backend latency is identical.
- **Content-hash duplicate detection is more robust than filename-based** — This clicked during testing when re-uploading renamed files correctly skipped chunks.

**What confused us:**
- **ChromaDB's embedding dimension lock-in** — We didn't realize collections are permanently typed by the first embedding dimension. Switching from test vectors (3D) to production embeddings (384D) required a full database reset. This was a painful lesson in schema migration.
- **LangGraph state representation differences** — State is a dict during `stream()` but an object during `invoke()`. This caused `AttributeError: dict has no attribute 'messages'` until we built dict-safe accessors. The inconsistency still feels like a framework rough edge.
- **Why similarity threshold calibration matters so much** — Small changes (0.3 → 0.4) drastically affected retrieval behavior. We expected embeddings to be more forgiving. Realizing this needs empirical validation, not guesswork, was humbling.
- **PDF chunking is messier than expected** — We thought PyPDFLoader would handle academic papers cleanly. Reference sections, headers, and equations produced 15-20% noise chunks. This taught us that raw text extraction ≠ semantic content extraction.

**One thing each team member would study before a real interview:**

- **Corpus Architect:**  
  "How to evaluate chunk quality systematically. I'd study intrinsic metrics like semantic coherence (average cosine similarity within a chunk vs. between chunks) and extrinsic metrics like retrieval precision@k on labeled test sets. Also, advanced chunking strategies like semantic splitting (split on topic shifts, not fixed character counts) using sentence transformers to detect boundaries."

- **Pipeline Engineer:**  
  "Embedding model selection and evaluation. I'd study the landscape beyond all-MiniLM-L6-v2: when to use BGE, E5, or domain-specific models; how to benchmark on BEIR or MTEB; and the trade-offs between model size, latency, and quality. Also, production RAG patterns like hybrid search (BM25 + dense), query decomposition, and parent-child chunking."

- **UX Lead:**  
  "Async patterns in Streamlit and how to build non-blocking UIs. I'd study background tasks (using stqdm or threading), Server-Sent Events for real-time updates, and when to switch to FastAPI + React for complex interactions. Also, accessibility fundamentals—keyboard navigation, screen reader support, and WCAG compliance."

- **Prompt Engineer:**  
  "Prompt evaluation frameworks and systematic testing. I'd study DSPy for prompt optimization, how to build labeled test sets for few-shot examples, and metrics like BLEU/ROUGE for generated text quality. Also, structured output techniques beyond JSON—using grammar constraints (like Outlines or LMQL) to guarantee valid responses."

- **QA Lead:**  
  "RAG evaluation methodologies. I'd study RAGAS (RAG Assessment framework), how to measure context relevance, answer faithfulness, and answer relevance separately. Also, adversarial testing for RAG—injection attacks, context poisoning, and hallucination triggers. Finally, load testing retrieval systems and understanding ChromaDB's performance characteristics under concurrent queries."