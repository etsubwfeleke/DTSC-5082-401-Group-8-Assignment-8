# System Architecture
## Team: ___________________
## Date: ___________________
## Members and Roles:
- Corpus Architect: Akhil Sai Yalavarthi
- Pipeline Engineer: Etsub Feleke
- UX Lead: ___________________
- Prompt Engineer: Sreekanth Taduru
- QA Lead: ___________________

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
  **0.7 (or whatever your settings.similarity_threshold is — usually 0.7–0.75). This prevents low-relevance results from being used, acting as a hallucination guard

- **Metadata filtering:**
  *yes users can filter by topic or difficulty, which is done by metadata filtering, and is implemented using ChromaDB's WHERE clause during query execution.*

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  *(describe what each node does in one sentence)*
  | Node | Responsibility |
  |---|---|
  | query_rewrite_node | |
  | retrieval_node | |
  | generation_node | |

- **Conditional edges:**
  *(what condition triggers each edge? what happens when no context is found?)*

- **Hallucination guard:**
  *(exactly what does your system return when similarity threshold is not met?
  paste the message here)*

- **Query rewriting:**
  *(give one example of a raw user query and how your system rewrites it)*
  - Raw query:
  - Rewritten query:

- **Conversation memory:**
  *(how is history maintained across turns? what happens when context window fills up?)*

- **LLM provider:**
  *(which provider did your team use — Groq, Ollama, or LM Studio? which model?)*

- **Why this provider:**
  *(what was the deciding factor for your team?)*

---

### Prompt Layer

- **System prompt summary:**
  *(describe the agent persona and the key constraints in your system prompt)*

- **Question generation prompt:**
  *(what inputs does it take and what does it return?)*

- **Answer evaluation prompt:**
  *(how does it score a candidate answer? what is the scoring rubric?)*

- **JSON reliability:**
  *(what did you add to your prompts to ensure consistent JSON output?)*

- **Failure modes identified:**
  *(list at least one failure mode per prompt and how you addressed it)*
  -
  -
  -

---

### Interface Layer

- **Framework:** *(Streamlit / Gradio)*
- **Deployment platform:** *(Streamlit Community Cloud / HuggingFace Spaces)*
- **Public URL:** *(paste your deployed app URL here once live)*

- **Ingestion panel features:**
  *(describe what the user sees — file uploader, status display, document list)*

- **Document viewer features:**
  *(describe how users browse ingested documents and chunks)*

- **Chat panel features:**
  *(describe how citations appear, how the hallucination guard is surfaced,
  and any filters available)*

- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | chat_history | |
  | ingested_documents | |
  | selected_document | |
  | thread_id | |

- **Stretch features implemented:**
  *(streaming responses, async ingestion, hybrid search, re-ranking, other)*

---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:**
   *(e.g. chunk size of 512 with 50 character overlap)*
   **Rationale:**
   *(why this over alternatives? what would break if you changed it?)*
   **Interview answer:**
   *(write a two sentence answer you could give in a technical screen)*

2. **Decision:**
   **Rationale:**
   **Interview answer:**

3. **Decision:**
   **Rationale:**
   **Interview answer:**

4. **Decision:** *(optional — bonus points in Hour 3)*
   **Rationale:**
   **Interview answer:**

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | | |
| Off-topic query | No context found message | | |
| Duplicate ingestion | Second upload skipped | | |
| Empty query | Graceful error, no crash | | |
| Cross-topic query | Multi-topic retrieval | | |

**Critical failures fixed before Hour 3:**
-
-

**Known issues not fixed (and why):**
-
-

---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- *(e.g. PDF chunking produces noisy chunks from reference sections)*
- *(e.g. similarity threshold was calibrated manually, not empirically)*
- *(e.g. conversation memory is lost when the app restarts)*

---

## What We Would Do With More Time

- *(e.g. implement hybrid search combining vector and BM25 keyword search)*
- *(e.g. add a re-ranking step using a cross-encoder)*
- *(e.g. async ingestion so large PDFs don't block the UI)*

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:**

Model answer:

**Question 2:**

Model answer:

**Question 3:**

Model answer:

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
-

**What confused us:**
-

**One thing each team member would study before a real interview:**
- Corpus Architect:
- Pipeline Engineer:
- UX Lead:
- Prompt Engineer:
- QA Lead:
