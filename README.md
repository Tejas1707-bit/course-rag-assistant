# 🎓 Course RAG Assistant:-

> An AI-powered Course Assistant that answers questions from course video transcripts using **Retrieval-Augmented Generation (RAG)**, semantic search, vector embeddings, and **Google Gemini AI**.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Tejas1707-bit/course-rag-assistant)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)](https://tejas1707-bit-course-rag-assistant-app-gemini-lipbbs.streamlit.app/)

---

## 📌 Overview:-

**Course RAG Assistant** is a Retrieval-Augmented Generation system designed to help students interact with course content through natural-language questions.

Instead of manually searching through long course videos, the system:

1. Converts course videos into text using **Whisper**.
2. Splits the transcripts into smaller meaningful chunks.
3. Generates vector embeddings for the chunks.
4. Uses **semantic search** to retrieve the most relevant content.
5. Sends the retrieved context to **Google Gemini**.
6. Generates a context-aware answer through a **Streamlit chat interface**.

This approach helps the LLM answer questions using the actual course material rather than relying only on its pretrained knowledge.

---

## 🚀 Features:-

* 🎥 **Course Video Transcription**

  * Converts audio/video content into text using OpenAI Whisper.

* ✂️ **Transcript Chunking**

  * Splits large transcripts into smaller chunks for efficient retrieval.

* 🔎 **Semantic Search**

  * Finds course content that is semantically related to the user's question.

* 🧠 **Vector Embeddings**

  * Represents transcript chunks as numerical vectors for similarity-based retrieval.

* 🤖 **Gemini AI Generation**

  * Uses Google's Gemini model to generate answers from retrieved course context.

* 💬 **Interactive Chat Interface**

  * Provides a simple Streamlit-based interface for asking questions.

* 📚 **Course-Specific Answers**

  * Grounds responses in the uploaded course material.

---

## 🏗️ RAG Architecture:-

```text
                Course Videos
                     │
                     ▼
             ┌───────────────┐
             │    Whisper    │
             │ Transcription │
             └───────┬───────┘
                     │
                     ▼
              Course Transcript
                     │
                     ▼
             ┌───────────────┐
             │    Chunking   │
             └───────┬───────┘
                     │
                     ▼
              Transcript Chunks
                     │
                     ▼
             ┌───────────────┐
             │   Embedding   │
             │    Model      │
             └───────┬───────┘
                     │
                     ▼
              Vector Embeddings
                     │
                     │
User Question ───────┤
                     ▼
             ┌───────────────┐
             │ Semantic      │
             │ Search        │
             └───────┬───────┘
                     │
                     ▼
             Relevant Chunks
                     │
                     ▼
             ┌───────────────┐
             │ Gemini LLM    │
             │ Generation    │
             └───────┬───────┘
                     │
                     ▼
                Final Answer
                     │
                     ▼
             Streamlit Interface
```

---

## 🧩 How RAG Works in This Project:-

The system follows a standard Retrieval-Augmented Generation pipeline.

### 1. Data Collection

Course videos are used as the source of knowledge.

### 2. Speech-to-Text

Whisper converts the audio from course videos into text transcripts.

```text
Video → Audio → Whisper → Transcript
```

### 3. Chunking

Large transcripts are divided into smaller chunks.

This is important because sending an entire long transcript to an LLM is inefficient and can exceed the model's context limitations.

```text
Transcript
    ↓
Chunk 1
Chunk 2
Chunk 3
...
Chunk N
```

### 4. Embedding Generation

Each chunk is converted into a vector representation.

```text
Text Chunk → Embedding Model → Vector
```

These vectors allow the system to compare the meaning of the user's question with the meaning of transcript chunks.

### 5. Semantic Retrieval

When a user asks a question, the question is also converted into an embedding.

The system compares the query embedding with stored transcript embeddings and retrieves the most relevant chunks.

```text
User Question
      ↓
Query Embedding
      ↓
Similarity Search
      ↓
Relevant Transcript Chunks
```

### 6. Gemini Generation

The retrieved chunks are provided as context to Gemini.

Gemini then generates an answer based on the retrieved course material.

```text
Question + Retrieved Context
            ↓
        Gemini AI
            ↓
      Generated Answer
```

---

## 🛠️ Technologies Used

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| **Python**            | Core development                |
| **Whisper**           | Speech-to-text transcription    |
| **Google Gemini**     | LLM-based answer generation     |
| **Vector Embeddings** | Semantic representation of text |
| **NumPy / Pickle**    | Vector storage and processing   |
| **Streamlit**         | Web-based chat interface        |
| **JSON**              | Transcript and chunk storage    |

---

## 📂 Project Structure:-

```text
course-rag-assistant/
│
├── audios/
│   └── Course audio/video files
│
├── jsons/
│   └── Transcript data
│
├── whisper/
│   └── Whisper-related files
│
├── app.py
│   └── Main Streamlit application
│
├── app_gemini.py
│   └── Gemini-powered Streamlit application
│
├── stt.py
│   └── Speech-to-text transcription
│
├── create_all_chunks.py
│   └── Creates chunks from transcripts
│
├── embed_gemini.py
│   └── Generates Gemini embeddings
│
├── check_embedding.py
│   └── Checks embedding generation
│
├── rag_generate.py
│   └── RAG response generation
│
├── chunks.json
│   └── Processed transcript chunks
│
├── embeddings.pkl
│   └── Stored embeddings
│
├── embeddings_gemini.pkl
│   └── Gemini-generated embeddings
│
├── requirements.txt
│   └── Project dependencies
│
└── .gitignore
```

The repository currently follows this general structure, including separate scripts for transcription, chunking, embeddings, RAG generation, and the Streamlit applications.

---

## ⚙️ Installation:-

### 1. Clone the Repository

```bash
git clone https://github.com/Tejas1707-bit/course-rag-assistant.git
cd course-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables:-

Create a `.env` file in the project directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_api_key_here
```

> Never commit API keys or other secrets to GitHub.

---

## ▶️ Running the Application:-

Run the Gemini Streamlit application:

```bash
streamlit run app_gemini.py
```

Then open the local Streamlit URL shown in the terminal.

---

## 🌐 Live Demo:-

Try the deployed application:

**[Course RAG Assistant – Live Demo](https://tejas1707-bit-course-rag-assistant-app-gemini-lipbbs.streamlit.app/)**

---

## 💡 Example Questions

You can ask questions such as:

```text
What is Retrieval-Augmented Generation?

How does semantic search work?

What is the difference between embeddings and traditional keyword search?

Explain the concept discussed in this lecture.

What are the main points covered in this topic?
```

The assistant retrieves relevant transcript content before generating the response.

---

## 🔍 Why RAG?

Traditional LLMs have a limitation: they may not know the specific information contained in a private or newly created course.

RAG solves this by providing relevant external context to the LLM.

### Without RAG

```text
Question
   ↓
LLM
   ↓
Answer
```

The answer depends primarily on the model's pretrained knowledge.

### With RAG

```text
Question
   ↓
Semantic Search
   ↓
Course Knowledge
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
```

This makes the system more suitable for **domain-specific question answering**.

---

## 📈 Advantages

* Reduces the need to manually search through long videos.
* Provides answers based on course-specific information.
* Can work with newly added course material.
* Separates knowledge retrieval from answer generation.
* Makes course content easier to interact with.
* Demonstrates a practical implementation of a modern RAG pipeline.

---

## ⚠️ Limitations

* Answer quality depends on the quality of the transcript.
* Incorrect or incomplete transcription can affect retrieval.
* Poor chunking can cause relevant information to be missed.
* Semantic retrieval may occasionally return less relevant chunks.
* LLM responses can still contain inaccuracies.
* Large course collections may require a more scalable vector database.

---

## 🔮 Future Improvements

* [ ] Add timestamps for retrieved transcript sections.
* [ ] Add support for multiple courses.
* [ ] Improve chunking using semantic boundaries.
* [ ] Add a dedicated vector database such as FAISS, Chroma, or Pinecone.
* [ ] Implement hybrid keyword + semantic search.
* [ ] Add reranking for retrieved chunks.
* [ ] Add conversation memory.
* [ ] Display source transcript sections with every answer.
* [ ] Add document/video upload functionality.
* [ ] Improve evaluation using retrieval and generation metrics.

---

## 🎯 Learning Outcomes

This project demonstrates practical knowledge of:

* Retrieval-Augmented Generation (RAG)
* Large Language Models
* Prompt Engineering
* Semantic Search
* Vector Embeddings
* Speech-to-Text
* Text Chunking
* Information Retrieval
* Gemini API integration
* Streamlit application development
* AI-powered question answering

---

## 👨‍💻 Author

**Tejas Shinde**

GitHub: [@Tejas1707-bit](https://github.com/Tejas1707-bit)

---

## ⭐ Acknowledgement

This project was developed as a practical implementation of concepts related to **Retrieval-Augmented Generation, semantic search, embeddings, and LLM-based applications**.

If you find this project useful, consider giving the repository a ⭐.

---

## 📜 License

This project is available for educational and learning purposes.
