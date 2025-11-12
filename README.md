# VIT ChatBot based on LLM + RAG

A lightweight Retrieval-Augmented Generation (RAG) based chatbot designed to answer academic queries for VIT University students. The system uses a local vector database and Gemini API to retrieve accurate program-related information such as course details, credits, prerequisites, labs, curriculum structure, and more.

---

## Features

* RAG-powered contextual responses
* Fast API backend using **uvicorn**
* Simple and clean **HTML frontend**
* Easy setup with downloadable vector database
* Modular codebase (chat pipeline, backend server, and front-end separated)

---

## Project Structure

```
├── chat.py                # RAG pipeline
├── main.py                # FastAPI backend
├── index.html             # Frontend client UI
├── finetune.py            # Optional finetuning scripts (if used)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## Step 1 — Download the CSV files

Download the required folder from Google Drive:

 **Database Link:** [https://drive.google.com/drive/folders/1KkL47zz-CQGh7czbaoLSjvWu6cqzrbUJ](https://drive.google.com/drive/folders/1KkL47zz-CQGh7czbaoLSjvWu6cqzrbUJ)

After downloading:

* Extract the contents
* Place **ALL** files directly inside the **project main folder** (same folder as `main.py`)

 Final folder should look like:

```
project/
    main.py
    chat.py
    index.html
    <all CSV files here>
```

---

## Step 2 — Create a `.env` File

Create a `.env` file in the project root directory.

Add the following line:

```
GOOGLE_API_KEY=your_api_key_here
```

Make sure you have a valid Google Gemini API Key.

---

## Step 3 — Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

If using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

---

## Step 4 — Start the Backend Server

Run the backend with:

```bash
uvicorn main:app --reload
```

This will start the server at:

```
http://127.0.0.1:8000
```

---

## Step 5 — Use the Frontend

Simply open the **index.html** file in your browser.

No additional server required — it will automatically connect to the FastAPI backend.

---

## You're Ready!

You now have a fully functional RAG-powered VIT Academic Assistant running locally.

