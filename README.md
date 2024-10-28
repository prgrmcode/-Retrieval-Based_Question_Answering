# Retrieval-Based Question Answering with LangChain and Hugging Face

This repository contains a Jupyter notebook that demonstrates how to build a retrieval-based question-answering system using LangChain and Hugging Face. The notebook guides you through the process of setting up the environment, loading and processing documents, generating embeddings, and querying the system to retrieve relevant documents.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Setup](#setup)
  - [Parsing Documents](#parsing-documents)
  - [Generating Embeddings](#generating-embeddings)
  - [Retrieving Relevant Documents](#retrieving-relevant-documents)
  - [Using LangChain](#using-langchain)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository**:

   ```sh
   git clone https://github.com/prgrmcode/retrieval-based-qa-llm.git
   cd retrieval-based-qa-llm
   ```

2. **Create a virtual environment**:

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Setup

1. **Install required packages**:

   ```python
   %pip install -Uqqq rich tiktoken wandb langchain unstructured tabulate pdf2image chromadb
   %pip install --upgrade transformers
   %pip install -U "huggingface_hub[cli]"
   %pip install -U langchain-community
   %pip install -U langchain_huggingface
   %pip install sentence-transformers
   ```

2. **Login to Hugging Face**:

   ```python
   %huggingface-cli login
   ```

3. **Configure Hugging Face API token**:

   ```python
   import os
   from getpass import getpass

   if os.getenv("HUGGINGFACE_API_TOKEN") is None:
       os.environ["HUGGINGFACE_API_TOKEN"] = getpass("Paste your Hugging Face API token from: https://huggingface.co/settings/tokens\n")

    assert os.getenv("HUGGINGFACE_API_TOKEN", "").startswith("hf_"), "This doesn't look like a valid Hugging Face API token"
    print("Hugging Face API token is configured")
   ```

4. **Configure W&B tracing**:
   ```python
   os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
   os.environ["WANDB_PROJECT"] = "llmapps"
   ```

### Parsing Documents

1. **Load documents from the specified directory**:

   ```python
   import time
   from langchain_community.document_loaders import DirectoryLoader, TextLoader

   def find_md_files(directory):
       start_time = time.time()
       loader = DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
       documents = loader.load()
       end_time = time.time()
       print(f"Time taken to load documents: {end_time - start_time:.2f} seconds")
       return documents

   documents = find_md_files(directory="docs_sample/")
   print(f"Number of documents loaded: {len(documents)}")
   ```

2. **Count tokens in each document**:

   ```python
   def count_tokens(documents):
       token_counts = [len(tokenizer.encode(document.page_content)) for document in documents]
       return token_counts

   token_counts = count_tokens(documents)
   print(f"Token counts: {token_counts}")
   ```

3. **Split documents into sections**:

   ```python
   from langchain.text_splitter import MarkdownTextSplitter

   md_text_splitter = MarkdownTextSplitter(chunk_size=1000)
   document_sections = md_text_splitter.split_documents(documents)
   print(f"Number of document sections: {len(document_sections)}")
   print(f"Max tokens in a section: {max(count_tokens(document_sections))}")
   ```

### Generating Embeddings

1. **Initialize the tokenizer and model**:

   ```python
   import torch
   import transformers

   model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
   tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
   model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
   ```

2. **Generate embeddings using HuggingFaceEmbeddings**:

   ```python
   from langchain.vectorstores import Chroma
   from langchain_huggingface import HuggingFaceEmbeddings

   model_name = "sentence-transformers/all-mpnet-base-v2"
   model_kwargs = {"device": "cuda"}
   encode_kwargs = {"normalize_embeddings": False}
   embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

   db = Chroma.from_documents(document_sections, embeddings)
   ```

### Retrieving Relevant Documents

1. **Create a retriever from the database**:

   ```python
   retriever = db.as_retriever(search_kwargs=dict(k=3))
   ```

2. **Run a query to retrieve relevant documents**:

   ```python
   query = "How can I share my W&B report with my team members in a public W&B project?"
   docs = retriever.invoke(query)

   for doc in docs:
       print(doc.metadata["source"])
   ```

### Using LangChain

1. **Create a RetrievalQA chain**:

   ```python
   from langchain.chains import RetrievalQA
   from langchain_huggingface import HuggingFacePipeline
   from transformers import pipeline
   from tqdm import tqdm

   pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=70)
   llm = HuggingFacePipeline(pipeline=pipe)

   qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
   ```

2. **Run the query using the RetrievalQA chain**:

   ```python
   with tqdm(total=1, desc="Running RetrievalQA") as pbar:
       result = qa.run(query)
       pbar.update(1)

   display(Markdown(result))
   ```

## Examples

The `examples.txt` contains example inputs and outputs for various tasks. These examples can help you understand the expected behavior of the models and scripts.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
