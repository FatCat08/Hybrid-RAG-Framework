├── Chain/  
│   ├── Chain.py                  # Implements the Chain of LangChain for chaining various sub-tasks or questions in the multi-hop QA process.
│   ├── tmp.db                    # Temporary SQLite database storing intermediate results during the chain execution.
├── Config/  
│   ├── Config.py                 # Contains configuration settings, such as API keys, file paths, and retrieval parameters.
├── data/  
│   ├── link.txt                  # Link for TKGQA dataset
├── Dataset Construction/          # Directory for scripts and data related to constructing the TKGQA dataset.
├── Evaluation/  
│   ├── Evaluation.py             # Evaluation script that measures performance using metrics like Exact Match (EM), F1 score, etc.
├── KnowledgeGraph/  
│   ├── kb_to_neo4j.py            # Script to convert the knowledge graph data into Neo4j format for graph-based queries.
│   ├── kb_to_vector.py           # Script for vectorizing the knowledge graph data for use in vector-based retrieval methods.
│   ├── KG_api.py                 # API for interacting with the knowledge graph using various retrieval methods like Neo4j or vector search.
├── outputs/  
│   ├── df_neo4j/                 # Results and logs from using DataFrame for table retrieval and Neo4j for KG retrieval.
│   │   ├── bad_case_2.txt        # Text file logging cases where the retrieval or answer generation failed.
│   │   ├── results_1.txt         # Logs or results for one of the experimental runs using the `df_neo4j` model combination.
│   │   ├── results_3.txt         # Additional results or logs for another experimental run.
│   ├── df_vs/                    # Logs and results from experiments using DataFrame for table retrieval and vector store for KG retrieval.
│   │   ├── bad_case_2.txt
│   │   ├── results_1.txt
│   │   ├── results_2.txt
│   │   ├── results_3.txt
│   ├── sqlite_neo4j/             # Results and logs for the SQLite and Neo4j combination in hybrid retrieval.
│   │   ├── bad_case_2.txt
│   │   ├── results_1.txt
│   │   ├── results_2.txt
│   │   ├── results_3.txt
│   ├── sqlite_vs/                # Logs and results for SQLite and vector store combination.
│   │   ├── bad_case_2.txt
│   │   ├── results_1.txt
│   │   ├── results_2.txt
│   │   ├── results_3.txt
│   ├── vs_neo4j/                 # Logs and results for vector store and Neo4j combination.
│   │   ├── bad_case_2.txt
│   │   ├── results_1.txt
│   │   ├── results_2.txt
│   │   ├── results_3.txt
│   ├── vs_vs/                    # Logs and results for vector store-based retrieval for both table and KG data.
│   │   ├── bad_case_2.txt
│   │   ├── results_1.txt
│   │   ├── results_2.txt
│   │   ├── results_3.txt
├── Parser/  
│   ├── Parser.py                 # Parsing for the output of LLMs
├── Print_structure.py            # Script that generates and prints the repository or dataset structure, likely used for logging.
├── prompts/  
│   ├── prompts.json              # JSON file containing pre-defined prompts for interacting with the LLM and guiding the QA process.
├── repository_structure.md       # Documentation of the project’s repository structure in markdown format.
├── Table/  
│   ├── Table_api.py              # API for querying and retrieving data from tables, either using SQL or DataFrame methods.
│   ├── table_to_sqlite.py        # Script for loading table data into an SQLite database for structured retrieval.
│   ├── table_to_vector.py        # Script for converting tabular data into vector format for use in vector-based retrieval methods.
│   ├── tmp.db                    # Temporary SQLite database used during table retrieval experiments.
