| S.N | Topics | Link |
| :--: | ---- | ---- |
| 1 | Improving RAG with Knowledge Graphs | [link](./Improving-RAG-with-Knowledge-Graphs.md) |

*Resources*

| S.N | Title | Link | Resource Type |
| :--: | ---- | :--: | ---- |
| 1 | Introduction to Knowledge Graph | [link](https://neo4j.com/developer-blog/knowledge-graph-rag-application/) | Blogs |
| 2 | Constructing Knowledge Graph with Unstructured Data | [link](https://neo4j.com/developer-blog/construct-knowledge-graphs-unstructured-text/) | Blogs |
| 3 | RAG with Knowledge Graph Code | [link](https://github.com/tomasonjo/blogs/blob/master/llm/devops_rag.ipynb) | GitHub |
| 4 | Hybrid RAG with Knowledge Graph | [link](https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/) | Blogs |
| 5 | GRetriever Paper | [link]([https://arxiv.org/pdf/2402.07630.pdf](https://arxiv.org/pdf/2402.07630.pdf)) | Blogs |
| 6 | Microsoft GraphRAG | [link]([https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)) | Blogs |

# Common Graph RAG Architectures
| S.N | Architectures | Overview | References |
| ---- | ---- | ---- | ---- |
| 1 | Knowledge Graph with Semantic Clustering | **Indexing Process:**<br><br>- **Text Unit Segmentation:** The input corpus is divided into smaller text chunks (e.g., paragraphs or sentences) to extract detailed information.<br>- **Entity, Relationship, and Claims Extraction:** LLMs identify entities, relationships, and key claims from each text unit to build an initial knowledge graph.<br>- **Hierarchical Clustering:** The Leiden algorithm performs hierarchical clustering to organize the knowledge graph into communities, with densely connected nodes grouped together for in-depth analysis.<br>- **Community Summary Generation:** Summaries are created for each community, including key entities, relationships, and claims, providing an overview of the data for future queries.<br><br>**Querying:**<br><br>- **Global Search:** Used for reasoning about broad questions across the entire corpus, utilizing community summaries.<br>- **Local Search:** Focuses on specific entities, expanding to their neighbors and related concepts for more targeted analysis. | - [Microsoft Research - Graph RAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)<br>- [Zilliz's blog](https://medium.com/@zilliz_learn/graphrag-explained-enhancing-rag-with-knowledge-graphs-3312065f99e1)<br>- [CohesionForce's blog](https://blog.cohesionforce.com/2024/05/15/graphrag-enhancing-retrieval-augmented-generation-with-llm-generated-knowledge-graphs/) |
| 2 | Knowledge Graph and Vector Database Integraion |  | - [How to Implement Graph RAG using Knowledge Graphs and Vector Databases](https://towardsdatascience.com/how-to-implement-graph-rag-using-knowledge-graphs-and-vector-databases-60bb69a22759) |
