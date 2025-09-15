GraphGNN — KG → KGE → GNN → QA (pipeline)
Pequeña documentación para el script graphgnn.py (prototipo de investigación).

Resumen
Este script implementa un pipeline en español que:

Construye un Knowledge Graph (nodos: lemas y noun‑chunks; aristas: co‑ocurrencia ventana=5 + dependencias).
Calcula PPMI para ponderar aristas.
Entrena KGE (CompGCN por defecto; fallback/handling de triples inversos y fallback a RotatE si hace falta).
Entrena un GNN (GAT) usando las incrustaciones KGE como features.
Responde consultas en español priorizando respuestas nominales concisas y ofrece comparación opcional con una respuesta de LLM local (ollama).
Genera visualizaciones (KG, embeddings, comparaciones) y tiene un modo --quick para pruebas rápidas.

Requisitos (alto nivel)

Python 3.8+ (se recomienda 3.10+).
Paquetes principales: spaCy (modelo es_core_news_sm), numpy, networkx, matplotlib, scikit-learn, umap‑learn (opcional), torch, torch‑geometric, pykeen, rapidfuzz (opcional), tiktoken (opcional), ollama (opcional).
Nota: torch-geometric puede requerir instalación específica según CPU/GPU; siga la guía oficial de PyG.

Instalación mínima sugerida (PowerShell):

py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1
py -3 -m pip install -U pip
py -3 -m pip install numpy networkx matplotlib scikit-learn umap-learn spacy torch pykeen rapidfuzz tiktoken
py -3 -m spacy download es_core_news_sm


Ejecución rápida
Desde PowerShell:

py -3 "C:\Users\<tu_usuario>\Desktop\graphgnn.py" --quick --no-llm

--quick: reduce dimensiones y epochs para pruebas.
--no-llm: desactiva llamadas a ollama.

El script pedirá una consulta en español y generará imágenes PNG y una respuesta concisa.

Diseño y decisiones clave

Nodos: lemas + noun‑chunks (multi‑palabra).
Aristas: co‑ocurrencia (ventana 5) y dependencias sintácticas; PPMI como peso.
KGE: CompGCN por defecto (se reintenta con create_inverse_triples=True si PyKEEN lo requiere). Per‑query KGE también usa CompGCN con fallback a RotatE.
GNN: GAT de dos capas (PyG), entrenado con negative sampling para link‑prediction.
Entity linking: token overlap + rapidfuzz/difflib fallback (no TF‑IDF por requisito).
Razonamiento por caminos: paths simples hasta longitud L=3, puntuación basada en número/inversa de longitud.
Preferencia en respuesta: se priorizan NOUN/PROPN y noun‑chunks que aparecen en los documentos originales.

Salidas generadas

Figuras: global_kg.png, kge_embeddings.png, gnn_embeddings.png, query_graph.png, comparison_*.
Si --run-llm: ficheros llm_response_<safe_name>.txt.
(Opcional) puede añadirse volcado de embeddings con np.save para persistencia.



Reproducibilidad
Use un entorno virtual y guarde dependencias:

py -3 -m pip freeze > deps.txt

Capture información del sistema (Windows PowerShell ejemplos):

py -3 --version
Get-CimInstance Win32_ComputerSystem | select TotalPhysicalMemory
wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors
# si aplica:
nvidia-smi


Fije semillas en NumPy y PyTorch para reducir aleatoriedad; establezca random_state en UMAP/TSNE cuando proceda.

Problemas comunes y soluciones

PyKEEN/CompGCN error por falta de triples inversos: el script reconstruye TriplesFactory con inversos y reintenta; si persiste, usar fallback RotatE.
spaCy: si falta el modelo, ejecutar py -3 -m spacy download es_core_news_sm.
PyG: siga las instrucciones específicas para la versión de PyTorch y su plataforma.
Ollama: si no lo tiene, use --no-llm.


Extensiones sugeridas
Persistir embeddings en un vector DB para producción.
Convertir el script a microservicio (FastAPI) que devuelva únicamente la respuesta corta.
Añadir pruebas unitarias para extracción de triples, creación de TriplesFactory y entrenamiento rápido en modo --quick.
