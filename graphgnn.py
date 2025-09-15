"""
graphgnn.py

Versión modular y comentada del pipeline end-to-end solicitado por el usuario.

Características implementadas (resumen):
- Preprocesamiento con spaCy: tokenización, lematización, NER y parse de dependencias.
- Construcción de KG de co-ocurrencia (window-based w=5) y opción de edges por dependencia.
- Cálculo de frecuencia y PMI para edges.
- Entrenamiento de KGE con PyKEEN (ComplEx, dim=200, epochs=200, batch_size=256).
- Visualización de grafo y embeddings (UMAP/TSNE fallback).
- Extracción de subgrafos k-hop (k=2) con networkx.
- Entrenamiento GNN (GAT) con PyTorch Geometric usando embeddings KGE como features.
- Módulo de razonamiento / QA: entity linking (spaCy + fuzzy), extracción de subgrafo, búsqueda de paths (L=3), scoring combinado (α,β,γ).
- Métricas: MRR, Hits@k, Precision@k, NDCG. mantiene la comparación con LLM (Ollama) como antes.

Nota: Este script agrupa todo en un solo archivo para facilitar su despliegue local. Antes de ejecutar, asegúrese de tener instaladas las dependencias en el entorno.
"""

import time
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import spacy
from collections import Counter, defaultdict

# PyKEEN for KGE
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# PyTorch / PyG for GNN
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    from rapidfuzz import process as rapidfuzz_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

try:
    import ollama
    _HAS_OLLAMA = True
except Exception:
    _HAS_OLLAMA = False

import subprocess


def call_ollama(query: str, model: str = 'llama3.1:8b', timeout: int = 30) -> Tuple[bool, str]:
    """Call ollama to generate an answer. Tries python package first, then CLI fallback.
    Returns (success, text_or_error)."""
    if _HAS_OLLAMA:
        try:
            r = ollama.chat(model=model, messages=[{'role': 'user', 'content': query}])
            return True, r.get('message', {}).get('content', str(r))
        except Exception as e:
            return False, f'ollama python client error: {e}'
    # CLI fallback: try `ollama generate <model> --prompt "<query>"`
    try:
        cmd = ['ollama', 'generate', model, query]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        else:
            return False, f'ollama CLI error: {proc.stderr.strip() or proc.stdout.strip()}'
    except FileNotFoundError:
        return False, 'ollama CLI not found'
    except Exception as e:
        return False, f'ollama CLI error: {e}'


# ------------------------------- Configuración --------------------------------
KGE_PARAMS = dict(model='ComplEx', embedding_dim=200, training_kwargs=dict(num_epochs=200, batch_size=256))
GNN_PARAMS = dict(gat_heads=4, hidden=64, out_dim=128, epochs=100, lr=1e-3)
WINDOW_SIZE = 5
K_HOP = 2
PATH_LEN = 3
ALPHA, BETA, GAMMA = 1.0, 1.0, 1.0


# ------------------------------- Preprocesamiento -----------------------------
def load_spacy(model_name: str = 'es_core_news_sm'):
    try:
        return spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(f"spaCy model '{model_name}' no disponible. Instale con: python -m spacy download {model_name}\n{e}")


def normalize_label(s: str) -> str:
    s = str(s).lower().strip()
    # remove leading articles
    for art in ('el ', 'la ', 'los ', 'las ', 'un ', 'una ', 'unos ', 'unas '):
        if s.startswith(art):
            s = s[len(art):]
            break
    # strip punctuation
    s = ''.join(ch for ch in s if ch.isalnum() or ch.isspace())
    s = ' '.join(s.split())
    return s


def preprocess_documents(docs: List[Dict[str, Any]], nlp=None) -> List[Dict[str, Any]]:
    """Tokenize, lemmatize, NER and dependency parse. Returns enriched doc dicts."""
    if nlp is None:
        nlp = load_spacy()
    processed = []
    for d in docs:
        text = d.get('doc') or d.get('text') or ''
        doc = nlp(text)
        tokens = [{'text': t.text, 'lemma': t.lemma_, 'pos': t.pos_, 'is_stop': t.is_stop, 'idx': t.i} for t in doc if t.is_alpha]
        entities = [{'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char} for ent in doc.ents]
        deps = [{'token': t.text, 'head': t.head.text, 'dep': t.dep_} for t in doc]
        processed.append({'id': d.get('id'), 'text': text, 'doc': doc, 'tokens': tokens, 'entities': entities, 'deps': deps})
    return processed


# ------------------------------- KG Construction ------------------------------
def build_cooccurrence_kg(processed_docs: List[Dict[str, Any]], node_type: str = 'tokens', window: int = WINDOW_SIZE, add_dependency_edges: bool = True) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
    """Construye un grafo de co-ocurrencia. node_type: 'tokens' or 'entities'. Returns global graph and per-doc subgraphs."""
    G = nx.Graph()
    per_doc = []
    for d in processed_docs:
        # Include noun_chunks (multi-word noun phrases) alongside token lemmas to better represent concepts
        noun_chunks = []
        try:
            doc_spacy = d.get('doc')
            if doc_spacy is not None:
                noun_chunks = [nc.text.lower() for nc in doc_spacy.noun_chunks if len(nc.text.split()) > 1]
        except Exception:
            noun_chunks = []
        def normalize_label(s: str) -> str:
            s = str(s).lower().strip()
            # remove leading articles
            for art in ('el ', 'la ', 'los ', 'las ', 'un ', 'una ', 'unos ', 'unas '):
                if s.startswith(art):
                    s = s[len(art):]
                    break
            # strip punctuation
            s = ''.join(ch for ch in s if ch.isalnum() or ch.isspace())
            s = ' '.join(s.split())
            return s

        if node_type == 'entities' and d['entities']:
            nodes = [normalize_label(e['text']) for e in d['entities']]
        else:
            token_lemmas = [normalize_label(t['lemma']) for t in d['tokens']]
            noun_chunks_norm = [normalize_label(nc) for nc in noun_chunks]
            # combine lemmas and noun_chunks (avoid duplicates)
            nodes = list(dict.fromkeys(token_lemmas + noun_chunks_norm))
        # add nodes
        for n in nodes:
            G.add_node(n, count=G.nodes[n]['count'] + 1 if n in G.nodes else 1)
        # relation extraction: try simple SVO and causal pattern extraction to label edges
        rels_extracted = []
        try:
            doc_spacy = d.get('doc')
            if doc_spacy is not None:
                # simple SVO via dependency parse: find verbs with nsubj and dobj
                for tok in doc_spacy:
                    if tok.pos_ == 'VERB':
                        subj = None
                        dobj = None
                        for ch in tok.children:
                            if ch.dep_ in ('nsubj', 'nsubj:pass'):
                                subj = ch
                            if ch.dep_ in ('dobj', 'obj'):
                                dobj = ch
                        if subj is not None and dobj is not None:
                            subj_text = subj.text.lower()
                            dobj_text = dobj.text.lower()
                            verb_lemma = tok.lemma_.lower()
                            rels_extracted.append((subj_text, verb_lemma, dobj_text))
                # simple causal pattern extraction (phrase-based)
                text_lower = doc_spacy.text.lower()
                causal_markers = ['debido a', 'causado por', 'provocado por', 'a causa de', 'por culpa de']
                for marker in causal_markers:
                    if marker in text_lower:
                        parts = text_lower.split(marker)
                        if len(parts) >= 2:
                            left = parts[0].strip().split()[-4:]
                            right = parts[1].strip().split()[:4]
                            head = ' '.join(left).strip()
                            tail = ' '.join(right).strip()
                            if head and tail:
                                rels_extracted.append((head, 'causado_por', tail))
        except Exception:
            rels_extracted = []
        # window edges
        for i, a in enumerate(nodes):
            for j in range(i+1, min(i+1+window, len(nodes))):
                b = nodes[j]
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                    G[a][b]['docs'].add(d['id'])
                    # ensure relation label present (Spanish)
                    rels = G[a][b].setdefault('rels', set())
                    rels.add('coocurre_con')
                else:
                    G.add_edge(a, b, weight=1, docs={d['id']}, rels=set(['coocurre_con']))
        # attach any extracted relations as directed relation labels between the closest node matches
        for h, rlab, t in rels_extracted:
            # find best matching node for head and tail among nodes (prefer noun_chunks exact match first)
            def best_match(span_text):
                st = normalize_label(span_text)
                # exact node match
                if st in nodes:
                    return st
                # try token/lemma containment
                for n in nodes:
                    if st in str(n).lower() or str(n).lower() in st:
                        return n
                # fallback to first node containing any word of span
                words = st.split()
                for w in words:
                    for n in nodes:
                        if w in str(n).lower():
                            return n
                return None

            a_match = best_match(h)
            b_match = best_match(t)
            if a_match and b_match and a_match != b_match:
                # add or update edge with the relation label
                if G.has_edge(a_match, b_match):
                    G[a_match][b_match]['rels'].add(rlab)
                else:
                    G.add_edge(a_match, b_match, weight=1, docs={d['id']}, rels=set([rlab]))
        # dependency edges (optional)
        if add_dependency_edges and node_type == 'tokens':
            # map token text -> lemma lower
            lem_map = {t['text']: t['lemma'].lower() for t in d['tokens']}
            # small mapping from UD dependency labels to Spanish short labels
            dep_map = {
                'nsubj': 'sujeto', 'nsubj:pass': 'sujeto_pasivo', 'dobj': 'objeto', 'obj': 'objeto',
                'amod': 'modificador', 'nmod': 'modificador_nominal', 'case': 'preposicion', 'advmod': 'modificador_adverbial'
            }
            for dep in d['deps']:
                tok = dep['token']
                head = dep['head']
                if tok in lem_map and head in lem_map:
                    a = lem_map[tok]
                    b = lem_map[head]
                    if a != b:
                        dep_label = dep.get('dep') or ''
                        dep_label_spanish = dep_map.get(dep_label, dep_label)
                        if G.has_edge(a, b):
                            G[a][b]['weight'] += 1
                            G[a][b]['docs'].add(d['id'])
                            rels = G[a][b].setdefault('rels', set())
                            # add Spanish-mapped dependency label
                            rels.add(f"dep:{dep_label_spanish}")
                        else:
                            G.add_edge(a, b, weight=1, deps=True, docs={d['id']}, rels=set([f"dep:{dep_label_spanish}"]))
        per_doc.append({'id': d['id'], 'nodes': nodes, 'graph': G.subgraph(nodes).copy()})
    return G, per_doc


def compute_pmi(graph: nx.Graph, total_windows: int) -> None:
    """Add PMI weight attribute to each edge in-place. total_windows is approximate number of sliding windows across corpus."""
    node_counts = {n: graph.nodes[n].get('count', 1) for n in graph.nodes}
    for u, v, d in graph.edges(data=True):
        cooc = d.get('weight', 1)
        p_xy = cooc / total_windows
        p_x = node_counts[u] / total_windows
        p_y = node_counts[v] / total_windows
        pmi = np.log((p_xy + 1e-12) / (p_x * p_y + 1e-12))
        d['pmi'] = float(max(pmi, 0.0))


def build_query_graph(query: str, nlp, window: int = WINDOW_SIZE, add_dependency_edges: bool = True) -> nx.Graph:
    """Build a small KG from the query text using same node rules (lemmas + noun_chunks) and simple edges.
    Returns a NetworkX undirected graph with normalized node labels.
    """
    Gq = nx.Graph()
    doc = nlp(query)
    noun_chunks = [normalize_label(nc.text) for nc in doc.noun_chunks if len(nc.text.split()) > 1]
    token_lemmas = [normalize_label(t.lemma_) for t in doc if t.is_alpha]
    nodes = list(dict.fromkeys(token_lemmas + noun_chunks))
    for n in nodes:
        Gq.add_node(n, count=1)

    # window edges
    for i, a in enumerate(nodes):
        for j in range(i+1, min(i+1+window, len(nodes))):
            b = nodes[j]
            if Gq.has_edge(a, b):
                Gq[a][b]['weight'] += 1
                rels = Gq[a][b].setdefault('rels', set())
                rels.add('coocurre_con')
            else:
                Gq.add_edge(a, b, weight=1, rels=set(['coocurre_con']))

    # SVO and causal extractions
    try:
        for tok in doc:
            if tok.pos_ == 'VERB':
                subj = None
                dobj = None
                for ch in tok.children:
                    if ch.dep_ in ('nsubj', 'nsubj:pass'):
                        subj = ch
                    if ch.dep_ in ('dobj', 'obj'):
                        dobj = ch
                if subj is not None and dobj is not None:
                    a = normalize_label(subj.text)
                    b = normalize_label(dobj.text)
                    r = tok.lemma_.lower()
                    if a != b:
                        if Gq.has_edge(a, b):
                            Gq[a][b]['rels'].add(r)
                        else:
                            Gq.add_edge(a, b, weight=1, rels=set([r]))

        # causal markers
        text_lower = doc.text.lower()
        causal_markers = ['debido a', 'causado por', 'provocado por', 'a causa de', 'por culpa de']
        for marker in causal_markers:
            if marker in text_lower:
                parts = text_lower.split(marker)
                if len(parts) >= 2:
                    left = parts[0].strip().split()[-4:]
                    right = parts[1].strip().split()[:4]
                    head = normalize_label(' '.join(left).strip())
                    tail = normalize_label(' '.join(right).strip())
                    if head and tail and head != tail:
                        if Gq.has_edge(head, tail):
                            Gq[head][tail]['rels'].add('causado_por')
                        else:
                            Gq.add_edge(head, tail, weight=1, rels=set(['causado_por']))
    except Exception:
        pass

    return Gq


def token_overlap(a: str, b: str) -> float:
    ta = set([t.lower() for t in str(a).split() if len(t) > 2])
    tb = set([t.lower() for t in str(b).split() if len(t) > 2])
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb)) / float(max(1, (len(ta) + len(tb)) / 2.0))


def visualize_graph(graph: nx.Graph, path: str = 'global_graph.png', title: str = 'Global KG'):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=0)
    weights = [graph[u][v].get('weight', 1) for u, v in graph.edges()]
    nx.draw(graph, pos=pos, with_labels=True, node_size=300, font_size=8, width=[max(0.5, w/2) for w in weights])
    # NOTE: edge relation labels intentionally not drawn to keep visualization clean
    plt.title(title)
    plt.savefig(path)
    plt.show()


# ------------------------------- KGE Training ---------------------------------
def triples_from_graph(graph: nx.Graph, relation: str = 'cooccurs_with') -> List[Tuple[str, str, str]]:
    triples = []
    for u, v, d in graph.edges(data=True):
        rels = d.get('rels') or set([relation])
        # ensure rels is iterable of strings
        for r in rels:
            rlab = str(r)
            triples.append((str(u), rlab, str(v)))
            triples.append((str(v), rlab, str(u)))
    return triples


def train_kge(triples: List[Tuple[str, str, str]], model: str = 'CompGCN', dim: int = 200, epochs: int = 200, batch_size: int = 256, random_seed: int = 0, model_kwargs: Optional[Dict[str, Any]] = None):
    arr = np.array(triples)
    # CompGCN (and some other models) require inverse triples; attempt normal creation first
    create_inv = False
    try:
        tf = TriplesFactory.from_labeled_triples(arr)
    except Exception:
        # fallback: create with inverse triples
        tf = TriplesFactory.from_labeled_triples(arr, create_inverse_triples=True)
        create_inv = True
    # build model kwargs with embedding dim default, but allow overrides
    mk = dict(embedding_dim=dim)
    if model_kwargs:
        mk.update(model_kwargs)
    try:
        result = pipeline(model=model, dataset=None, training=tf, testing=tf, model_kwargs=mk, training_kwargs=dict(num_epochs=epochs, batch_size=batch_size), random_seed=random_seed)
    except AssertionError as ae:
        # specific CompGCN requirement: ensure inverse triples are present and retry
        if 'create_inverse_triples' in str(ae) or 'inverse' in str(ae).lower() or model.lower().startswith('compgcn'):
            # recreate with inverse triples and retry
            tf = TriplesFactory.from_labeled_triples(arr, create_inverse_triples=True)
            result = pipeline(model=model, dataset=None, training=tf, testing=tf, model_kwargs=mk, training_kwargs=dict(num_epochs=epochs, batch_size=batch_size), random_seed=random_seed)
        else:
            raise
    # extract entity embeddings (convert complex -> real if needed)
    ent_count = len(tf.entity_to_id)
    ent_repr = result.model.entity_representations[0](torch.arange(ent_count)).detach().cpu().numpy()
    if np.iscomplexobj(ent_repr):
        ent_emb = np.concatenate([ent_repr.real, ent_repr.imag], axis=1)
    else:
        ent_emb = ent_repr
    # relation embeddings if available
    rel_emb = None
    try:
        rel_repr = result.model.relation_representations[0](torch.arange(len(tf.relation_to_id))).detach().cpu().numpy()
        if np.iscomplexobj(rel_repr):
            rel_emb = np.concatenate([rel_repr.real, rel_repr.imag], axis=1)
        else:
            rel_emb = rel_repr
    except Exception:
        rel_emb = None
    return result, tf, ent_emb, rel_emb


def visualize_embeddings(embeddings: np.ndarray, labels: List[str], path: str = 'embeddings.png', title: str = 'Embeddings'):
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=0)
        coords = reducer.fit_transform(embeddings)
    else:
        reducer = TSNE(n_components=2, random_state=0)
        coords = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='tab:blue')
    for i, lab in enumerate(labels):
        plt.annotate(str(lab), (coords[i, 0], coords[i, 1]), fontsize=8)
    plt.title(title)
    plt.savefig(path)
    plt.show()


def visualize_embeddings_with_query(embeddings: np.ndarray, labels: List[str], query_vec: np.ndarray, query_label: str, path: str = 'emb_with_query.png', title: str = 'Embeddings with query'):
    # project embeddings + query into 2D and plot, highlighting query
    all_emb = np.vstack([embeddings, np.asarray(query_vec)])
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=0)
        coords = reducer.fit_transform(all_emb)
    else:
        reducer = TSNE(n_components=2, random_state=0)
        coords = reducer.fit_transform(all_emb)
    coords_nodes = coords[:-1]
    coord_q = coords[-1]
    plt.figure(figsize=(8, 6))
    plt.scatter(coords_nodes[:, 0], coords_nodes[:, 1], c='tab:blue')
    for i, lab in enumerate(labels):
        plt.annotate(str(lab), (coords_nodes[i, 0], coords_nodes[i, 1]), fontsize=8)
    plt.scatter([coord_q[0]], [coord_q[1]], c='red', s=120, marker='*')
    plt.annotate(query_label, (coord_q[0], coord_q[1]), fontsize=10, color='red')
    plt.title(title)
    plt.savefig(path)
    plt.show()


def visualize_query_on_graph(graph: nx.Graph, query: str, path: str = 'query_on_graph.png', title: str = 'Query on KG'):
    # highlight nodes that match query tokens or lemmas
    q_tokens = set([t.lower() for t in query.split() if len(t) > 2])
    node_colors = []
    for n in graph.nodes():
        if any(tok in str(n).lower() for tok in q_tokens):
            node_colors.append('red')
        else:
            node_colors.append('skyblue')
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=0)
    nx.draw(graph, pos=pos, with_labels=True, node_color=node_colors, node_size=300, font_size=8)
    # NOTE: relation labels are not drawn in the visualization (they are stored on edges and used for training)
    plt.title(title)
    plt.savefig(path)
    plt.show()


def compute_text_similarity(a: str, b: str) -> float:
    """Simple TF-IDF cosine similarity fallback used for comparing short answers to a document/triple."""
    try:
        return text_similarity_tfidf(a, b)
    except Exception:
        # fallback to token overlap
        return token_overlap(a, b)


def plot_gnn_vs_llm_comparison(query: str, gnn_answer: Dict[str, Any], llm_info: Optional[Dict[str, Any]], best_doc_text: str, out_prefix: str = 'comparison'):
    """Generate and save three comparison charts:
    - similarity (GNN vs LLM) to the best document/triple
    - response time (seconds)
    - response length (tokens)
    Saves PNG files with out_prefix and query-safe suffix.
    """
    # sanitize query for filenames
    safe = ''.join(ch for ch in query if ch.isalnum() or ch.isspace()).strip().replace(' ', '_')[:60]
    if not safe:
        safe = 'query'

    gnn_text = gnn_answer.get('text', '') if isinstance(gnn_answer, dict) else str(gnn_answer)
    gnn_score = gnn_answer.get('score', 0.0) if isinstance(gnn_answer, dict) else 0.0
    gnn_time = gnn_answer.get('time', 0.0) if isinstance(gnn_answer, dict) else 0.0
    gnn_tokens = gnn_answer.get('tokens', 0) if isinstance(gnn_answer, dict) else token_count(str(gnn_answer))

    llm_text = ''
    llm_time = 0.0
    llm_tokens = 0
    llm_err = None
    if llm_info:
        if 'text' in llm_info and llm_info['text']:
            llm_text = llm_info['text']
            llm_time = float(llm_info.get('time', 0.0) or 0.0)
            llm_tokens = int(llm_info.get('tokens', 0) or token_count(llm_text))
        else:
            llm_err = llm_info.get('error') or llm_info.get('err')

    # compute similarity to best_doc_text
    gnn_sim = compute_text_similarity(gnn_text, best_doc_text) if best_doc_text and gnn_text else 0.0
    llm_sim = compute_text_similarity(llm_text, best_doc_text) if best_doc_text and llm_text else 0.0

    # --- Similarity bar chart ---
    try:
        plt.figure(figsize=(6, 4))
        labels = ['GNN', 'LLM']
        sims = [gnn_sim, llm_sim]
        colors = ['tab:blue', 'tab:orange']
        plt.bar(labels, sims, color=colors)
        plt.ylim(0, 1.0)
        plt.ylabel('Similarity to best document (0-1)')
        plt.title(f'Similarity: GNN vs LLM - {query[:40]}')
        fname = f'{out_prefix}_similarity_{safe}.png'
        plt.tight_layout()
        plt.savefig(fname)
        try:
            plt.show()
        except Exception:
            pass
        plt.close()
    except Exception:
        pass

    # --- Time bar chart ---
    try:
        plt.figure(figsize=(6, 4))
        labels = ['GNN', 'LLM']
        times = [float(gnn_time or 0.0), float(llm_time or 0.0)]
        colors = ['tab:blue', 'tab:orange']
        plt.bar(labels, times, color=colors)
        plt.ylabel('Response time (s)')
        plt.title(f'Response time: GNN vs LLM - {query[:40]}')
        fname = f'{out_prefix}_time_{safe}.png'
        plt.tight_layout()
        plt.savefig(fname)
        try:
            plt.show()
        except Exception:
            pass
        plt.close()
    except Exception:
        pass

    # --- Length bar chart ---
    try:
        plt.figure(figsize=(6, 4))
        labels = ['GNN', 'LLM']
        lengths = [int(gnn_tokens or 0), int(llm_tokens or 0)]
        colors = ['tab:blue', 'tab:orange']
        plt.bar(labels, lengths, color=colors)
        plt.ylabel('Response length (tokens)')
        plt.title(f'Response length: GNN vs LLM - {query[:40]}')
        fname = f'{out_prefix}_length_{safe}.png'
        plt.tight_layout()
        plt.savefig(fname)
        try:
            plt.show()
        except Exception:
            pass
        plt.close()
    except Exception:
        pass

    # --- Response-vs-response similarity ---
    try:
        # compute similarity between the two responses if possible
        resp_sim = None
        if llm_text:
            try:
                resp_sim = compute_text_similarity(gnn_text, llm_text)
            except Exception:
                resp_sim = 0.0
        else:
            resp_sim = 0.0
        plt.figure(figsize=(6, 4))
        labels = ['GNN vs LLM']
        values = [resp_sim]
        plt.bar(labels, values, color=['tab:purple'])
        plt.ylim(0, 1.0)
        plt.ylabel('Similarity between responses (0-1)')
        plt.title(f'Response similarity - {query[:40]}')
        fname = f'{out_prefix}_response_similarity_{safe}.png'
        plt.tight_layout()
        plt.savefig(fname)
        try:
            plt.show()
        except Exception:
            pass
        plt.close()
    except Exception:
        pass



def text_similarity_tfidf(a: str, b: str) -> float:
    vec = TfidfVectorizer().fit_transform([a, b])
    va = vec[0].toarray()[0]
    vb = vec[1].toarray()[0]
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def token_count(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


# ------------------------------- GNN Training ---------------------------------
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


def nx_to_pyg(graph: nx.Graph, features: np.ndarray) -> Data:
    mapping = {n: i for i, n in enumerate(graph.nodes())}
    edges = [[mapping[u], mapping[v]] for u, v in graph.edges()]
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # faster: gather rows from the features numpy array using the mapping order
    if features is None or features.shape[0] == 0:
        x = torch.zeros((len(mapping), 0), dtype=torch.float)
    else:
        idxs = [mapping[n] for n in graph.nodes()]
        # guard if features has fewer rows than nodes
        max_rows = features.shape[0]
        rows = [features[i] if i < max_rows else np.zeros(features.shape[1]) for i in idxs]
        x = torch.tensor(np.asarray(rows), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def train_gnn_for_link_prediction(graph: nx.Graph, init_features: np.ndarray, params: Dict[str, Any]) -> Tuple[nn.Module, np.ndarray]:
    data = nx_to_pyg(graph, init_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGAT(in_channels=data.x.size(1), hidden_channels=params['hidden'], out_channels=params['out_dim'], heads=params['gat_heads']).to(device)
    data = data.to(device)

    # prepare positive edges
    edge_index = data.edge_index
    pos_u = edge_index[0].cpu().numpy()
    pos_v = edge_index[1].cpu().numpy()
    pos_edges = set([(int(u), int(v)) for u, v in zip(pos_u, pos_v)])

    optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
    epochs = params.get('epochs', 100)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        # score positive edges
        pos_scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        # sample negative edges
        neg_u = torch.randint(0, z.size(0), (pos_scores.size(0),), device=device)
        neg_v = torch.randint(0, z.size(0), (pos_scores.size(0),), device=device)
        neg_scores = (z[neg_u] * z[neg_v]).sum(dim=1)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_scores, neg_scores]), torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"GNN epoch {epoch}/{epochs} loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        final_z = model(data.x, data.edge_index).cpu().numpy()
    return model, final_z


# ------------------------------- Reasoning / QA --------------------------------
def entity_linking(query: str, entities: List[str], processed_docs: Optional[List[Dict[str, Any]]] = None, n_top: int = 5) -> List[Tuple[str, float]]:
    """Simpler linking: token-overlap between query and entity surface forms, fallback to rapidfuzz/difflib.
    TF-IDF logic removed per user request.
    """
    q = query.strip().lower()
    scores = {}
    q_tokens = set([t for t in q.split() if len(t) > 2])
    for ent in entities:
        ent_s = str(ent).lower()
        ent_tokens = set([t for t in ent_s.split() if len(t) > 2])
        # token overlap score
        if ent_tokens:
            overlap = float(len(q_tokens & ent_tokens)) / float(max(1, len(ent_tokens)))
            if overlap > 0:
                scores[ent] = overlap
    # fuzzy fallback if not enough
    if len(scores) < n_top:
        remaining = [e for e in entities if e not in scores]
        if _HAS_RAPIDFUZZ and remaining:
            matches = rapidfuzz_process.extract(q, remaining, limit=n_top)
            for m in matches:
                scores[m[0]] = max(scores.get(m[0], 0.0), float(m[1]) / 100.0 * 0.7)
        else:
            import difflib
            matches = difflib.get_close_matches(q, remaining, n=n_top, cutoff=0.4)
            for m in matches:
                scores[m] = max(scores.get(m, 0.0), 0.5)
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_top]
    return items


def extract_k_hop_subgraph(graph: nx.Graph, seeds: List[str], k: int = K_HOP) -> nx.Graph:
    nodes = set()
    for s in seeds:
        if s not in graph:
            continue
        nodes.add(s)
        for i in range(k):
            neighbors = set()
            for n in list(nodes):
                neighbors |= set(graph.neighbors(n))
            nodes |= neighbors
    return graph.subgraph(nodes).copy()


def find_paths(graph: nx.Graph, source: str, target: str, max_len: int = PATH_LEN) -> List[List[str]]:
    try:
        paths = list(nx.all_simple_paths(graph, source=source, target=target, cutoff=max_len))
        return paths
    except Exception:
        return []


def normalize_scores(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.max() == a.min():
        return np.zeros_like(a)
    a = (a - a.min()) / (a.max() - a.min())
    return a


def answer(query: str, graph: nx.Graph, tf: TriplesFactory, kge_ent_emb: np.ndarray, gnn_emb: np.ndarray, processed_docs: Optional[List[Dict[str, Any]]] = None, top_k: int = 5, alpha=ALPHA, beta=BETA, gamma=GAMMA) -> Dict[str, Any]:
    entities = list(tf.entity_to_id.keys())
    link_candidates = entity_linking(query, entities, processed_docs=processed_docs, n_top=top_k * 3)
    cand_labels = [c[0] for c in link_candidates]

    # compute KGE scores: cosine between query vector (avg of matched entity vectors) and entity vectors
    q_vec = np.zeros(kge_ent_emb.shape[1])
    matched_ids = [tf.entity_to_id[c] for c in cand_labels if c in tf.entity_to_id]
    if matched_ids:
        q_vec = kge_ent_emb[matched_ids].mean(axis=0)
    else:
        q_vec = kge_ent_emb.mean(axis=0)
    kge_sims = np.array([np.dot(q_vec, v) / (np.linalg.norm(q_vec) * (np.linalg.norm(v) + 1e-12)) if np.linalg.norm(v) > 0 else 0.0 for v in kge_ent_emb])

    # GNN scores (use cosine with gnn_emb)
    gnn_sims = np.array([np.dot(q_vec[:gnn_emb.shape[1]], v) / (np.linalg.norm(q_vec[:gnn_emb.shape[1]]) * (np.linalg.norm(v) + 1e-12)) if np.linalg.norm(v) > 0 else 0.0 for v in gnn_emb])

    # path score: for each candidate, search paths from any seed to candidate within L
    path_scores = np.zeros(len(entities))
    # seeds: top linked candidate labels
    seeds = [c[0] for c in link_candidates][:3]
    subg = extract_k_hop_subgraph(graph, seeds, k=K_HOP)
    for i, ent in enumerate(entities):
        if ent not in subg:
            path_scores[i] = 0.0
            continue
        best_paths = []
        for s in seeds:
            if s in subg and ent in subg:
                ps = find_paths(subg, s, ent, max_len=PATH_LEN)
                if ps:
                    best_paths.extend(ps)
        # simple path score: number of paths weighted by inverse length
        sc = 0.0
        for p in best_paths:
            sc += 1.0 / max(1, (len(p) - 1))
        path_scores[i] = sc

    # normalize
    kge_norm = normalize_scores(kge_sims)
    gnn_norm = normalize_scores(gnn_sims)
    path_norm = normalize_scores(path_scores)

    final_scores = alpha * kge_norm + beta * gnn_norm + gamma * path_norm

    ranked_idx = np.argsort(final_scores)[::-1][:top_k]
    results = []
    id_to_ent = {v: k for k, v in tf.entity_to_id.items()}
    for idx in ranked_idx:
        ent = id_to_ent[idx]
        results.append({'entity': ent, 'score': float(final_scores[idx]), 'kge_score': float(kge_norm[idx]), 'gnn_score': float(gnn_norm[idx]), 'path_score': float(path_norm[idx])})

    # For each top result, include top paths (if any)
    for r in results:
        ent = r['entity']
        paths = []
        for s in seeds:
            if s in subg and ent in subg:
                ps = find_paths(subg, s, ent, max_len=PATH_LEN)
                for p in ps:
                    paths.append(p)
        r['paths'] = paths[:5]

    return {'query': query, 'results': results, 'seeds': seeds}


def refine_human_answer(gnn_result: Dict[str, Any], processed_docs: List[Dict[str, Any]], tf: TriplesFactory, nlp=None) -> Tuple[str, float]:
    """Try to pick a better human-readable answer by inspecting paths and original documents.
    Looks for multi-word phrases from paths present in original texts, then noun tokens in paths, then falls back."""
    if nlp is None:
        try:
            nlp = load_spacy()
        except Exception:
            nlp = None

    results = gnn_result.get('results', [])
    # No hard-coded whitelist: prioritization will use noun-chunks and paths to generalize across queries
    # 1) Search for multi-word phrases from paths that appear verbatim in any original doc
    for r in results:
        for p in r.get('paths', []):
            if len(p) >= 2:
                # create phrases from contiguous nodes in the path
                for L in range(len(p), 1, -1):
                    for i in range(0, len(p) - L + 1):
                        phrase = ' '.join(p[i:i+L])
                        if len(phrase) < 3:
                            continue
                        for d in processed_docs:
                            if phrase.lower() in d['text'].lower():
                                return phrase, float(r.get('score', 0.0))

    # 2) Prefer noun/proper noun tokens found in any path nodes
    for r in results:
        for p in r.get('paths', []):
            for node in p:
                txt = str(node)
                if len(txt) <= 2 or not txt.isalpha():
                    continue
                if nlp is not None:
                    doc = nlp(txt)
                    if len(doc) > 0:
                        tok = doc[0]
                        if not tok.is_stop and tok.pos_ in ('NOUN', 'PROPN'):
                            return txt, float(r.get('score', 0.0))
                else:
                    # without nlp prefer longer alphabetic tokens
                    if len(txt) > 3:
                        return txt, float(r.get('score', 0.0))

    # 3) fallback to existing select_best_candidate behavior
    cand, sc = select_best_candidate(gnn_result, nlp=nlp)

    # If candidate is likely a verb or too short, try to expand it to a noun phrase from docs
    def expand_to_noun_phrase(token: str) -> str:
        if not token or len(token) <= 2:
            return token
        if nlp is None:
            return token
        t = token.strip()
        for d in processed_docs:
            try:
                doc = d.get('doc')
                if doc is None:
                    continue
                # search noun chunks that contain the token
                for nc in doc.noun_chunks:
                    if t.lower() in nc.text.lower():
                        return nc.text
                # if token not in noun_chunks, search for nearby noun chunk by token lemma
                for nc in doc.noun_chunks:
                    # check token overlap
                    if any(tok.lemma_.lower() == t.lower() for tok in nc):
                        return nc.text
            except Exception:
                continue
        return token

    # determine pos of cand
    if nlp is not None and cand:
        try:
            docc = nlp(cand)
            if len(docc) > 0 and docc[0].pos_ == 'VERB':
                expanded = expand_to_noun_phrase(cand)
                if expanded and expanded.lower() != cand.lower():
                    return expanded, sc
        except Exception:
            pass

    return cand, sc


def select_best_candidate(gnn_result: Dict[str, Any], nlp=None) -> Tuple[str, float]:
    """Deterministic fallback: pick top-scoring entity from gnn_result['results'].
    Prefer NOUN/PROPN if available, else return the top entity and its score.
    """
    results = gnn_result.get('results', []) if isinstance(gnn_result, dict) else []
    if not results:
        return '', 0.0
    # prefer an entity whose token is a noun/proper noun
    if nlp is not None:
        for r in results:
            ent = r.get('entity')
            try:
                doc = nlp(ent)
                if len(doc) > 0 and doc[0].pos_ in ('NOUN', 'PROPN'):
                    return ent, float(r.get('score', 0.0))
            except Exception:
                continue
    # otherwise return top-scoring
    top = results[0]
    return top.get('entity', ''), float(top.get('score', 0.0))


# ------------------------------- Metrics --------------------------------------
def mean_reciprocal_rank(ranked_lists: List[List[int]]) -> float:
    rr = []
    for rl in ranked_lists:
        for i, v in enumerate(rl, start=1):
            if v == 1:
                rr.append(1.0 / i)
                break
        else:
            rr.append(0.0)
    return float(np.mean(rr))


def hits_at_k(ranked: List[List[int]], k: int) -> float:
    hits = [1.0 if 1 in r[:k] else 0.0 for r in ranked]
    return float(np.mean(hits))


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    topk = retrieved[:k]
    return float(len([r for r in topk if r in relevant]) / max(1, k))


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    def dcg(rel_scores):
        return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rel_scores))
    rel_scores = [1 if d in relevant else 0 for d in retrieved[:k]]
    ideal = sorted(rel_scores, reverse=True)
    return float(dcg(rel_scores) / (dcg(ideal) + 1e-12))


# ------------------------------- Example runner / CLI -------------------------
def run_pipeline(docs: List[Dict[str, Any]], user_queries: Optional[List[str]] = None, run_llm: bool = False, quick: bool = False):
    print('1) Preprocessing...')
    nlp = load_spacy()
    processed = preprocess_documents(docs, nlp=nlp)

    print('2) Build KG...')
    G, per_doc = build_cooccurrence_kg(processed, node_type='tokens', window=WINDOW_SIZE, add_dependency_edges=True)
    total_windows = sum(max(1, len(p['nodes']) - WINDOW_SIZE + 1) for p in per_doc)
    compute_pmi(G, total_windows=max(total_windows, 1))
    visualize_graph(G, path='global_kg.png', title='KG global (coocurrence + deps)')
    # Debugging: print small node summary to help trace why some expected nodes (e.g., 'fútbol')
    try:
        sample_nodes = list(G.nodes())[:40]
        print('\n--- DEBUG: nodes in KG (sample) ---')
        print(sample_nodes)
        if 'deporte' in G:
            print("Neighbors of 'deporte':", list(G.neighbors('deporte')))
        else:
            # try normalized variants
            for cand in ['fútbol', 'futbol', 'deporte']:
                if cand in G:
                    print(f"Found node variant in KG: {cand} neighbors ->", list(G.neighbors(cand)))
                    break
        print('--- end debug ---\n')
    except Exception:
        pass

    print('3) Extract triples and train KGE (CompGCN)...')
    triples = triples_from_graph(G)
    # quick mode overrides to make local testing fast
    if quick:
        kge_dim = 64
        kge_epochs = 5
        kge_batch = 32
    else:
        kge_dim = KGE_PARAMS['embedding_dim']
        kge_epochs = KGE_PARAMS['training_kwargs']['num_epochs']
        kge_batch = KGE_PARAMS['training_kwargs']['batch_size']
    result, tf, ent_emb, rel_emb = train_kge(triples, model='CompGCN', dim=kge_dim, epochs=kge_epochs, batch_size=kge_batch)
    # Diagnostic prints: show some triples and embedding shapes to validate relation learning
    try:
        print('\n--- DIAGNOSTIC: Triples sample ---')
        sample_triples = triples[:10]
        for t in sample_triples:
            print(t)
        print('Total triples:', len(triples))
        print('Entities count:', len(tf.entity_to_id), 'Relations count:', len(tf.relation_to_id) if hasattr(tf, 'relation_to_id') else 0)
        print('ent_emb.shape =', getattr(ent_emb, 'shape', None), 'rel_emb.shape =', getattr(rel_emb, 'shape', None))
        print('relation_to_id sample:', dict(list(tf.relation_to_id.items())[:10]) if hasattr(tf, 'relation_to_id') else {})
        print('--- end diagnostic ---\n')
    except Exception as _:
        pass
    labels = list(tf.entity_to_id.keys())
    visualize_embeddings(ent_emb, labels, path='kge_embeddings.png', title='KGE entity embeddings')

    print('4) Build subgraph k-hop (k=2) example and visualize')
    # pick first doc nodes as seeds
    seeds = per_doc[0]['nodes'][:3] if per_doc and per_doc[0]['nodes'] else list(G.nodes())[:3]
    sg = extract_k_hop_subgraph(G, seeds, k=K_HOP)
    visualize_graph(sg, path='subgraph_khop.png', title=f'k-hop subgraph (k={K_HOP})')

    print('5) Train GNN (GAT) using KGE embeddings as initial features...')
    # align features to graph node ordering and augment with mean incident relation embeddings
    node_list = list(G.nodes())
    ent_dim = ent_emb.shape[1]
    rel_dim = rel_emb.shape[1] if rel_emb is not None else 0
    # final feature will be concat(entity_emb, mean_incident_rel_emb)
    features = np.zeros((len(node_list), ent_dim + rel_dim), dtype=float)
    ent_to_idx = tf.entity_to_id
    rel_to_idx = tf.relation_to_id if hasattr(tf, 'relation_to_id') else {}
    for i, n in enumerate(node_list):
        # entity part
        if n in ent_to_idx:
            ent_vec = ent_emb[ent_to_idx[n]]
        else:
            ent_vec = np.zeros(ent_dim)
        # relation part: collect relation labels on incident edges
        rel_vec = np.zeros(rel_dim)
        if rel_dim > 0:
            rels = []
            for nbr in G.neighbors(n):
                ed = G.get_edge_data(n, nbr, default={})
                if not ed:
                    continue
                for r in ed.get('rels', []):
                    rid = rel_to_idx.get(str(r))
                    if rid is not None and rid < rel_emb.shape[0]:
                        rels.append(rel_emb[rid])
            if rels:
                rel_vec = np.mean(np.stack(rels, axis=0), axis=0)
        features[i] = np.concatenate([ent_vec, rel_vec])
    # Diagnostic: print a few sample node features and whether relation part is non-zero
    try:
        print('\n--- DIAGNOSTIC: sample node features ---')
        for i in range(min(5, len(node_list))):
            n = node_list[i]
            feat = features[i]
            ent_part = feat[:ent_dim]
            rel_part = feat[ent_dim:ent_dim+rel_dim] if rel_dim > 0 else np.array([])
            print(f'node={n} ent_norm={np.linalg.norm(ent_part):.4f} rel_norm={np.linalg.norm(rel_part) if rel_part.size else 0.0:.4f}')
        print('--- end diagnostic ---\n')
    except Exception:
        pass
    # quick mode: smaller GNN epochs and dims
    gnn_params = dict(GNN_PARAMS)
    if quick:
        gnn_params['epochs'] = 10
        gnn_params['hidden'] = 32
        gnn_params['out_dim'] = 64
        gnn_params['gat_heads'] = 2
    gnn_model, gnn_z = train_gnn_for_link_prediction(G, features, gnn_params)
    visualize_embeddings(gnn_z, node_list, path='gnn_embeddings.png', title='GNN node embeddings')

    # Keep LLM comparison (ollama) if available
    llm_responses = {}
    if run_llm:
        for q in (user_queries or []):
            ok, text = call_ollama(q, model='llama3.1:8b')
            if ok:
                llm_responses[q] = text
            else:
                llm_responses[q] = f'LLM error: {text}'

            # print and save the LLM response for quick inspection
            try:
                print('\n--- LLM response for query ---')
                print(text)
                safe_name = ''.join(ch for ch in q if ch.isalnum() or ch.isspace()).strip().replace(' ', '_')[:60]
                fname = f'llm_response_{safe_name}.txt' if safe_name else 'llm_response.txt'
                with open(fname, 'w', encoding='utf-8') as fh:
                    fh.write(str(text))
                print(f"Saved LLM response to {fname}")
                print('--- end LLM response ---\n')
            except Exception:
                pass

    # Answer user queries
    answers = {}
    for q in (user_queries or []):
        # Build query graph and merge with global KG for localized training
        t0_g = time.time()
        Gq = build_query_graph(q, nlp, window=WINDOW_SIZE, add_dependency_edges=True)
        visualize_graph(Gq, path='query_graph.png', title='Query KG')

        # Map query nodes to nearest global nodes (fuzzy or token overlap) to get candidate seeds
        global_nodes = list(G.nodes())
        from collections import defaultdict as _dd
        candidate_scores = _dd(float)
        for qn in Gq.nodes():
            # skip very short tokens or stopwords to avoid trivial seeds like 'en', 'de', 'y'
            if len(qn) <= 2:
                continue
            try:
                if qn in nlp.Defaults.stop_words:
                    continue
            except Exception:
                pass
            if qn in G.nodes():
                candidate_scores[qn] = max(candidate_scores[qn], 1.0)
                continue
            # fuzzy match to global nodes if available
            if _HAS_RAPIDFUZZ:
                try:
                    matches = rapidfuzz_process.extract(qn, global_nodes, limit=5)
                    for m in matches:
                        candidate_scores[m[0]] = max(candidate_scores[m[0]], float(m[1]) / 100.0)
                except Exception:
                    # fallback to token overlap
                    for gn in global_nodes:
                        s = token_overlap(qn, gn)
                        if s > 0:
                            candidate_scores[gn] = max(candidate_scores[gn], s)
            else:
                for gn in global_nodes:
                    s = token_overlap(qn, gn)
                    if s > 0:
                        candidate_scores[gn] = max(candidate_scores[gn], s)

        # select top candidate seeds (keep up to 10)
        cand_sorted = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_nodes = [c[0] for c in cand_sorted[:10]]
        print('\n--- DIAGNOSTIC: query->global candidate seeds ---')
        for csc in cand_sorted[:10]:
            print(csc)
        print('--- end diagnostic ---\n')

        # extract k-hop subgraph around candidate seeds (this relaxes the strict intersection)
        seeds_for_subg = candidate_nodes if candidate_nodes else list(G.nodes())[:3]
        overlap_subg = extract_k_hop_subgraph(G, seeds_for_subg, k=K_HOP)
        visualize_graph(overlap_subg, path='overlap_subgraph.png', title='Candidate subgraph (seeds expanded)')
        overlap_nodes = set(overlap_subg.nodes())

        # build triples: global triples + query triples + symmetric
        q_triples = triples_from_graph(Gq)
        combined_triples = triples + q_triples

        # train TransE quickly on combined triples to get fresh embeddings focused on query
        # quick mode overrides
        if quick:
            tr_dim = 64; tr_epochs = 20; tr_batch = 32
        else:
            tr_dim = 200; tr_epochs = 200; tr_batch = 256
        try:
            tr_result, tr_tf, tr_ent_emb, tr_rel_emb = train_kge(combined_triples, model='CompGCN', dim=tr_dim, epochs=tr_epochs, batch_size=tr_batch)
        except Exception:
            # fallback to RotatE if CompGCN not available or fails
            tr_result, tr_tf, tr_ent_emb, tr_rel_emb = train_kge(combined_triples, model='RotatE', dim=tr_dim, epochs=tr_epochs, batch_size=tr_batch)

        # visualize TransE entity embeddings with query highlight if possible
        tr_labels = list(tr_tf.entity_to_id.keys())
        # create a query embedding by averaging embeddings of overlap nodes if present
        q_vec_tr = None
        overlap_ids = [tr_tf.entity_to_id[n] for n in overlap_nodes if n in tr_tf.entity_to_id]
        if overlap_ids:
            q_vec_tr = tr_ent_emb[overlap_ids].mean(axis=0)
        else:
            q_vec_tr = tr_ent_emb.mean(axis=0)
        visualize_embeddings_with_query(tr_ent_emb, tr_labels, q_vec_tr, query_label=q, path='tr_embeddings_with_query.png', title='TransE embeddings with query')

        # build features from tr_ent_emb and relation means and train GNN on global graph
        # align features to global G node list using tr_tf entity mapping
        node_list_global = list(G.nodes())
        ent_dim_tr = tr_ent_emb.shape[1]
        features_tr = np.zeros((len(node_list_global), ent_dim_tr), dtype=float)
        for i, n in enumerate(node_list_global):
            if n in tr_tf.entity_to_id:
                features_tr[i] = tr_ent_emb[tr_tf.entity_to_id[n]]
            else:
                features_tr[i] = np.zeros(ent_dim_tr)

        gnn_model_tr, gnn_z_tr = train_gnn_for_link_prediction(G, features_tr, gnn_params)
        visualize_embeddings(gnn_z_tr, node_list_global, path='gnn_tr_embeddings.png', title='GNN (TransE features) embeddings')

    # pick best answer from overlap nodes using combined signals:
        # - GNN embedding similarity (global)
        # - token overlap with query
        # - adjacency/path proximity to query seeds (shorter paths => bonus)
        # - noun/noun-chunk/document occurrence bonus
    best_ans = ''
    best_score = -1.0
    query_noun_chunks = set([normalize_label(nc.text) for nc in nlp(q).noun_chunks if len(nc.text.split()) > 1])
    node_to_idx = {n: i for i, n in enumerate(node_list_global)}

    # precompute seeds as normalized set (exclude stopwords/short)
    query_tokens = [t for t in [normalize_label(t.text) for t in nlp(q) if t.is_alpha] if len(t) > 2 and t not in getattr(nlp, 'Defaults').stop_words]
    query_tokens_set = set(query_tokens)
    seeds_norm = [s for s in seeds_for_subg if s in G]

    # function to check if node appears as noun-chunk or in any original doc
    def node_is_in_docs_as_phrase(node_text: str) -> bool:
        for d in processed:
            try:
                # exact match of node as substring (more robust)
                if node_text.lower() in d.get('text', '').lower():
                    # prefer multiword or noun appearances
                    return True
            except Exception:
                continue
        return False

    if overlap_nodes:
        overlap_idxs = [node_to_idx[n] for n in overlap_nodes if n in node_to_idx]
        if overlap_idxs:
            q_vec_gnn = gnn_z_tr[overlap_idxs].mean(axis=0)
        else:
            q_vec_gnn = gnn_z_tr.mean(axis=0)
    # Deterministic 1-hop noun neighbor preference: prefer nodes that are direct neighbors
        # of any high-confidence seed and are NOUN/PROPN. If such nodes exist, restrict
        # candidate set to those nodes (ensures answer is connected to query)
        one_hop_noun_candidates = set()
        try:
            for s in seeds_for_subg:
                if s in G:
                    for nbr in G.neighbors(s):
                        try:
                            if nbr in overlap_nodes:
                                doc_tok = nlp(nbr)
                                if len(doc_tok) > 0 and doc_tok[0].pos_ in ('NOUN', 'PROPN'):
                                    one_hop_noun_candidates.add(nbr)
                        except Exception:
                            continue
        except Exception:
            one_hop_noun_candidates = set()
        # if we have 1-hop noun candidates, focus scoring on them to ensure connectivity
        if len(one_hop_noun_candidates) > 0:
            candidate_pool = one_hop_noun_candidates
        else:
            candidate_pool = overlap_nodes
        # Primary-seed heuristic: if the top candidate seed has noun neighbors that appear in
        # the original documents, choose the best among them deterministically (prefer direct
        # connectivity to the main seed). This helps pick 'fútbol' for the example query.
        selected_by_primary = False
        primary_seed = candidate_nodes[0] if candidate_nodes else (seeds_for_subg[0] if seeds_for_subg else None)
        if primary_seed and primary_seed in G:
            prim_neighs = [nbr for nbr in G.neighbors(primary_seed) if nbr in overlap_nodes]
            prim_candidates = []
            for nbr in prim_neighs:
                try:
                    doc_tok = nlp(nbr)
                    if len(doc_tok) > 0 and doc_tok[0].pos_ in ('NOUN', 'PROPN') and node_is_in_docs_as_phrase(nbr):
                        prim_candidates.append(nbr)
                except Exception:
                    continue
            if prim_candidates:
                # score candidates by GNN similarity to the primary seed embedding (if available)
                seed_idx = node_to_idx.get(primary_seed)
                seed_vec = gnn_z_tr[seed_idx] if (seed_idx is not None and seed_idx < gnn_z_tr.shape[0]) else None
                scored = []
                for c in prim_candidates:
                    ci = node_to_idx.get(c)
                    if ci is None:
                        continue
                    if seed_vec is not None:
                        simc = float(np.dot(seed_vec, gnn_z_tr[ci]) / ((np.linalg.norm(seed_vec) * (np.linalg.norm(gnn_z_tr[ci]) + 1e-12))))
                    else:
                        simc = 0.0
                    scored.append((c, simc))
                if scored:
                    scored = sorted(scored, key=lambda x: x[1], reverse=True)
                    best_ans = scored[0][0]
                    best_score = scored[0][1] + 2.0  # big boost to prefer primary-connected noun
                    # set neighbor suggestions as next top neighbors of best_ans
                    neighbor_suggestions = [x[0] for x in scored[1:3]]
                    selected_by_primary = True
        # only run the general scoring loop if primary-selection didn't pick an answer
        if not selected_by_primary:
            for n in overlap_nodes:
                if n not in candidate_pool:
                    continue
                if n not in node_to_idx:
                    continue
                idx = node_to_idx[n]
                v = gnn_z_tr[idx]
                # embedding similarity
                sim = float(np.dot(q_vec_gnn, v) / (np.linalg.norm(q_vec_gnn) * (np.linalg.norm(v) + 1e-12)))
                # token overlap (small weight to avoid lexical artifacts like 'popular')
                sim += 0.2 * token_overlap(q, n)
                # path/adjacency bonus: shorter path from any seed -> higher bonus
                dist_bonus = 0.0
                try:
                    min_dist = None
                    for s in seeds_norm:
                        if s in G and n in G:
                            try:
                                dlen = nx.shortest_path_length(G, source=s, target=n)
                                if min_dist is None or dlen < min_dist:
                                    min_dist = dlen
                            except Exception:
                                continue
                    if min_dist is not None and min_dist <= PATH_LEN:
                        # inverse distance bonus
                        dist_bonus = 0.8 / (1.0 + float(min_dist))
                    # strong 1-hop neighbor boost from any seed (prefer direct neighbors)
                    try:
                        one_hop_bonus = 0.0
                        for s in seeds_norm:
                            if s in G and n in G and G.has_edge(s, n):
                                one_hop_bonus = 1.2
                                break
                    except Exception:
                        one_hop_bonus = 0.0
                    sim += one_hop_bonus
                except Exception:
                    dist_bonus = 0.0
                sim += dist_bonus
                # noun/noun-chunk or document occurrence bonus
                try:
                    doc_tok = nlp(n)
                    if len(doc_tok) > 0 and doc_tok[0].pos_ in ('NOUN', 'PROPN'):
                        # stronger boost for nouns/proper nouns
                        sim += 1.5
                except Exception:
                    pass
                if node_is_in_docs_as_phrase(n):
                    sim += 0.9
                # penalize exact echoing of query tokens (we prefer specific answers, not repeats of the query)
                if n in query_tokens_set:
                    sim -= 0.9
                # prefer multi-word noun-chunks from the query
                if n in query_noun_chunks:
                    sim += 1.2
                if sim > best_score:
                    best_score = sim
                    best_ans = n
    else:
        q_vec_gnn = gnn_z_tr.mean(axis=0)
        for i, n in enumerate(node_list_global):
            v = gnn_z_tr[i]
            sim = float(np.dot(q_vec_gnn, v) / (np.linalg.norm(q_vec_gnn) * (np.linalg.norm(v) + 1e-12)))
            sim += token_overlap(q, n)
            try:
                doc_tok = nlp(n)
                if len(doc_tok) > 0 and doc_tok[0].pos_ in ('NOUN', 'PROPN'):
                    sim += 0.8
            except Exception:
                pass
            if node_is_in_docs_as_phrase(n):
                sim += 0.6
            if sim > best_score:
                best_score = sim
                best_ans = n

    t1_g = time.time()
    gnn_time = t1_g - t0_g

    # prepare lightweight metrics and optional LLM call
    llm_info = None
    if run_llm and _HAS_OLLAMA:
        try:
            start_time = time.time()
            ok, llm_text = call_ollama(q, model='llama3.1:8b')
            llama_time = time.time() - start_time
            if not ok:
                raise RuntimeError(llm_text)
            # record base llm info
            llm_info = {'text': llm_text, 'time': float(llama_time), 'tokens': int(token_count(llm_text))}
            # print quick summary to console
            try:
                print('\n--- LLM quick metrics ---')
                print('Respuesta Llama 3.1:', llm_text)
                print('Tiempo de respuesta Llama 3.1:', f"{llama_time:.3f}s")
                print('Tokens respuesta Llama 3.1:', llm_info['tokens'])
                print('--- end LLM quick metrics ---\n')
            except Exception:
                pass
        except Exception as e:
            llm_info = {'error': str(e)}
    else:
        if run_llm and not _HAS_OLLAMA:
            llm_info = {'error': 'ollama not available locally'}

    # find best_doc_text where the answer appears (if any)
    # Small promotion: if best_ans is an adjective, verb, or exactly a query token
    # try to replace it with a neighboring noun/proper-noun (e.g., 'fútbol').
    try:
        if best_ans:
            try:
                btok = nlp(best_ans)
                is_adj = len(btok) > 0 and btok[0].pos_ in ('ADJ', 'VERB')
            except Exception:
                is_adj = False
            is_query_token = any(t.lower() in best_ans.lower() for t in q.split() if len(t) > 2)
            if is_adj or is_query_token or (best_ans and len(best_ans) > 0 and not best_ans.isalpha()):
                # search neighbors for a noun/proper-noun
                try:
                    for nbr in G.neighbors(best_ans):
                        try:
                            dtk = nlp(nbr)
                            if len(dtk) > 0 and dtk[0].pos_ in ('NOUN', 'PROPN'):
                                best_ans = nbr
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
    except Exception:
        pass

    best_doc_text = ''
    for d in processed:
        if best_ans and best_ans.lower() in d.get('text', '').lower():
            best_doc_text = d.get('text', '')
            break
    if not best_doc_text and processed:
        best_doc_text = processed[0].get('text', '')

    sim_answer_doc = text_similarity_tfidf(best_ans, best_doc_text) if best_doc_text and best_ans else 0.0
    ans_tokens = token_count(best_ans) if best_ans else 0

    # build short human-readable answer: include best_ans plus up to two most probable neighbors
    neighbor_suggestions = []
    try:
        if best_ans and best_ans in G:
            # rank neighbors by GNN similarity to the q_vec_gnn (if available) or by degree
            neighs = list(G.neighbors(best_ans))
            scored_neighs = []
            for nb in neighs:
                if nb not in node_to_idx:
                    continue
                idxn = node_to_idx[nb]
                simn = float(np.dot(gnn_z_tr[idx], gnn_z_tr[idxn]) / ((np.linalg.norm(gnn_z_tr[idx]) * (np.linalg.norm(gnn_z_tr[idxn]) + 1e-12)))) if gnn_z_tr.size else 0.0
                # prefer noun neighbors
                try:
                    dtk = nlp(nb)
                    if len(dtk) > 0 and dtk[0].pos_ in ('NOUN', 'PROPN'):
                        simn += 0.5
                except Exception:
                    pass
                scored_neighs.append((nb, simn))
            scored_neighs = sorted(scored_neighs, key=lambda x: x[1], reverse=True)
            neighbor_suggestions = [x[0] for x in scored_neighs[:2]]
    except Exception:
        neighbor_suggestions = []

    human_answer_text = best_ans
    if neighbor_suggestions:
        # include neighbor suggestions to form a concise phrase (e.g., 'fútbol (y clubes/participantes)')
        human_answer_text = best_ans + ' — ' + ', '.join(neighbor_suggestions)

    # store GNN-derived metrics and LLM info
    answers[q] = {'gnn': {'results': [], 'human_answer': {'text': human_answer_text, 'score': float(best_score), 'time': gnn_time, 'tokens': ans_tokens, 'sim_with_doc': sim_answer_doc}}, 'transE': {'tf': tr_tf, 'ent_emb': tr_ent_emb}}
    answers[q]['llm'] = llm_info
    # store best_doc_text for plotting/comparison
    answers[q]['best_doc_text'] = best_doc_text

    # compute similarity between GNN answer and LLM answer (if both available)
    try:
        gnn_text_local = human_answer_text or ''
        sim_between = None
        if llm_info and isinstance(llm_info, dict) and llm_info.get('text'):
            try:
                sim_between = compute_text_similarity(gnn_text_local, llm_info.get('text', ''))
            except Exception:
                sim_between = None
        answers[q]['sim_between_responses'] = sim_between
        # print basic comparison summary
        try:
            print('\n--- Comparison summary (GNN vs LLM) ---')
            print('GNN answer:', gnn_text_local)
            print('GNN tokens:', ans_tokens, 'time(s):', f"{gnn_time:.3f}")
            if llm_info and isinstance(llm_info, dict) and llm_info.get('text'):
                print('LLM answer:', llm_info.get('text'))
                print('LLM tokens:', llm_info.get('tokens', 0), 'time(s):', f"{llm_info.get('time', 0.0):.3f}")
                print('Similitud entre respuestas (0-1):', f"{sim_between:.3f}" if sim_between is not None else 'n/a')
            else:
                print('LLM not run or error:', llm_info)
            print('Similitud GNN vs documento:', f"{sim_answer_doc:.3f}")
            print('--- end comparison summary ---\n')
        except Exception:
            pass
    except Exception:
        answers[q]['sim_between_responses'] = None

    # create comparison plots for this query (GNN vs LLM)
    try:
        plot_gnn_vs_llm_comparison(q, answers[q]['gnn']['human_answer'], answers[q]['llm'], best_doc_text, out_prefix='comparison')
    except Exception:
        pass

    return {'G': G, 'tf': tf, 'kge_result': result, 'ent_emb': ent_emb, 'rel_emb': rel_emb, 'gnn_model': gnn_model, 'gnn_z': gnn_z, 'answers': answers, 'llm': llm_responses}


if __name__ == '__main__':
    # small toy corpus in Spanish
    corpus_docs = [
        {'id': 1, 'doc': "La economía global está experimentando cambios significativos debido a la inteligencia artificial."},
        {'id': 2, 'doc': "El cambio climático afecta la biodiversidad y la vida de millones de especies en el planeta."},
        {'id': 3, 'doc': "La educación en línea ha transformado la manera en que los estudiantes acceden al conocimiento."},
        {'id': 4, 'doc': "El fútbol es el deporte más popular en muchos países y une a las personas a través de la pasión."}
    ]

    parser = argparse.ArgumentParser(description='Run graph KGE+GNN pipeline')
    parser.add_argument('--quick', action='store_true', help='Run a fast smoke test with reduced epochs and dims')
    # LLM runs by default; provide --no-llm to disable
    parser.add_argument('--run-llm', dest='run_llm', action='store_true', help='Enable LLM queries (requires ollama)')
    parser.add_argument('--no-llm', dest='run_llm', action='store_false', help='Disable LLM queries')
    parser.set_defaults(run_llm=True)
    args = parser.parse_args()
    q = input('Escribe tu consulta: ')
    out = run_pipeline(corpus_docs, user_queries=[q], run_llm=args.run_llm, quick=args.quick)
    # Print only concise human-readable answer (text) for user
    human = out['answers'][q]['gnn'].get('human_answer', {})
    answer_text = human.get('text', '') if human else ''
    if answer_text:
        print(answer_text)
    else:
        res = out['answers'][q]['gnn'].get('results', [])
        if res:
            print(res[0]['entity'])
        else:
            print('No puedo determinar una respuesta.')