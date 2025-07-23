import os
import json
import faiss
import numpy as np
import pandas as pd
import openai
import torch
import unicodedata
import logging
import re

from dotenv import load_dotenv
load_dotenv()

from flask import url_for
from transformers import AutoModel, AutoTokenizer
from config import Config  # ✅ Import config

# ─── CONFIG & LOGGING ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger()
for lib in ("httpcore", "httpx", "openai", "transformers"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ─── OPENAI CLIENT ────────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── LOAD DATA & INDEX ────────────────────────────────────────────────────────
data = pd.read_csv(Config.DATA_CSV, encoding="ISO-8859-1")
index = faiss.read_index(Config.FAISS_INDEX)

# ─── LOAD KEY‑TERM VARIATIONS FROM JSON ───────────────────────────────────────
with open(Config.KEY_TERMS_FILE, "r") as f:
    key_term_variations = json.load(f)

# ─── PUBMED‑BERT EMBEDDING MODEL ───────────────────────────────────────────────
logger.info("Loading PubMedBERT model…")
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed_text(text: str) -> np.ndarray:
    """Return a normalized [1×D] embedding for `text`."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

# ─── GPT CLARIFICATION ────────────────────────────────────────────────────────
def embed_text(text):
    """Generate normalized embeddings using PubMedBERT for a given text."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings from the [CLS] token
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    # Normalize embeddings to unit length
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return normalized_embeddings

def extract_key_terms_and_clarify(query):
    """Use GPT to directly extract key terms and clarify the query in a structured format."""
    clarification_prompt = f"""
    Given the query: '{query}', please structure your response as follows:
    Clinical Field: <Identify the clinical field of the query, such as cardiac, chest, neuro, msk, abdomen, oncology, or other.>
    Key Terms: <List key terms here that exactly match the terms in the query which are related to medical imaging or cpmputed tomography or radiology  without additional words.>
    Clarified Query: <Rephrase the query into a direct question written in a professional and scientific manner, as if a radiologist or medical physicist is asking it.>
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
             "content": (
                    "You are an expert radiologist and medical physicist. Provide structured and specific responses. "
                    "Only proceed if the query is clearly a DOI link starting with ""https://doi.org"" or related to CT, photon-counting CT, radiology, or medical imaging. "
                    "If the input is irrelevant, offensive, nonsensical, or inappropriate, respond exactly as:\n\n"
                    "Clinical Field: error_extracting\n"
                    "Key Terms: error_extracting\n"
                    "Clarified Query: error_extracting"
                )},
            {"role": "system", "content": f"Original user query: {query}"},
            {"role": "user", "content": clarification_prompt}
        ],
        max_tokens=150,
        temperature=0.3
    )

    # Extract the response content
    response_content = response.choices[0].message.content.strip()

    # Use regular expressions to extract structured fields
    clinical_field = re.search(r"Clinical Field:\s*(.+)", response_content)
    key_terms = re.search(r"Key Terms:\s*(.+)", response_content)
    clarified_query = re.search(r"Clarified Query:\s*(.+)", response_content)

    # Assign extracted values, removing any surrounding whitespace
    clinical_field = clinical_field.group(1).strip() if clinical_field else "error_extracting_clinical_field"
    key_terms = key_terms.group(1).strip() if key_terms else "error_extracting_key_terms"
    clarified_query = clarified_query.group(1).strip() if clarified_query else "error_extracting_clarified_query"

    # Ensure clinical field formatting matches dataset representation
    clinical_field = clinical_field.capitalize()  # Normalize for dataset consistency
    return clinical_field, key_terms, clarified_query

TOP_K = 50  # You can set this to whatever number you prefer for the top results

# Retrieve relevant documents with clinical field filtering first
from collections import Counter

from collections import Counter
import unicodedata
import logging

def retrieve_relevant_documents(query, top_k=50, similarity_threshold=0.85, max_results=3):
    """
    Retrieve the most relevant documents for a given query.
    1) Try direct matches on Title, DOI, Author, or Diagnostic Task.
    2) If none, fallback to GPT + FAISS semantic search with clinical field filtering.
    Returns:
       docs (list of dict), found (bool), message (str or None)
    """
    q = unicodedata.normalize('NFKC', query.strip().lower())

    # --- STEP 1: direct matches ---
    def build_docs_from_mask(mask):
        docs = []
        for _, row in data[mask].iterrows():
            docs.append({
                'citation':   row['Citation'],
                'text':       row['Summary'],
                'doi_url':    row['DOI Link'],
                'similarity': 1.0,
            })
        return docs

    # 1a) Title matches
    mask_title = data['Title'].astype(str).str.lower().str.contains(q, na=False)
    if mask_title.any():
        return build_docs_from_mask(mask_title), True, None

    # 1b) DOI matches
    mask_doi = data['DOI Link'].astype(str).str.lower().str.contains(q, na=False)
    if mask_doi.any():
        return build_docs_from_mask(mask_doi), True, None

    # 1c) Author matches
    mask_author = data['Author'].astype(str).str.lower().str.contains(q, na=False)
    if mask_author.any():
        return build_docs_from_mask(mask_author), True, None

    # 1d) Diagnostic Task matches
    if 'Diagnostic Task' in data.columns:
        mask_task = data['Diagnostic Task'].astype(str).str.lower().str.contains(q, na=False)
        if mask_task.any():
            return build_docs_from_mask(mask_task), True, None

    # --- STEP 2: fallback to GPT + FAISS ---
    logging.info("[STEP 1] No Title/DOI/Author/Task match → GPT/FAISS fallback")
    clinical_field, key_terms, clarified_query = extract_key_terms_and_clarify(query)
    q_emb = embed_text(clarified_query)
    D, I = index.search(q_emb, top_k)

    filtered = []
    for score, idx in zip(D[0], I[0]):
        if score < similarity_threshold:
            continue
        if idx < 0 or idx >= len(data):
            continue
        row = data.iloc[idx]
        if row['Clinical Field'].strip().capitalize() == clinical_field:
            filtered.append((idx, score))

    if not filtered:
        filtered = [
            (idx, score)
            for score, idx in zip(D[0], I[0])
            if score > similarity_threshold and 0 <= idx < len(data)
        ]

    matching = []
    key_terms_list = [t.strip().lower() for t in key_terms.split(',') if t.strip()]
    for idx, sim in filtered:
        row = data.iloc[idx]
        summary = row['Summary'].lower()
        term_counts = {
            term: any(v in summary for v in key_term_variations.get(term, [term]))
            for term in key_terms_list
        }
        cnt = sum(term_counts.values())
        if cnt > 0:
            matching.append({
                'citation':    row['Citation'],
                'text':        row['Summary'],
                'doi_url':     row['DOI Link'],
                'similarity':  sim,
                'match_count': cnt,
                'term_counts': term_counts
            })

    matching.sort(key=lambda x: (x['match_count'], x['similarity']), reverse=True)

    if not matching:
        return [], False, "No studies found matching the query. Please refine your search."

    return matching[:max_results], True, None


        
def clean_response_text(response_text):
    """Clean the response text by removing any leading '###' and trimming unnecessary leading or trailing whitespace."""
    return response_text.lstrip('#').strip()


def generate_response(query, retrieved_docs, max_tokens=500):
    """Generates a response based on the query and the most relevant document."""
    context = retrieved_docs[0]['text']
    references = [doc['citation'] for doc in retrieved_docs[:3]]
    doi_links = [doc['doi_url'] for doc in retrieved_docs[:3]]

    icon_url = url_for('static', filename='link-16.png')

    messages = [
        {"role": "system", "content": "You are a knowledgeable radiologist and computed tomography scientist."},
        {"role": "user", "content": f"Context:\n{context}\n\nPlease provide a very short cohesive summary of the study and key findings without listing them separately. Focus on creating a single, clear, and concise response.Additionally, if applicable, extract the scan protocol and reconstruction protocol information in the following format:\n\nScan Protocol:\nScan Mode: \nRotation Time (s): \nPitch: \nDetector Configuration (mm): \nkV (ref.): \nQuality ref. mAs: \nIQ level: \nCARE Dose4D: \nCARE k(e)V: \nCTDIvol (mGy): \nkeV: \nKernel (IR / FBP): \nWindow: \nSlice (mm): \nIncrement (mm): \n\nIf any information is not specified, write 'Not Specified'."}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7
    )

    response_text = clean_response_text(response.choices[0].message.content.strip())

    # Initialize the dictionary for protocol data
    combined_protocol = {}

    # Split the response to find the Scan Protocol section
    main_response = response_text.split("Scan Protocol:")[0].strip()

    if "Scan Protocol:" in response_text:
        protocol_text = response_text.split("Scan Protocol:")[1].strip()

        if "N/A" in protocol_text:
            combined_protocol = {"N/A": "Protocol details not available"}
        else:
            scan_protocol = {}
            recon_protocol = {}

            # Split the protocol into lines and process each line
            protocol_lines = protocol_text.split("\n")
            for line in protocol_lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    value = value.strip()
                    # Only include parameters that are not "Not Specified"
                    if value != "Not Specified":
                        if key in ["kV", "Kernel (IR / FBP)", "Window", "Slice (mm)", "Increment (mm)"]:
                            recon_protocol[key.strip()] = value
                        else:
                            scan_protocol[key.strip()] = value

            combined_protocol = {**scan_protocol, **recon_protocol}

    if not combined_protocol:
        combined_protocol = {"N/A": "Protocol details not available"}

    # Create the formatted references output
    formatted_references = ''.join([
        f"{idx + 1}. {citation} <a href='{doi}' target='_blank'><img src='{icon_url}' alt='link icon' style='width:16px; height:16px; vertical-align:middle;'></a><br>"
        for idx, (citation, doi) in enumerate(zip(references, doi_links))
    ])

    return main_response.strip(), formatted_references, combined_protocol
