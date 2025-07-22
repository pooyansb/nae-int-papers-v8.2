# search.py

from flask import Blueprint, request, redirect, url_for, render_template
import pandas as pd
import json
from markupsafe import Markup

from services import (
    retrieve_relevant_documents,
    generate_response,
    extract_key_terms_and_clarify
)
from config import Config

search_bp = Blueprint("search", __name__, url_prefix="/search")

# 1) Load your CT study metadata
data = pd.read_csv(Config.DATA_CSV, encoding="ISO-8859-1")

# 2) Load your CT keyword list once
with open(Config.KEY_TERMS_FILE, "r") as f:
    terms_dict = json.load(f)
CT_TERMS = [
    syn.lower()
    for syn_list in terms_dict.values()
    for syn in syn_list
]

@search_bp.route("", methods=["GET", "POST"])
def search():
    # ── Handle Sankey‑diagram DOI clicks ─────────────────────────────
    doi_list = request.args.getlist("doi")
    if doi_list:
        subset = data[data["DOI Link"].isin(doi_list)]
        if subset.empty:
            return render_template(
                "search.html",
                no_dois=True, irrelevant=False,
                no_results=False, sankey_refs=None,
                active_tab="home"
            )
        icon_url = url_for("static", filename="link-16.png")
        refs_html = ""
        for i, doi in enumerate(doi_list):
            citation = subset[subset["DOI Link"] == doi].iloc[0]["Citation"]
            refs_html += (
                f"{i+1}. {citation} "
                f"<a href='{doi}' target='_blank'>"
                f"<img src='{icon_url}' style='width:16px;vertical-align:middle'></a><br>"
            )
        return render_template(
            "search.html",
            no_dois=False, irrelevant=False,
            no_results=False,
            sankey_refs=Markup(refs_html),
            active_tab="home"
        )

    # ── Only accept POST for normal searches ─────────────────────────
    if request.method != "POST":
        return redirect(url_for("home.home"))

    original_query = request.form.get("query", "").strip()
    if not original_query:
        return redirect(url_for("home.home"))

    lower_q = original_query.lower()

    # ── Shortcut: direct DOI input ──────────────────────────────────
    if original_query.startswith("https://doi.org/"):
        clinical_field, key_terms, clarified_query = "bypass", "doi", original_query
    else:
        # 3) GPT‐based extraction + clarification
        clinical_field, key_terms, clarified_query = extract_key_terms_and_clarify(original_query)
        clarified_query = clarified_query.strip().lstrip("*").strip()

        # 4) Off‑topic / irrelevance check *with local rescue*
        #    If GPT says “Other” *and* we found *no* CT term locally, flag irrelevant.
        found_local = (
            any(term in lower_q for term in CT_TERMS)
            or any(term in key_terms.lower() for term in CT_TERMS)
            or any(term in clarified_query.lower() for term in CT_TERMS)
        )

        # Override clarified_query for rescued CT terms
        if found_local:
            clarified_query = original_query

        if (
            clinical_field.lower() == "other" and not found_local
            or "error_extracting" in clinical_field.lower()
            or "error_extracting" in key_terms.lower()
            or "error_extracting" in clarified_query.lower()
            or clarified_query.lower().startswith("could you provide more information")
        ):
            return render_template(
                "search.html",
                no_dois=False, irrelevant=True,
                no_results=False, sankey_refs=None,
                active_tab="home"
            )

    # ── 5) Pull top‐3 PCCT studies from your pool ───────────────────
    docs, found, _ = retrieve_relevant_documents(clarified_query)
    docs = docs[:3]

    # ── 6) CT‑related but zero matches → “no_results” ───────────────
    if not found:
        return render_template(
            "search.html",
            no_dois=False, irrelevant=False,
            no_results=True, sankey_refs=None,
            active_tab="home"
        )

    # ── 7) Matches found → generate GPT response & protocol ────────
    response, formatted_refs, protocol = generate_response(clarified_query, docs)
    return render_template(
        "search.html",
        no_dois=False, irrelevant=False,
        no_results=False, sankey_refs=None,
        clarified_query=original_query,
        response=response,
        protocol=protocol,
        formatted_refs=Markup(formatted_refs),
        active_tab="home"
    )
