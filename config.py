import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY       = os.environ.get("SECRET_KEY", "replace-this-in-prod")
    DATA_CSV         = os.path.join(BASE_DIR, "processed_data_PublishedAlphaPapers_PubMed_RadiologyReview_UsePapersDirectly_V2_Sankey.csv")
    FAISS_INDEX      = os.path.join(BASE_DIR, "processed_data_PublishedAlphaPapers_PubMed_RadiologyReview_UsePapersDirectly_V2.index")
    KEY_TERMS_FILE   = os.path.join(BASE_DIR, "key_terms.json")
    OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
