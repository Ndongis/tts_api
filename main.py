from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from num2words import num2words
import re, os, torch, base64

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
AUDIO_DIR = "/workspace/audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI(title="TTS API", version="1.0")

tts_model = None
SPEAKERS  = []
SPEAKER   = None


# ─────────────────────────────────────────
# CHARGEMENT MODELE
# ─────────────────────────────────────────
def get_tts():
    global tts_model, SPEAKERS, SPEAKER
    if tts_model is not None:
        return tts_model

    from TTS.api import TTS
    print("⏳ Chargement du modèle TTS...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                    gpu=torch.cuda.is_available())

    try:
        SPEAKERS = list(tts_model.synthesizer.tts_model.speaker_manager.name_to_id.keys())
        print(f"✅ Méthode 1 : {SPEAKERS}")
    except:
        try:
            SPEAKERS = list(tts_model.synthesizer.tts_model.speaker_manager.speakers.keys())
            print(f"✅ Méthode 2 : {SPEAKERS}")
        except:
            try:
                SPEAKERS = tts_model.synthesizer.tts_model.speaker_manager.speaker_names
                print(f"✅ Méthode 3 : {SPEAKERS}")
            except:
                try:
                    SPEAKERS = list(tts_model.synthesizer.tts_model.hps.data.spk2id.keys())
                    print(f"✅ Méthode 4 : {SPEAKERS}")
                except Exception as e:
                    print(f"❌ Impossible : {e}")
                    SPEAKERS = []

    SPEAKER = SPEAKERS[0] if SPEAKERS else None
    print(f"✅ Speaker sélectionné : {SPEAKER}")
    print(f"✅ Tous les speakers   : {SPEAKERS}")
    return tts_model


# ─────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────
NOMS_COLONNES = {
    "titre"               : "Titre",
    "auteur"              : "Auteur",
    "date"                : "Date",
    "technique"           : "Technique",
    "sujet"               : "Sujet",
    "inscription"         : "Inscription",
    "description_visuelle": "Description visuelle",
    "historique"          : "Historique",
}

def preparer_texte(texte: str, filename: str = "", langue: str = "fr") -> str:

    # ── 1. Convertir les chiffres en lettres ─────────────────────────────────
    def remplacer_nombre(match):
        nombre = match.group()
        try:
            if '.' in nombre or ',' in nombre:
                return num2words(float(nombre.replace(',', '.')), lang=langue)
            return num2words(int(nombre), lang=langue)
        except:
            return nombre

    texte_converti = re.sub(r'\d+[.,]?\d*', remplacer_nombre, texte)

    # ── 2. Extraire le nom de la colonne depuis le filename ───────────────────
    # "3/titre"   → colonne = "titre"
    # "titre"     → colonne = "titre"
    colonne     = filename.split("/")[-1].replace(".wav", "")
    nom_colonne = NOMS_COLONNES.get(colonne, colonne.replace("_", " ").capitalize())

    # ── 3. Ajouter le nom de colonne avec pause naturelle ─────────────────────
    texte_final = f"{nom_colonne}... {texte_converti}"

    return texte_final


# ─────────────────────────────────────────
# MODÈLES PYDANTIC
# ─────────────────────────────────────────
class AudioEntry(BaseModel):
    filename: str   # ex: "3/titre"
    texte: str

class BatchRequest(BaseModel):
    langue: str = "fr"
    speaker_index: int = 2
    entrees: list[AudioEntry]


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu"   : torch.cuda.is_available(),
        "modele": "non chargé" if tts_model is None else "prêt"
    }

@app.get("/load")
def load_model():
    get_tts()
    return {
        "status" : "modèle chargé",
        "speaker": SPEAKER,
        "speakers": SPEAKERS
    }

@app.get("/speakers")
def list_speakers():
    get_tts()
    return {"speakers": SPEAKERS, "speaker_actuel": SPEAKER}


# ── Générer + renvoyer les WAVs en base64 vers Django ────────────────────────
@app.post("/save_audio_wav_oeuvre")
def save_audio_wav_oeuvre(req: BatchRequest):
    tts     = get_tts()
    speaker = SPEAKERS[req.speaker_index] if req.speaker_index < len(SPEAKERS) else (SPEAKERS[0] if SPEAKERS else None)
    resultats = []

    for entree in req.entrees:
        # entree.filename = "3/titre"
        filename = entree.filename.replace(".wav", "") + ".wav"  # "3/titre.wav"
        path     = os.path.join(AUDIO_DIR, filename)             # /workspace/audios/3/titre.wav

        # Créer le sous-dossier de l'oeuvre
        os.makedirs(os.path.dirname(path), exist_ok=True)

        texte_propre = preparer_texte(entree.texte, entree.filename, req.langue)
        print(f"🎙️ {filename} → {texte_propre}")

        try:
            kwargs = {
                "text"     : texte_propre,
                "language" : req.langue,
                "file_path": path
            }
            if speaker:
                kwargs["speaker"] = speaker

            tts.tts_to_file(**kwargs)

            with open(path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            resultats.append({
                "filename" : filename,
                "texte_lu" : texte_propre,
                "audio_b64": audio_b64,
                "status"   : "ok"
            })

        except Exception as e:
            resultats.append({
                "filename": filename,
                "status"  : "erreur",
                "detail"  : str(e)
            })

    return {"total": len(resultats), "resultats": resultats}


# ── Unity lit un WAV ──────────────────────────────────────────────────────────
@app.get("/audio/{oeuvre_id}/{filename}")
def get_audio(oeuvre_id: str, filename: str):
    path = os.path.join(AUDIO_DIR, oeuvre_id, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{oeuvre_id}/{filename} introuvable")
    return FileResponse(path, media_type="audio/wav", filename=filename)


# ── Lister les WAVs d'une oeuvre ─────────────────────────────────────────────
@app.get("/audio/{oeuvre_id}")
def list_audios_oeuvre(oeuvre_id: str):
    dossier = os.path.join(AUDIO_DIR, oeuvre_id)
    if not os.path.exists(dossier):
        raise HTTPException(status_code=404, detail=f"Oeuvre {oeuvre_id} introuvable")
    fichiers = [f for f in os.listdir(dossier) if f.endswith(".wav")]
    return {
        "oeuvre_id": oeuvre_id,
        "total"    : len(fichiers),
        "fichiers" : fichiers
    }


# ─────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)