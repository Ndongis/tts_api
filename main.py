from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from num2words import num2words
import re, os, torch

AUDIO_DIR = "/workspace/audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI(title="TTS API", version="1.0")

# ── Chargement lazy ───────────────────────────────────────────────────────────
tts_model   = None
SPEAKERS    = []
SPEAKER     = None

def get_tts():
    global tts_model, SPEAKERS, SPEAKER

    if tts_model is not None:
        return tts_model  # déjà chargé

    from TTS.api import TTS
    print("⏳ Chargement du modèle TTS...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                    gpu=torch.cuda.is_available())

    # Récupérer les speakers sans planter
    try:
        SPEAKERS = list(tts_model.synthesizer.tts_model.speaker_manager.name_to_id.keys())
    except:
        try:
            SPEAKERS = list(tts_model.synthesizer.tts_model.hps.data.spk2id.keys())
        except:
            SPEAKERS = ["default"]

    SPEAKER = SPEAKERS[2] if len(SPEAKERS) > 2 else SPEAKERS[0]
    print(f"✅ Modèle prêt — Speaker : {SPEAKER}")
    print(f"✅ Speakers : {SPEAKERS}")
    return tts_model


# ─────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────
def preparer_texte(texte: str, langue: str = "fr") -> str:
    def remplacer_nombre(match):
        nombre = match.group()
        try:
            if '.' in nombre or ',' in nombre:
                return num2words(float(nombre.replace(',', '.')), lang=langue)
            return num2words(int(nombre), lang=langue)
        except:
            return nombre
    return re.sub(r'\d+[.,]?\d*', remplacer_nombre, texte)


# ─────────────────────────────────────────
# MODÈLES PYDANTIC
# ─────────────────────────────────────────
class AudioEntry(BaseModel):
    filename: str
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
        "status"  : "ok",
        "gpu"     : torch.cuda.is_available(),
        "modele"  : "non chargé" if tts_model is None else "prêt",
        "speaker" : SPEAKER,
        "speakers": SPEAKERS
    }

@app.get("/load")
def load_model():
    """Charger le modèle manuellement avant la première requête"""
    get_tts()
    return {"status": "modèle chargé", "speaker": SPEAKER, "speakers": SPEAKERS}

@app.post("/generate/batch")
def generate_batch(req: BatchRequest):
    tts = get_tts()  # charge si pas encore fait
    speaker = SPEAKERS[req.speaker_index] if req.speaker_index < len(SPEAKERS) else SPEAKER
    resultats = []

    for entree in req.entrees:
        filename = entree.filename.replace(".wav", "") + ".wav"
        path     = os.path.join(AUDIO_DIR, filename)
        texte_propre = preparer_texte(entree.texte, req.langue)

        try:
            tts.tts_to_file(
                text     = texte_propre,
                language = req.langue,
                speaker  = speaker,
                file_path= path
            )
            resultats.append({
                "filename" : filename,
                "texte_lu" : texte_propre,
                "url"      : f"/audio/{filename}",
                "status"   : "ok"
            })
        except Exception as e:
            resultats.append({
                "filename": filename,
                "status"  : "erreur",
                "detail"  : str(e)
            })

    return {"total": len(resultats), "resultats": resultats}

@app.get("/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} introuvable")
    return FileResponse(path, media_type="audio/wav", filename=filename)

@app.get("/audio/list")
def list_audios():
    fichiers = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    return {"total": len(fichiers), "fichiers": fichiers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)