# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from TTS.api import TTS
from num2words import num2words
import re, os, torch

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI(title="TTS API", version="1.0")

print("⏳ Chargement du modèle TTS...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
          gpu=torch.cuda.is_available())
SPEAKER = tts.speakers[2]
print(f"✅ Modèle prêt — Speaker : {SPEAKER}")


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
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "speaker": SPEAKER
    }


# ── 1. Recevoir la liste et générer les WAVs (appelé par ton serveur) ─────────
@app.post("/generate/batch")
def generate_batch(req: BatchRequest):
    """
    Appelé par TON SERVEUR (pas Unity).
    Reçoit une liste {filename, texte} et génère un WAV par entrée.
    """
    speaker = tts.speakers[req.speaker_index] if req.speaker_index < len(tts.speakers) else SPEAKER
    resultats = []

    for entree in req.entrees:
        filename = entree.filename.replace(".wav", "") + ".wav"
        path = os.path.join(AUDIO_DIR, filename)
        texte_propre = preparer_texte(entree.texte, req.langue)

        try:
            tts.tts_to_file(
                text=texte_propre,
                language=req.langue,
                speaker=speaker,
                file_path=path
            )
            resultats.append({
                "filename": filename,
                "texte_lu": texte_propre,
                "url": f"/audio/{filename}",
                "status": "ok"
            })
        except Exception as e:
            resultats.append({
                "filename": filename,
                "status": "erreur",
                "detail": str(e)
            })

    return {"total": len(resultats), "resultats": resultats}


# ── 2. Unity lit un WAV ───────────────────────────────────────────────────────
@app.get("/audio/{filename}")
def get_audio(filename: str):
    """
    Appelé par UNITY uniquement pour lire un WAV.
    GET /audio/victoire.wav
    """
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} introuvable")
    return FileResponse(path, media_type="audio/wav", filename=filename)


# ── 3. Unity liste les WAVs disponibles ──────────────────────────────────────
@app.get("/audio/list")
def list_audios():
    """
    Unity peut vérifier quels fichiers sont disponibles.
    """
    fichiers = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    return {
        "total": len(fichiers),
        "fichiers": [{"filename": f, "url": f"/audio/{f}"} for f in fichiers]
    }


# ─────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)