from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from num2words import num2words
import re, os, torch, base64

AUDIO_DIR = "/workspace/audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

app = FastAPI(title="TTS API", version="1.0")

tts_model = None
SPEAKERS  = []
SPEAKER   = None


def get_tts():
    global tts_model, SPEAKERS, SPEAKER
    if tts_model is not None:
        return tts_model
    from TTS.api import TTS
    print("⏳ Chargement du modèle TTS...")
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                    gpu=torch.cuda.is_available())

    # ── Essayer toutes les méthodes pour récupérer les speakers ──────────────
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
                    # Forcer via hps
                    SPEAKERS = list(tts_model.synthesizer.tts_model.hps.data.spk2id.keys())
                    print(f"✅ Méthode 4 : {SPEAKERS}")
                except Exception as e:
                    print(f"❌ Impossible : {e}")
                    SPEAKERS = []

    SPEAKER = SPEAKERS[0] if SPEAKERS else None
    print(f"✅ Speaker sélectionné : {SPEAKER}")
    print(f"✅ Tous les speakers   : {SPEAKERS}")
    return tts_model

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

class AudioEntry(BaseModel):
    filename: str
    texte: str

class BatchRequest(BaseModel):
    langue: str = "fr"
    speaker_index: int = 2
    entrees: list[AudioEntry]

@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}

@app.get("/load")
def load_model():
    get_tts()
    return {"status": "modèle chargé", "speaker": SPEAKER}

@app.post("/save_audio_wav_oeuvre")
def save_audio_wav_oeuvre(req: BatchRequest):
    tts     = get_tts()
    speaker = SPEAKERS[req.speaker_index] if req.speaker_index < len(SPEAKERS) else SPEAKER
    resultats = []

    for entree in req.entrees:
        filename     = entree.filename.replace(".wav", "") + ".wav"
        path         = os.path.join(AUDIO_DIR, filename)
        texte_propre = preparer_texte(entree.texte, req.langue)

        try:
            tts.tts_to_file(
                text      = texte_propre,
                language  = req.langue,
                speaker   = speaker,
                file_path = path
            )
            with open(path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            resultats.append({
                "filename" : filename,
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