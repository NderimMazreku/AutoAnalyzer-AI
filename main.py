import os
import json
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel


# =========================
# CONFIGURAZIONE
# =========================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(title="AutoAnalyzer AI")


# =========================
# MODELLO DATI
# =========================

class AutoData(BaseModel):
    marca: str
    modello: str
    anno: int
    prezzo: float
    km: int
    potenza_cv: int

    anticipo: float = 0
    durata_mesi: int = 84
    tasso_annuo: float = 8


# =========================
# HOMEPAGE
# =========================

@app.get("/")
def home():
    return FileResponse("index.html")


# =========================
# ANALISI AI
# =========================

def genera_analisi_ai(auto: AutoData):

    if client is None:
        return {
            "disponibile": False,
            "errore": "API key OpenAI non configurata."
        }

    prompt = f"""
Sei un assistente specializzato nell'analisi pre-acquisto
di automobili usate.

Analizza questa automobile:

Marca: {auto.marca}
Modello: {auto.modello}
Anno: {auto.anno}
Chilometri: {auto.km}
Potenza: {auto.potenza_cv} CV
Prezzo richiesto: {auto.prezzo} euro

Il tuo compito NON è dichiarare che l'auto è buona o cattiva
senza averla ispezionata.

Devi invece aiutare un potenziale acquirente a capire cosa
controllare.

Restituisci ESCLUSIVAMENTE JSON valido con questa struttura:

{{
  "riassunto": "breve analisi dell'auto",
  "punti_controllo": [
    "punto 1",
    "punto 2",
    "punto 3"
  ],
  "problemi_modello": [
    "possibile problema o componente da verificare"
  ],
  "test_drive": [
    "controllo durante il test drive"
  ],
  "domande_venditore": [
    "domanda da fare al venditore"
  ]
}}

Regole importanti:

- Non inventare guasti specifici come se fossero presenti
  su questa automobile.
- Se menzioni problemi conosciuti del modello, presentali
  come aspetti da verificare e non come difetti certi.
- Considera anno, chilometraggio, motorizzazione e tipo
  di automobile.
- Dai consigli pratici.
- Massimo 6 elementi per ogni lista.
- Scrivi tutto in italiano.
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            reasoning={
                "effort": "low"
            },
            input=prompt
        )

        testo = response.output_text.strip()

        # Nel caso il modello restituisca accidentalmente
        # un blocco markdown ```json
        if testo.startswith("```"):
            testo = testo.replace("```json", "")
            testo = testo.replace("```", "")
            testo = testo.strip()

        dati_ai = json.loads(testo)

        dati_ai["disponibile"] = True

        return dati_ai

    except Exception as e:

        print("ERRORE OPENAI:", str(e))

        return {
            "disponibile": False,
            "errore": "Analisi AI temporaneamente non disponibile."
        }


# =========================
# ANALISI AUTO
# =========================

@app.post("/analyze")
def analyze_auto(auto: AutoData):

    anno_corrente = datetime.now().year

    eta = max(
        anno_corrente - auto.anno,
        0
    )


    # =========================
    # KM ANNUI
    # =========================

    if eta > 0:
        km_annui = auto.km / eta
    else:
        km_annui = auto.km


    if km_annui < 8000:

        valutazione_km = (
            "Chilometraggio annuale basso"
        )

    elif km_annui <= 20000:

        valutazione_km = (
            "Chilometraggio annuale nella norma"
        )

    else:

        valutazione_km = (
            "Chilometraggio annuale elevato"
        )


    # =========================
    # PREZZO PER KM
    # =========================

    if auto.km > 0:
        prezzo_per_km = (
            auto.prezzo / auto.km
        )
    else:
        prezzo_per_km = 0


    # =========================
    # FINANZIAMENTO
    # =========================

    capitale = max(
        auto.prezzo - auto.anticipo,
        0
    )

    tasso_mensile = (
        auto.tasso_annuo / 100
    ) / 12


    if capitale == 0:

        rata = 0

    elif auto.durata_mesi <= 0:

        rata = 0

    elif tasso_mensile == 0:

        rata = (
            capitale /
            auto.durata_mesi
        )

    else:

        rata = (
            capitale
            * tasso_mensile
            * (1 + tasso_mensile)
            ** auto.durata_mesi
            /
            (
                (1 + tasso_mensile)
                ** auto.durata_mesi
                - 1
            )
        )


    totale_rate = (
        rata * auto.durata_mesi
    )

    interessi = (
        totale_rate - capitale
    )


    # =========================
    # CHECKLIST BASE
    # =========================

    checklist = [
        "Controllare lo storico delle manutenzioni.",
        "Verificare TÜV e relativa scadenza.",
        "Controllare eventuali incidenti precedenti.",
        "Verificare il numero di proprietari precedenti.",
        "Controllare pneumatici e freni.",
        "Controllare carrozzeria e verniciatura.",
        "Verificare eventuali spie sul quadro strumenti.",
        "Effettuare un test drive.",
        "Controllare che VIN e documenti coincidano."
    ]


    # =========================
    # ANALISI AI
    # =========================

    analisi_ai = genera_analisi_ai(auto)


    # Se AI funziona, utilizziamo anche
    # le sue domande specifiche.

    if analisi_ai.get("disponibile"):

        domande_venditore = (
            analisi_ai.get(
                "domande_venditore",
                []
            )
        )

        controlli_specifici = (
            analisi_ai.get(
                "punti_controllo",
                []
            )
        )

    else:

        # Fallback se OpenAI non è disponibile

        domande_venditore = [
            "L'auto ha avuto incidenti?",
            "Il libretto dei tagliandi è completo?",
            "Sono disponibili le fatture delle manutenzioni?",
            "Quando è stato effettuato l'ultimo tagliando?",
            "Ci sono problemi tecnici conosciuti?",
            "Per quale motivo viene venduta?"
        ]

        controlli_specifici = [
            "Verificare attentamente lo storico di manutenzione.",
            "Effettuare un controllo pre-acquisto presso un'officina indipendente.",
            "Controllare eventuali rumori, vibrazioni o spie durante il test drive."
        ]


    # =========================
    # RISPOSTA
    # =========================

    return {

        "auto":
            f"{auto.marca} {auto.modello}",

        "anno":
            auto.anno,

        "eta_anni":
            eta,

        "prezzo":
            round(auto.prezzo, 2),

        "km":
            auto.km,

        "potenza_cv":
            auto.potenza_cv,

        "prezzo_per_km":
            round(prezzo_per_km, 2),


        "finanziamento": {

            "anticipo":
                round(auto.anticipo, 2),

            "capitale_finanziato":
                round(capitale, 2),

            "durata_mesi":
                auto.durata_mesi,

            "tasso_annuo":
                auto.tasso_annuo,

            "rata_mensile":
                round(rata, 2),

            "interessi_totali":
                round(interessi, 2),

            "totale_rate":
                round(totale_rate, 2),

            "costo_totale_con_anticipo":
                round(
                    totale_rate
                    + auto.anticipo,
                    2
                )
        },


        "analisi": {

            "km_annui":
                round(km_annui),

            "valutazione_km":
                valutazione_km,

            "checklist":
                checklist,

            "domande_venditore":
                domande_venditore,

            "controlli_specifici":
                controlli_specifici
        },


        "ai": analisi_ai
    }