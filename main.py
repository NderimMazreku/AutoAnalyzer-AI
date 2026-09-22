import os
import json
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(title="AutoAnalyzer AI")


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
    lingua: str = "de"


@app.get("/")
def home():
    return FileResponse("index.html")


# ==========================================
# TRADUZIONI
# ==========================================

def get_language_name(lang):
    languages = {
        "de": "German",
        "en": "English",
        "it": "Italian"
    }

    return languages.get(lang, "German")


def get_local_text(lang):

    texts = {

        "de": {
            "km_low": "Niedrige jährliche Fahrleistung",
            "km_normal": "Normale jährliche Fahrleistung",
            "km_high": "Hohe jährliche Fahrleistung",

            "checklist": [
                "Wartungshistorie und Serviceheft prüfen.",
                "HU/TÜV und Gültigkeit prüfen.",
                "Nach früheren Unfallschäden fragen.",
                "Anzahl der Vorbesitzer prüfen.",
                "Reifen und Bremsen kontrollieren.",
                "Karosserie und Lack auf Reparaturspuren prüfen.",
                "Kontrollleuchten im Kombiinstrument beachten.",
                "Eine ausführliche Probefahrt durchführen.",
                "FIN/VIN mit den Fahrzeugpapieren vergleichen."
            ],

            "fallback_questions": [
                "Hatte das Fahrzeug einen Unfall?",
                "Ist das Serviceheft vollständig?",
                "Sind Wartungsrechnungen vorhanden?",
                "Wann wurde die letzte Inspektion durchgeführt?",
                "Sind technische Probleme bekannt?",
                "Warum wird das Fahrzeug verkauft?"
            ],

            "fallback_checks": [
                "Wartungshistorie sorgfältig prüfen.",
                "Vor dem Kauf eine unabhängige Fahrzeugprüfung durchführen lassen.",
                "Bei der Probefahrt auf Geräusche, Vibrationen und Warnleuchten achten."
            ]
        },

        "en": {
            "km_low": "Low annual mileage",
            "km_normal": "Normal annual mileage",
            "km_high": "High annual mileage",

            "checklist": [
                "Check the maintenance and service history.",
                "Check the current roadworthiness inspection.",
                "Ask about previous accident damage.",
                "Check the number of previous owners.",
                "Inspect the tyres and brakes.",
                "Inspect the bodywork and paint for repair marks.",
                "Check for warning lights on the dashboard.",
                "Carry out a thorough test drive.",
                "Compare the VIN with the vehicle documents."
            ],

            "fallback_questions": [
                "Has the vehicle ever been involved in an accident?",
                "Is the service history complete?",
                "Are maintenance invoices available?",
                "When was the last service carried out?",
                "Are there any known technical problems?",
                "Why is the vehicle being sold?"
            ],

            "fallback_checks": [
                "Check the maintenance history carefully.",
                "Consider an independent pre-purchase inspection.",
                "Check for unusual noises, vibrations or warning lights during the test drive."
            ]
        },

        "it": {
            "km_low": "Chilometraggio annuale basso",
            "km_normal": "Chilometraggio annuale nella norma",
            "km_high": "Chilometraggio annuale elevato",

            "checklist": [
                "Controllare lo storico delle manutenzioni.",
                "Verificare TÜV/revisione e relativa scadenza.",
                "Controllare eventuali incidenti precedenti.",
                "Verificare il numero di proprietari precedenti.",
                "Controllare pneumatici e freni.",
                "Controllare carrozzeria e verniciatura.",
                "Verificare eventuali spie sul quadro strumenti.",
                "Effettuare un test drive approfondito.",
                "Controllare che VIN e documenti coincidano."
            ],

            "fallback_questions": [
                "L'auto ha avuto incidenti?",
                "Il libretto dei tagliandi è completo?",
                "Sono disponibili le fatture delle manutenzioni?",
                "Quando è stato effettuato l'ultimo tagliando?",
                "Ci sono problemi tecnici conosciuti?",
                "Per quale motivo viene venduta?"
            ],

            "fallback_checks": [
                "Verificare attentamente lo storico di manutenzione.",
                "Effettuare un controllo pre-acquisto presso un'officina indipendente.",
                "Controllare rumori, vibrazioni e spie durante il test drive."
            ]
        }
    }

    return texts.get(lang, texts["de"])


# ==========================================
# ANALISI AI
# ==========================================

def genera_analisi_ai(auto: AutoData):

    if client is None:
        return {
            "disponibile": False,
            "errore": "OpenAI API not configured."
        }

    language = get_language_name(auto.lingua)

    prompt = f"""
You are an expert assistant for used-car pre-purchase analysis.

Analyze this vehicle:

Make: {auto.marca}
Model: {auto.modello}
Year: {auto.anno}
Mileage: {auto.km} km
Power: {auto.potenza_cv} HP
Price: {auto.prezzo} EUR

IMPORTANT LANGUAGE INSTRUCTION:

The selected language is {language}.

Every user-facing sentence in your response MUST be written in {language}.
Do not use Italian unless the selected language is Italian.

Your task is to help the buyer understand what should be checked
before purchasing this vehicle.

Do not claim that this specific vehicle has a defect unless that
information was explicitly provided.

When mentioning known or commonly reported model-specific issues,
describe them only as potential points worth checking.

Return ONLY valid JSON.

Use exactly this JSON structure:

{{
    "riassunto": "short analysis",
    "punti_controllo": [
        "specific item worth checking"
    ],
    "problemi_modello": [
        "potential model-specific issue worth checking"
    ],
    "test_drive": [
        "specific test-drive check"
    ],
    "domande_venditore": [
        "useful question to ask the seller"
    ]
}}

Rules:

- Consider the exact model, year, mileage and engine/powertrain when identifiable.
- Give practical advice.
- Do not invent accidents.
- Do not invent service history.
- Do not invent defects.
- Maximum 6 items per list.
- JSON only.
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            reasoning={"effort": "low"},
            input=prompt
        )

        testo = response.output_text.strip()

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
            "errore": "AI analysis temporarily unavailable."
        }


# ==========================================
# ENDPOINT ANALISI
# ==========================================

@app.post("/analyze")
def analyze_auto(auto: AutoData):

    # Controllo lingua
    if auto.lingua not in ["de", "en", "it"]:
        auto.lingua = "de"

    local = get_local_text(auto.lingua)

    # Età veicolo
    anno_corrente = datetime.now().year
    eta = max(anno_corrente - auto.anno, 0)

    # Km annui
    if eta > 0:
        km_annui = auto.km / eta
    else:
        km_annui = auto.km

    if km_annui < 8000:
        valutazione_km = local["km_low"]

    elif km_annui <= 20000:
        valutazione_km = local["km_normal"]

    else:
        valutazione_km = local["km_high"]

    # Prezzo per km
    if auto.km > 0:
        prezzo_per_km = auto.prezzo / auto.km
    else:
        prezzo_per_km = 0

    # ==========================================
    # FINANZIAMENTO
    # ==========================================

    capitale = max(auto.prezzo - auto.anticipo, 0)

    tasso_mensile = (auto.tasso_annuo / 100) / 12

    if capitale == 0:
        rata = 0

    elif auto.durata_mesi <= 0:
        rata = 0

    elif tasso_mensile == 0:
        rata = capitale / auto.durata_mesi

    else:
        rata = (
            capitale
            * tasso_mensile
            * (1 + tasso_mensile) ** auto.durata_mesi
            /
            (
                (1 + tasso_mensile) ** auto.durata_mesi
                - 1
            )
        )

    totale_rate = rata * auto.durata_mesi
    interessi = totale_rate - capitale

    # ==========================================
    # AI
    # ==========================================

    analisi_ai = genera_analisi_ai(auto)

    if analisi_ai.get("disponibile"):

        domande_venditore = analisi_ai.get(
            "domande_venditore",
            []
        )

        controlli_specifici = analisi_ai.get(
            "punti_controllo",
            []
        )

        problemi_modello = analisi_ai.get(
            "problemi_modello",
            []
        )

        test_drive = analisi_ai.get(
            "test_drive",
            []
        )

        riassunto_ai = analisi_ai.get(
            "riassunto",
            ""
        )

    else:

        domande_venditore = local["fallback_questions"]
        controlli_specifici = local["fallback_checks"]
        problemi_modello = []
        test_drive = []
        riassunto_ai = ""

    # ==========================================
    # RISPOSTA
    # ==========================================

    return {

        "auto": f"{auto.marca} {auto.modello}",

        "lingua": auto.lingua,

        "anno": auto.anno,

        "eta_anni": eta,

        "prezzo": round(auto.prezzo, 2),

        "km": auto.km,

        "potenza_cv": auto.potenza_cv,

        "prezzo_per_km": round(prezzo_per_km, 2),

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
                    totale_rate + auto.anticipo,
                    2
                )
        },

        "analisi": {

            "km_annui":
                round(km_annui),

            "valutazione_km":
                valutazione_km,

            "checklist":
                local["checklist"],

            "domande_venditore":
                domande_venditore,

            "controlli_specifici":
                controlli_specifici,

            "problemi_modello":
                problemi_modello,

            "test_drive":
                test_drive,

            "riassunto_ai":
                riassunto_ai
        },

        "ai": analisi_ai
    }