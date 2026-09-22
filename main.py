import os
import json
import socket
import ipaddress
import io
from datetime import datetime, timezone
from threading import Lock
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from xml.sax.saxutils import escape


# =========================================================
# CONFIGURAZIONE
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(title="AutoAnalyzer AI")


# =========================================================
# LIMITI DI UTILIZZO API
# =========================================================

# Limiti separati: un utente può importare fino a 5 annunci e
# avviare fino a 5 analisi AI al giorno.
DAILY_IMPORT_LIMIT = 5
DAILY_ANALYSIS_LIMIT = 5

_rate_limit_lock = Lock()
_rate_limit_usage = {}


def _client_ip(request: Request):
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def _check_daily_limit(request: Request, action: str, limit: int):
    """
    Semplice rate limit in memoria per l'MVP.
    Si azzera ogni giorno UTC e anche quando il servizio Render si riavvia.
    """
    ip = _client_ip(request)
    today = datetime.now(timezone.utc).date().isoformat()
    key = (today, ip, action)

    with _rate_limit_lock:
        used = _rate_limit_usage.get(key, 0)

        if used >= limit:
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "daily_limit_reached",
                    "action": action,
                    "limit": limit,
                    "message": "Daily usage limit reached. Please try again tomorrow."
                }
            )

        _rate_limit_usage[key] = used + 1



# =========================================================
# MODELLI
# =========================================================

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

    carburante: str | None = None
    cambio: str | None = None
    tuv: str | None = None
    incidenti: str | None = None

    modifiche: list[str] = Field(default_factory=list)


class ListingURL(BaseModel):
    url: str
    lingua: str = "de"


class PDFReportRequest(BaseModel):
    auto: AutoData
    result: dict


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():
    return FileResponse("index.html")


# =========================================================
# LINGUE
# =========================================================

def get_language_name(lang):
    languages = {
        "de": "German",
        "en": "English",
        "it": "Italian",
        "es": "Spanish"
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
                "Warum wird das Fahrzeug verkauft?",
                "Sind Unfälle oder Nachlackierungen bekannt?",
                "Ist die Wartungshistorie vollständig dokumentiert?",
                "Welche Reparaturen wurden zuletzt durchgeführt?",
                "Gibt es aktuell bekannte technische Probleme?",
                "Kann das Fahrzeug vor dem Kauf unabhängig geprüft werden?"
            ],

            "fallback_checks": [
                "Motor auf ungewöhnliche Geräusche und Undichtigkeiten prüfen.",
                "Getriebe und Kupplung bei der Probefahrt prüfen.",
                "Bremsen und Reifen kontrollieren.",
                "Fehlerspeicher auslesen lassen.",
                "Unterboden und Fahrwerk prüfen.",
                "Servicehistorie mit Rechnungen nachvollziehen."
            ]
        },

        "en": {
            "km_low": "Low annual mileage",
            "km_normal": "Normal annual mileage",
            "km_high": "High annual mileage",

            "checklist": [
                "Check maintenance history and service records.",
                "Check roadworthiness inspection validity.",
                "Ask about previous accident damage.",
                "Check the number of previous owners.",
                "Inspect tyres and brakes.",
                "Inspect bodywork and paint for repair traces.",
                "Check dashboard warning lights.",
                "Carry out a thorough test drive.",
                "Compare the VIN with the vehicle documents."
            ],

            "fallback_questions": [
                "Why is the vehicle being sold?",
                "Are there any known accidents or repainting?",
                "Is the maintenance history fully documented?",
                "What repairs were carried out recently?",
                "Are there any currently known technical problems?",
                "Can the vehicle be independently inspected before purchase?"
            ],

            "fallback_checks": [
                "Check the engine for unusual noises and leaks.",
                "Check the transmission and clutch during the test drive.",
                "Inspect brakes and tyres.",
                "Have the diagnostic fault memory checked.",
                "Inspect the underbody and suspension.",
                "Verify the service history with invoices."
            ]
        },

        "it": {
            "km_low": "Chilometraggio annuale basso",
            "km_normal": "Chilometraggio annuale normale",
            "km_high": "Chilometraggio annuale elevato",

            "checklist": [
                "Controllare storico manutenzione e libretto service.",
                "Controllare revisione/TÜV e validità.",
                "Chiedere informazioni su eventuali incidenti precedenti.",
                "Controllare il numero dei precedenti proprietari.",
                "Controllare pneumatici e freni.",
                "Controllare carrozzeria e vernice per eventuali riparazioni.",
                "Controllare eventuali spie accese sul quadro strumenti.",
                "Effettuare una prova su strada approfondita.",
                "Confrontare VIN/telaio con i documenti dell'auto."
            ],

            "fallback_questions": [
                "Perché viene venduta l'auto?",
                "Ci sono incidenti o riverniciature conosciute?",
                "Lo storico della manutenzione è completamente documentato?",
                "Quali riparazioni sono state effettuate recentemente?",
                "Ci sono problemi tecnici attualmente conosciuti?",
                "È possibile far controllare l'auto da un'officina indipendente?"
            ],

            "fallback_checks": [
                "Controllare il motore per rumori insoliti e perdite.",
                "Controllare cambio e frizione durante la prova su strada.",
                "Controllare freni e pneumatici.",
                "Far leggere la memoria errori tramite diagnosi.",
                "Controllare sottoscocca e sospensioni.",
                "Verificare lo storico service tramite fatture."
            ]
        },

        "es": {
            "km_low": "Kilometraje anual bajo",
            "km_normal": "Kilometraje anual normal",
            "km_high": "Kilometraje anual alto",

            "checklist": [
                "Comprobar el historial de mantenimiento y el libro de revisiones.",
                "Comprobar la ITV/TÜV y su validez.",
                "Preguntar por posibles daños de accidentes anteriores.",
                "Comprobar el número de propietarios anteriores.",
                "Revisar neumáticos y frenos.",
                "Revisar la carrocería y la pintura en busca de reparaciones.",
                "Comprobar si hay testigos de advertencia encendidos en el cuadro de instrumentos.",
                "Realizar una prueba de conducción completa.",
                "Comparar el VIN/número de bastidor con la documentación del vehículo."
            ],

            "fallback_questions": [
                "¿Por qué se vende el vehículo?",
                "¿Se conocen accidentes o repintados?",
                "¿Está completamente documentado el historial de mantenimiento?",
                "¿Qué reparaciones se han realizado recientemente?",
                "¿Hay algún problema técnico conocido actualmente?",
                "¿Se puede revisar el vehículo en un taller independiente antes de comprarlo?"
            ],

            "fallback_checks": [
                "Comprobar el motor en busca de ruidos inusuales y fugas.",
                "Comprobar la transmisión y el embrague durante la prueba de conducción.",
                "Revisar frenos y neumáticos.",
                "Realizar una lectura de la memoria de averías mediante diagnóstico.",
                "Revisar los bajos y la suspensión.",
                "Verificar el historial de mantenimiento mediante facturas."
            ]
        }
    }

    return texts.get(lang, texts["de"])


# =========================================================
# ANALISI AI
# =========================================================

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

Fuel: {auto.carburante or "Not provided"}
Transmission: {auto.cambio or "Not provided"}
TÜV/HU: {auto.tuv or "Not provided"}
Accident information: {auto.incidenti or "Not provided"}

Modifications/tuning:
{", ".join(auto.modifiche) if auto.modifiche else "None provided"}

IMPORTANT LANGUAGE INSTRUCTION:

The selected language is {language}.

ALL user-facing text MUST be written exclusively in {language}.

This applies to EVERY textual value in the JSON response, including:

- riassunto
- auto_score.spiegazione
- every rischi.*.motivo
- every item in punti_controllo
- every item in problemi_modello
- every item in test_drive
- every item in domande_venditore

The JSON property names must remain exactly as specified below,
but ALL textual values must be in {language}.

Never reuse Italian, German, English, or another language from a
previous analysis when the selected language is {language}.

Your task is to help the buyer understand what should be checked
before purchasing this vehicle.

Do not claim that this specific vehicle has a defect unless that
information was explicitly provided.

When mentioning known or commonly reported model-specific issues,
describe them only as potential points worth checking.

If modifications or tuning are provided, specifically explain what
should be checked because of those modifications.

If Stage 1, Stage 2 or Stage 3 tuning is present, consider appropriate
checks concerning:

- engine
- turbocharger
- drivetrain
- clutch/transmission
- emissions equipment

If modifications are listed, consider whether documentation,
approval, registration, ABE or technical inspection may need to be
verified where relevant.

Use TÜV/HU, accident information, fuel type and transmission
information when generating checks and seller questions.


PRICE CHECK:

Estimate whether the asking price appears reasonable based ONLY on the
vehicle information currently available.

Consider:

- make and model
- year
- mileage
- engine / power
- fuel type
- transmission
- accident information
- TÜV/HU information
- modifications or tuning
- vehicle age
- approximate annual mileage

Return a cautious indicative price assessment.

IMPORTANT PRICE CHECK RULES:

This is NOT a verified real-time market valuation.

You do NOT have access to live marketplace prices unless they were
explicitly provided in the input.

Do not claim that the estimated range represents the current market
price.

The estimate must be treated only as an AI-based indication based on
the available vehicle information.

Use exactly one of these values for "valutazione":

"good_deal"
"fair"
"high"
"uncertain"

Meaning:

"good_deal":
The asking price appears relatively attractive based on the available
information.

"fair":
The asking price appears broadly reasonable based on the available
information.

"high":
The asking price appears relatively high based on the available
information.

"uncertain":
There is not enough reliable information to make a meaningful price
assessment.

For "prezzo_min" and "prezzo_max":

- Return estimated integer amounts in EUR.
- prezzo_min must not be greater than prezzo_max.
- Use null for both values if a meaningful range cannot be estimated.
- Do not invent false precision.

The "spiegazione" must clearly state the main factors affecting the
assessment and MUST be written in {language}.




AUTO SCORE:

Calculate an Auto Score from 0 to 100.

The score represents how reassuring the AVAILABLE INFORMATION is
for a potential used-car buyer.

Use this interpretation:

90-100:
Very reassuring based on the available information.

75-89:
Generally reassuring, with normal used-car checks still recommended.

60-74:
Some important points should be checked before purchase.

40-59:
Several important points require careful verification.

0-39:
Significant concerns require particularly careful verification.

When calculating the score consider:

- vehicle age
- mileage
- approximate annual mileage
- maintenance information
- accident information
- TÜV/HU information
- tuning and modifications
- engine and drivetrain considerations
- legality/documentation of modifications
- possible future maintenance costs

IMPORTANT AUTO SCORE RULES:

The score is NOT a diagnosis.

The score is NOT a guarantee of the actual condition of the vehicle.

Do not automatically give a low score merely because a vehicle is old.

Do not automatically give a low score merely because mileage is high.

Do not automatically penalize modifications if they appear properly
documented and legal.

However, significant tuning or emissions-related modifications may
increase the importance of technical and legal verification.

Do not assume defects that are not stated in the listing.

Missing information may reduce confidence moderately, but must not
be treated as proof of a defect.

The score must reflect the information currently available.

The "punteggio" value MUST be an integer between 0 and 100.

The "spiegazione" must briefly explain the most important reasons
behind the score.

The "spiegazione" MUST be written in {language}.


RISK CHECK:

For each risk category return one of these exact values:

"low"
"medium"
"high"

The risk level represents how important it is to verify that area
before purchase.

It is NOT a diagnosis of the actual condition of the vehicle.

Use these principles:

- "low":
  the available information does not show an obvious reason
  for special attention beyond normal used-car checks.

- "medium":
  there are elements or missing information that should
  be specifically verified before purchase.

- "high":
  the listing contains significant elements that deserve
  particularly careful verification before purchase.

Do not claim that a component is defective merely because the vehicle
has high mileage, tuning or modifications.

If information is missing, explain that it should be verified instead
of inventing facts.

Keep every risk explanation concise and practical.


Return ONLY valid JSON.

Do not use Markdown.

Do not use ```json code fences.

Do not add any text before or after the JSON.


Use exactly this JSON structure:

{{
    "riassunto": "short analysis",

    "price_check": {{
        "valutazione": "fair",
        "prezzo_min": 0,
        "prezzo_max": 0,
        "spiegazione": "short explanation of the indicative price assessment"
    }},

    "auto_score": {{
        "punteggio": 0,
        "spiegazione": "short explanation of the score"
    }},

    "rischi": {{

        "motore_tuning": {{
            "livello": "low",
            "motivo": "short explanation"
        }},

        "legalita_tuv": {{
            "livello": "medium",
            "motivo": "short explanation"
        }},

        "manutenzione": {{
            "livello": "medium",
            "motivo": "short explanation"
        }},

        "incidenti_carrozzeria": {{
            "livello": "low",
            "motivo": "short explanation"
        }},

        "costi": {{
            "livello": "medium",
            "motivo": "short explanation"
        }}
    }},

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

Maximum 6 items per list.
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            reasoning={"effort": "low"},
            input=prompt
        )

        testo = response.output_text.strip()

        if testo.startswith("```json"):
            testo = testo[7:]

        elif testo.startswith("```"):
            testo = testo[3:]

        if testo.endswith("```"):
            testo = testo[:-3]

        testo = testo.strip()

        dati_ai = json.loads(testo)

        if not isinstance(dati_ai, dict):
            raise ValueError(
                "AI response is not a JSON object."
            )

        # -----------------------------
        # PRICE CHECK
        # -----------------------------

        price_check = dati_ai.get("price_check")

        if not isinstance(price_check, dict):

            dati_ai["price_check"] = {
                "valutazione": "uncertain",
                "prezzo_min": None,
                "prezzo_max": None,
                "spiegazione": ""
            }

        else:

            valutazione = str(
                price_check.get(
                    "valutazione",
                    "uncertain"
                )
            ).lower()

            if valutazione not in [
                "good_deal",
                "fair",
                "high",
                "uncertain"
            ]:
                valutazione = "uncertain"

            price_check["valutazione"] = valutazione


            try:
                prezzo_min = price_check.get("prezzo_min")

                if prezzo_min is not None:
                    prezzo_min = int(
                        round(float(prezzo_min))
                    )

                    if prezzo_min < 0:
                        prezzo_min = None

            except (TypeError, ValueError):
                prezzo_min = None


            try:
                prezzo_max = price_check.get("prezzo_max")

                if prezzo_max is not None:
                    prezzo_max = int(
                        round(float(prezzo_max))
                    )

                    if prezzo_max < 0:
                        prezzo_max = None

            except (TypeError, ValueError):
                prezzo_max = None


            # Se la fascia è invertita, la correggiamo.
            if (
                prezzo_min is not None
                and prezzo_max is not None
                and prezzo_min > prezzo_max
            ):
                prezzo_min, prezzo_max = (
                    prezzo_max,
                    prezzo_min
                )


            # Se manca uno dei due valori,
            # consideriamo la fascia non disponibile.
            if (
                prezzo_min is None
                or prezzo_max is None
            ):
                prezzo_min = None
                prezzo_max = None


            price_check["prezzo_min"] = prezzo_min
            price_check["prezzo_max"] = prezzo_max


            if not isinstance(
                price_check.get("spiegazione"),
                str
            ):
                price_check["spiegazione"] = ""






        # -----------------------------
        # AUTO SCORE
        # -----------------------------

        auto_score = dati_ai.get("auto_score")

        if not isinstance(auto_score, dict):

            dati_ai["auto_score"] = {
                "punteggio": None,
                "spiegazione": ""
            }

        else:

            punteggio = auto_score.get("punteggio")

            try:

                punteggio = int(punteggio)

                punteggio = max(
                    0,
                    min(100, punteggio)
                )

            except (TypeError, ValueError):

                punteggio = None

            auto_score["punteggio"] = punteggio

            if not isinstance(
                auto_score.get("spiegazione"),
                str
            ):
                auto_score["spiegazione"] = ""


        # -----------------------------
        # RISCHI
        # -----------------------------

        if not isinstance(
            dati_ai.get("rischi"),
            dict
        ):
            dati_ai["rischi"] = {}


        # -----------------------------
        # LISTE
        # -----------------------------

        for key in [
            "punti_controllo",
            "problemi_modello",
            "test_drive",
            "domande_venditore"
        ]:

            if not isinstance(
                dati_ai.get(key),
                list
            ):
                dati_ai[key] = []


        dati_ai["disponibile"] = True

        return dati_ai


    except Exception as e:

        print(
            "AI analysis error:",
            repr(e)
        )

        return {
            "disponibile": False,
            "errore":
                "AI analysis temporarily unavailable.",
            "auto_score": {
                "punteggio": None,
                "spiegazione": ""
            },
            "rischi": {}
        }


# =========================================================
# ENDPOINT ANALISI AUTO
# =========================================================

@app.post("/analyze")
def analyze_auto(auto: AutoData, request: Request):

    if auto.lingua not in ["de", "en", "it", "es"]:
        auto.lingua = "de"

    local = get_local_text(auto.lingua)

    anno_corrente = datetime.now().year
    eta = max(anno_corrente - auto.anno, 0)

    if eta > 0:
        km_annui = auto.km / eta
    else:
        km_annui = auto.km


    # --------------------------------
    # VALUTAZIONE CHILOMETRAGGIO
    # --------------------------------

    if km_annui < 8000:

        valutazione_km = local["km_low"]

    elif km_annui <= 20000:

        valutazione_km = local["km_normal"]

    else:

        valutazione_km = local["km_high"]


    # --------------------------------
    # PREZZO PER KM
    # --------------------------------

    prezzo_per_km = (
        auto.prezzo / auto.km
        if auto.km > 0
        else 0
    )


    # --------------------------------
    # FINANZIAMENTO
    # --------------------------------

    capitale = max(
        auto.prezzo - auto.anticipo,
        0
    )

    durata = max(
        auto.durata_mesi,
        1
    )

    tasso_mensile = (
        auto.tasso_annuo / 100
    ) / 12


    if capitale <= 0:

        rata_mensile = 0

    elif tasso_mensile == 0:

        rata_mensile = (
            capitale / durata
        )

    else:

        rata_mensile = (
            capitale
            * tasso_mensile
            * (1 + tasso_mensile) ** durata
        ) / (
            (1 + tasso_mensile) ** durata
            - 1
        )


    totale_rate = (
        rata_mensile * durata
    )

    interessi_totali = max(
        totale_rate - capitale,
        0
    )

    costo_totale_con_anticipo = (
        totale_rate
        + auto.anticipo
    )


    # --------------------------------
    # ANALISI AI
    # --------------------------------

    _check_daily_limit(
        request,
        "analyze",
        DAILY_ANALYSIS_LIMIT
    )

    analisi_ai = genera_analisi_ai(auto)


    if analisi_ai.get("disponibile"):

        checklist = analisi_ai.get(
            "punti_controllo",
            local["checklist"]
        )

        domande_venditore = analisi_ai.get(
            "domande_venditore",
            local["fallback_questions"]
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


        # --------------------------------
        # AUTO SCORE
        # --------------------------------

        auto_score = analisi_ai.get(
            "auto_score",
            {
                "punteggio": None,
                "spiegazione": ""
            }
        )


        # --------------------------------
        # RISCHI
        # --------------------------------

        rischi = analisi_ai.get(
            "rischi",
            {}
        )


        # --------------------------------
        # CONTROLLI SPECIFICI
        # --------------------------------

        controlli_specifici = []

        controlli_specifici.extend(
            checklist
        )


        for item in problemi_modello:

            if item not in controlli_specifici:

                controlli_specifici.append(
                    item
                )


        for item in test_drive:

            if item not in controlli_specifici:

                controlli_specifici.append(
                    item
                )


        controlli_specifici = (
            controlli_specifici[:8]
        )


    else:

        checklist = local[
            "checklist"
        ]

        domande_venditore = local[
            "fallback_questions"
        ]

        controlli_specifici = local[
            "fallback_checks"
        ]

        problemi_modello = []

        test_drive = []

        riassunto_ai = ""


        auto_score = {
            "punteggio": None,
            "spiegazione": ""
        }


        rischi = {}


    # --------------------------------
    # RISPOSTA API
    # --------------------------------

    return {

        "auto":
            f"{auto.marca} {auto.modello}",

        "lingua":
            auto.lingua,

        "anno":
            auto.anno,

        "eta_anni":
            eta,

        "prezzo":
            round(
                auto.prezzo,
                2
            ),

        "km":
            auto.km,

        "potenza_cv":
            auto.potenza_cv,

        "prezzo_per_km":
            round(
                prezzo_per_km,
                4
            ),


        # --------------------------------
        # FINANZIAMENTO
        # --------------------------------

        "finanziamento": {

            "capitale_finanziato":
                round(
                    capitale,
                    2
                ),

            "rata_mensile":
                round(
                    rata_mensile,
                    2
                ),

            "totale_rate":
                round(
                    totale_rate,
                    2
                ),

            "interessi_totali":
                round(
                    interessi_totali,
                    2
                ),

            "costo_totale_con_anticipo":
                round(
                    costo_totale_con_anticipo,
                    2
                )
        },


        # --------------------------------
        # ANALISI
        # --------------------------------

        "analisi": {

            "km_annui":
                round(
                    km_annui
                ),

            "valutazione_km":
                valutazione_km,

            "checklist":
                checklist,

            "domande_venditore":
                domande_venditore,

            "controlli_specifici":
                controlli_specifici,

            "problemi_modello":
                problemi_modello,

            "test_drive":
                test_drive,

            "riassunto_ai":
                riassunto_ai,


            # AUTO SCORE

            "auto_score":
                auto_score,


            # RISCHI

            "rischi":
                rischi
        },


        # --------------------------------
        # RISPOSTA AI COMPLETA
        # --------------------------------

        "ai":
            analisi_ai
    }


# =========================================================
# REPORT PDF
# =========================================================

def _pdf_text(value):
    if value is None or value == "":
        return "-"
    return escape(str(value))


def _pdf_money(value, lang="de"):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"

    formatted = f"{number:,.2f}"
    if lang in ["de", "it", "es"]:
        formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted + " EUR"


def _pdf_labels(lang):
    labels = {
        "de": {
            "report": "Fahrzeuganalyse - PDF Report",
            "generated": "Erstellt am",
            "vehicle": "Fahrzeugdaten",
            "finance": "Finanzierung",
            "score": "Auto Score",
            "summary": "AI-Zusammenfassung",
            "risk": "AI-Risikocheck",
            "checks": "Checkliste vor dem Kauf",
            "specific": "Wichtige Pruefpunkte",
            "questions": "Fragen an den Verkaeufer",
            "make": "Marke", "model": "Modell", "year": "Baujahr",
            "price": "Kaufpreis", "km": "Kilometerstand", "power": "Leistung",
            "fuel": "Kraftstoff", "gearbox": "Getriebe", "tuv": "TUEV / HU",
            "accident": "Unfallangaben", "mods": "Aenderungen / Tuning",
            "deposit": "Anzahlung", "months": "Laufzeit", "interest": "Jahreszins",
            "monthly": "Monatliche Rate", "financed": "Finanzierter Betrag",
            "total_interest": "Zinsen gesamt", "payments": "Summe der Raten",
            "total_cost": "Gesamtkosten", "years": "Jahre", "months_unit": "Monate",
            "low": "NIEDRIG", "medium": "MITTEL", "high": "HOCH",
            "motore_tuning": "Motor & Tuning", "legalita_tuv": "Legalitaet & TUEV",
            "manutenzione": "Wartung", "incidenti_carrozzeria": "Unfaelle & Karosserie",
            "costi": "Kosten",
            "disclaimer": "Hinweis: Dieser Report basiert auf den verfuegbaren Angaben und ersetzt keine professionelle technische Fahrzeugpruefung oder ein verbindliches Finanzierungsangebot."
        },
        "en": {
            "report": "Vehicle Analysis - PDF Report",
            "generated": "Generated on",
            "vehicle": "Vehicle details",
            "finance": "Financing",
            "score": "Auto Score",
            "summary": "AI summary",
            "risk": "AI Risk Check",
            "checks": "Pre-purchase checklist",
            "specific": "Important points to check",
            "questions": "Questions for the seller",
            "make": "Make", "model": "Model", "year": "Year",
            "price": "Purchase price", "km": "Mileage", "power": "Power",
            "fuel": "Fuel", "gearbox": "Transmission", "tuv": "Roadworthiness / TUV",
            "accident": "Accident information", "mods": "Modifications / tuning",
            "deposit": "Down payment", "months": "Term", "interest": "Annual interest",
            "monthly": "Monthly payment", "financed": "Amount financed",
            "total_interest": "Total interest", "payments": "Total payments",
            "total_cost": "Total cost", "years": "years", "months_unit": "months",
            "low": "LOW", "medium": "MEDIUM", "high": "HIGH",
            "motore_tuning": "Engine & Tuning", "legalita_tuv": "Legality & Inspection",
            "manutenzione": "Maintenance", "incidenti_carrozzeria": "Accidents & Bodywork",
            "costi": "Costs",
            "disclaimer": "Note: This report is based on the available information and does not replace a professional vehicle inspection or a binding financing offer."
        },
        "it": {
            "report": "Analisi auto - Report PDF",
            "generated": "Generato il",
            "vehicle": "Dati dell'auto",
            "finance": "Finanziamento",
            "score": "Auto Score",
            "summary": "Riepilogo AI",
            "risk": "AI Risk Check",
            "checks": "Checklist prima dell'acquisto",
            "specific": "Punti importanti da controllare",
            "questions": "Domande da fare al venditore",
            "make": "Marca", "model": "Modello", "year": "Anno",
            "price": "Prezzo", "km": "Chilometri", "power": "Potenza",
            "fuel": "Carburante", "gearbox": "Cambio", "tuv": "TUV / Revisione",
            "accident": "Informazioni incidenti", "mods": "Modifiche / tuning",
            "deposit": "Anticipo", "months": "Durata", "interest": "Tasso annuo",
            "monthly": "Rata mensile", "financed": "Capitale finanziato",
            "total_interest": "Interessi totali", "payments": "Totale rate",
            "total_cost": "Costo totale", "years": "anni", "months_unit": "mesi",
            "low": "BASSO", "medium": "MEDIO", "high": "ALTO",
            "motore_tuning": "Motore & Tuning", "legalita_tuv": "Legalita & Revisione",
            "manutenzione": "Manutenzione", "incidenti_carrozzeria": "Incidenti & Carrozzeria",
            "costi": "Costi",
            "disclaimer": "Nota: questo report si basa sulle informazioni disponibili e non sostituisce un controllo tecnico professionale dell'auto o un'offerta finanziaria vincolante."
        },
        "es": {
            "report": "Análisis del vehículo - Informe PDF",
            "generated": "Generado el",
            "vehicle": "Datos del vehículo",
            "finance": "Financiación",
            "score": "Auto Score",
            "summary": "Resumen de IA",
            "risk": "Análisis de riesgos con IA",
            "checks": "Lista de comprobación antes de comprar",
            "specific": "Puntos importantes a comprobar",
            "questions": "Preguntas para el vendedor",
            "make": "Marca", "model": "Modelo", "year": "Año",
            "price": "Precio de compra", "km": "Kilometraje", "power": "Potencia",
            "fuel": "Combustible", "gearbox": "Transmisión", "tuv": "ITV / TÜV",
            "accident": "Información sobre accidentes", "mods": "Modificaciones / tuning",
            "deposit": "Entrada", "months": "Plazo", "interest": "Interés anual",
            "monthly": "Cuota mensual", "financed": "Importe financiado",
            "total_interest": "Intereses totales", "payments": "Total de cuotas",
            "total_cost": "Coste total", "years": "años", "months_unit": "meses",
            "low": "BAJO", "medium": "MEDIO", "high": "ALTO",
            "motore_tuning": "Motor y tuning", "legalita_tuv": "Legalidad e ITV",
            "manutenzione": "Mantenimiento", "incidenti_carrozzeria": "Accidentes y carrocería",
            "costi": "Costes",
            "disclaimer": "Nota: este informe se basa en la información disponible y no sustituye una inspección técnica profesional del vehículo ni una oferta de financiación vinculante."
        }
    }
    return labels.get(lang, labels["de"])


def _pdf_section_title(text, styles):
    return Paragraph(_pdf_text(text), styles["SectionTitle"])


def _pdf_bullets(items, styles):
    flowables = []
    if not isinstance(items, list) or not items:
        flowables.append(Paragraph("-", styles["BodyTextCustom"]))
        return flowables
    for item in items:
        flowables.append(Paragraph("- " + _pdf_text(item), styles["BodyTextCustom"]))
        flowables.append(Spacer(1, 2 * mm))
    return flowables


def _pdf_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(18 * mm, 10 * mm, "AutoAnalyzer AI")
    canvas.drawRightString(192 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf_report(payload: PDFReportRequest):
    auto = payload.auto
    result = payload.result if isinstance(payload.result, dict) else {}
    lang = auto.lingua if auto.lingua in ["de", "en", "it", "es"] else "de"
    t = _pdf_labels(lang)

    analysis = result.get("analisi", {}) if isinstance(result.get("analisi"), dict) else {}
    finance = result.get("finanziamento", {}) if isinstance(result.get("finanziamento"), dict) else {}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"AutoAnalyzer AI - {auto.marca} {auto.modello}",
        author="AutoAnalyzer AI"
    )

    base = getSampleStyleSheet()
    styles = {
        "TitleCustom": ParagraphStyle(
            "TitleCustom", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, textColor=colors.HexColor("#111827"),
            alignment=TA_CENTER, spaceAfter=4 * mm
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, leading=14, textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER, spaceAfter=8 * mm
        ),
        "SectionTitle": ParagraphStyle(
            "SectionTitle", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=14, leading=18, textColor=colors.HexColor("#111827"),
            spaceBefore=5 * mm, spaceAfter=3 * mm
        ),
        "BodyTextCustom": ParagraphStyle(
            "BodyTextCustom", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.5, leading=14, textColor=colors.HexColor("#374151")
        ),
        "Small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8, leading=11, textColor=colors.HexColor("#6b7280")
        ),
        "Score": ParagraphStyle(
            "Score", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=30, leading=34, alignment=TA_CENTER,
            textColor=colors.HexColor("#2563eb")
        )
    }

    story = []
    story.append(Paragraph("AutoAnalyzer AI", styles["TitleCustom"]))
    story.append(Paragraph(_pdf_text(t["report"]), styles["Subtitle"]))
    story.append(Paragraph(
        _pdf_text(f"{auto.marca} {auto.modello}"),
        ParagraphStyle("CarName", parent=styles["TitleCustom"], fontSize=17, leading=21)
    ))
    story.append(Paragraph(
        f'{_pdf_text(t["generated"])}: {datetime.now().strftime("%d.%m.%Y %H:%M")}',
        styles["Subtitle"]
    ))

    story.append(_pdf_section_title(t["vehicle"], styles))
    vehicle_rows = [
        [t["make"], _pdf_text(auto.marca), t["model"], _pdf_text(auto.modello)],
        [t["year"], _pdf_text(auto.anno), t["price"], _pdf_money(auto.prezzo, lang)],
        [t["km"], f"{_pdf_text(auto.km)} km", t["power"], f"{_pdf_text(auto.potenza_cv)} PS/HP"],
        [t["fuel"], _pdf_text(auto.carburante), t["gearbox"], _pdf_text(auto.cambio)],
        [t["tuv"], _pdf_text(auto.tuv), t["accident"], _pdf_text(auto.incidenti)]
    ]
    vehicle_table = Table(vehicle_rows, colWidths=[31 * mm, 50 * mm, 31 * mm, 50 * mm])
    vehicle_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))
    story.append(vehicle_table)

    if auto.modifiche:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(f'<b>{_pdf_text(t["mods"])}:</b>', styles["BodyTextCustom"]))
        story.extend(_pdf_bullets(auto.modifiche, styles))

    story.append(_pdf_section_title(t["finance"], styles))
    finance_rows = [
        [t["deposit"], _pdf_money(auto.anticipo, lang), t["months"], f"{auto.durata_mesi} {t['months_unit']}"],
        [t["interest"], f"{auto.tasso_annuo:.2f}%", t["monthly"], _pdf_money(finance.get("rata_mensile"), lang)],
        [t["financed"], _pdf_money(finance.get("capitale_finanziato"), lang), t["total_interest"], _pdf_money(finance.get("interessi_totali"), lang)],
        [t["payments"], _pdf_money(finance.get("totale_rate"), lang), t["total_cost"], _pdf_money(finance.get("costo_totale_con_anticipo"), lang)]
    ]
    finance_table = Table(finance_rows, colWidths=[31 * mm, 50 * mm, 31 * mm, 50 * mm])
    finance_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))
    story.append(finance_table)

    score = analysis.get("auto_score", {}) if isinstance(analysis.get("auto_score"), dict) else {}
    score_value = score.get("punteggio")
    if score_value is not None:
        story.append(_pdf_section_title(t["score"], styles))
        story.append(Paragraph(f"{_pdf_text(score_value)} / 100", styles["Score"]))
        if score.get("spiegazione"):
            story.append(Paragraph(_pdf_text(score.get("spiegazione")), styles["BodyTextCustom"]))

    if analysis.get("riassunto_ai"):
        story.append(_pdf_section_title(t["summary"], styles))
        story.append(Paragraph(_pdf_text(analysis.get("riassunto_ai")), styles["BodyTextCustom"]))

    risks = analysis.get("rischi", {}) if isinstance(analysis.get("rischi"), dict) else {}
    if risks:
        story.append(_pdf_section_title(t["risk"], styles))
        risk_rows = []
        for key in ["motore_tuning", "legalita_tuv", "manutenzione", "incidenti_carrozzeria", "costi"]:
            risk = risks.get(key)
            if not isinstance(risk, dict):
                continue
            level = str(risk.get("livello", "medium")).lower()
            if level not in ["low", "medium", "high"]:
                level = "medium"
            risk_rows.append([
                Paragraph(f'<b>{_pdf_text(t.get(key, key))}</b>', styles["BodyTextCustom"]),
                Paragraph(f'<b>{_pdf_text(t[level])}</b>', styles["BodyTextCustom"]),
                Paragraph(_pdf_text(risk.get("motivo", "-")), styles["BodyTextCustom"])
            ])
        if risk_rows:
            risk_table = Table(risk_rows, colWidths=[38 * mm, 24 * mm, 100 * mm], repeatRows=0)
            risk_table.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6)
            ]))
            story.append(risk_table)

    story.append(_pdf_section_title(t["checks"], styles))
    story.extend(_pdf_bullets(analysis.get("checklist", []), styles))

    story.append(_pdf_section_title(t["specific"], styles))
    story.extend(_pdf_bullets(analysis.get("controlli_specifici", []), styles))

    story.append(_pdf_section_title(t["questions"], styles))
    story.extend(_pdf_bullets(analysis.get("domande_venditore", []), styles))

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(_pdf_text(t["disclaimer"]), styles["Small"]))

    doc.build(story, onFirstPage=_pdf_page_number, onLaterPages=_pdf_page_number)
    buffer.seek(0)
    return buffer


@app.post("/generate-report")
def generate_report(payload: PDFReportRequest):
    try:
        pdf_buffer = build_pdf_report(payload)

        safe_make = "".join(c for c in payload.auto.marca if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
        safe_model = "".join(c for c in payload.auto.modello if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
        filename = f"AutoAnalyzer_{safe_make}_{safe_model}.pdf"[:120]

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        print("PDF report error:", repr(e))
        return {
            "success": False,
            "errore": "PDF report generation failed."
        }


# =========================================================
# ESTRAZIONE DATI ANNUNCIO CON AI
# =========================================================

def estrai_dati_annuncio_ai(titolo, testo, lingua="de"):

    if client is None:
        return {
            "success": False,
            "errore": "OpenAI API not configured."
        }

    language = get_language_name(lingua)

    prompt = f"""
You extract structured vehicle information from a used-car listing.

LISTING TITLE:

{titolo}

LISTING TEXT:

{testo}

Extract only information that is actually present or clearly stated
in the listing.

Important extraction rules:

- Price must be the actual vehicle price in EUR.
- Do not confuse monthly financing payments with the vehicle price.
- Mileage must be an integer representing kilometres.
- Power must be PS/HP, not kW.
- Do not confuse engine displacement, wheel size or other numbers
  with power, mileage, year or price.
- Year should represent first registration or manufacturing year
  when clearly available.
- If information is unavailable, return null.
- Keep the vehicle model concise.
- Detect tuning such as Stage 1, Stage 2 or Stage 3.
- Detect relevant technical or cosmetic modifications.
- Detect exhaust, downpipe, suspension, wheels and emissions changes.
- Accident information must only be returned if explicitly stated.
- Extract TÜV/HU information if available.
- Extract transmission and fuel type if available.

IMPORTANT LANGUAGE INSTRUCTION:

The selected language is {language}.

ALL user-facing textual values MUST be written in {language}.

This applies to:

- "carburante"
- "cambio"
- "tuv"
- "incidenti"
- every item in "modifiche"
- every item in "informazioni_extra"

Translate textual values when necessary.

If the selected language is German, use German terms.

Examples:
- manual transmission -> "Manuell"
- automatic transmission -> "Automatik"
- petrol -> "Benzin"
- diesel -> "Diesel"
- August 2027 -> "August 2027"
- accident-free -> "Unfallfrei"

If the selected language is English, use English terms.

Examples:
- manual transmission -> "Manual"
- automatic transmission -> "Automatic"
- petrol -> "Petrol"
- diesel -> "Diesel"
- August 2027 -> "August 2027"
- accident-free -> "Accident-free"

If the selected language is Italian, use Italian terms.

Examples:
- manual transmission -> "Manuale"
- automatic transmission -> "Automatico"
- petrol -> "Benzina"
- diesel -> "Diesel"
- August 2027 -> "Agosto 2027"
- accident-free -> "Senza incidenti"

If the selected language is Spanish, use Spanish terms.

Examples:
- manual transmission -> "Manual"
- automatic transmission -> "Automático"
- petrol -> "Gasolina"
- diesel -> "Diésel"
- August 2027 -> "Agosto 2027"
- accident-free -> "Sin accidentes"

Do NOT translate:

- vehicle manufacturer names
- vehicle model names
- brand names
- product names
- tuning company names

For example:
"Sachs Sportkupplung" may be described in the selected language,
but "Sachs" must remain unchanged.

Return ONLY valid JSON.

Do not use Markdown.
Do not use code fences.
Do not add text before or after the JSON.

Use exactly this structure:

{{
    "marca": null,
    "modello": null,
    "anno": null,
    "prezzo": null,
    "km": null,
    "potenza_cv": null,
    "carburante": null,
    "cambio": null,
    "tuv": null,
    "incidenti": null,
    "modifiche": [],
    "informazioni_extra": []
}}
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            reasoning={"effort": "low"},
            input=prompt
        )

        testo_ai = response.output_text.strip()

        if testo_ai.startswith("```json"):
            testo_ai = testo_ai[7:]

        elif testo_ai.startswith("```"):
            testo_ai = testo_ai[3:]

        if testo_ai.endswith("```"):
            testo_ai = testo_ai[:-3]

        testo_ai = testo_ai.strip()

        dati = json.loads(testo_ai)

        if not isinstance(dati, dict):
            raise ValueError("Extraction response is not an object.")

        if not isinstance(dati.get("modifiche"), list):
            dati["modifiche"] = []

        if not isinstance(dati.get("informazioni_extra"), list):
            dati["informazioni_extra"] = []

        return {
            "success": True,
            "dati": dati
        }

    except Exception as e:

        print("Listing extraction AI error:", repr(e))

        return {
            "success": False,
            "errore": "AI extraction failed."
        }


# =========================================================
# SICUREZZA URL
# =========================================================

def url_pubblico_valido(url):

    try:

        parsed = urlparse(url)

        if parsed.scheme not in ["http", "https"]:
            return False, "URL non valido."

        hostname = parsed.hostname

        if not hostname:
            return False, "Hostname non valido."

        if hostname.lower() == "localhost":
            return False, "URL locale non consentito."

        infos = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80)
        )

        for info in infos:

            ip_string = info[4][0]

            ip = ipaddress.ip_address(ip_string)

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False, "Indirizzo non consentito."

        return True, None

    except Exception:

        return False, "Impossibile verificare l'URL."


# =========================================================
# IMPORTAZIONE ANNUNCIO
# =========================================================

@app.post("/extract-listing")
def extract_listing(data: ListingURL, request: Request):

    url = data.url.strip()

    if data.lingua not in ["de", "en", "it", "es"]:
        data.lingua = "de"

    valido, errore = url_pubblico_valido(url)

    if not valido:
        return {
            "success": False,
            "errore": errore
        }

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36",

        "Accept":
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",

        "Accept-Language":
            "de-DE,de;q=0.9,en;q=0.8"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10,
            allow_redirects=False
        )

        # Segui al massimo un redirect
        if response.status_code in [301, 302, 303, 307, 308]:

            location = response.headers.get("Location")

            if not location:
                return {
                    "success": False,
                    "errore": "Redirect non valido."
                }

            redirect_url = urljoin(url, location)

            valido, errore = url_pubblico_valido(redirect_url)

            if not valido:
                return {
                    "success": False,
                    "errore": errore
                }

            response = requests.get(
                redirect_url,
                headers=headers,
                timeout=10,
                allow_redirects=False
            )

            url = redirect_url

        # Gestione specifica del blocco mobile.de
        if response.status_code == 403:

            return {
                "success": False,
                "errore": (
                    "mobile.de blocca l'importazione automatica "
                    "di questo annuncio. "
                    "Prova con un altro sito oppure inserisci "
                    "i dati dell'auto manualmente."
                )
            }

        if response.status_code != 200:

            return {
                "success": False,
                "errore":
                    f"Il sito ha risposto con codice {response.status_code}."
            }

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):

            return {
                "success": False,
                "errore": "La pagina non contiene HTML."
            }

        max_size = 5 * 1024 * 1024

        content_length = response.headers.get("Content-Length")

        if content_length:

            try:

                if int(content_length) > max_size:

                    return {
                        "success": False,
                        "errore": "Pagina troppo grande."
                    }

            except ValueError:
                pass

        if len(response.content) > max_size:

            return {
                "success": False,
                "errore": "Pagina troppo grande."
            }

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for element in soup([
            "script",
            "style",
            "noscript",
            "svg"
        ]):
            element.decompose()

        if soup.title:
            titolo = soup.title.get_text(
                " ",
                strip=True
            )
        else:
            titolo = ""

        testo = soup.get_text(
            " ",
            strip=True
        )

        testo = testo[:15000]

        if not testo:

            return {
                "success": False,
                "errore":
                    "Non è stato possibile leggere "
                    "il contenuto dell'annuncio."
            }

        _check_daily_limit(
            request,
            "extract",
            DAILY_IMPORT_LIMIT
        )

        estrazione_ai = estrai_dati_annuncio_ai(
            titolo,
            testo,
            data.lingua
        )

        if not estrazione_ai.get("success"):

            return {
                "success": False,
                "errore": estrazione_ai.get(
                    "errore",
                    "Errore durante l'estrazione."
                )
            }

        return {

            "success": True,

            "url": url,

            "titolo": titolo,

            "dati_estratti":
                estrazione_ai["dati"]
        }

    except requests.Timeout:

        return {
            "success": False,
            "errore":
                "Timeout durante il caricamento dell'annuncio."
        }

    except requests.RequestException as e:

        print(
            "Listing request error:",
            repr(e)
        )

        return {
            "success": False,
            "errore":
                "Impossibile caricare l'annuncio."
        }

    except Exception as e:

        print(
            "Listing import error:",
            repr(e)
        )

        return {
            "success": False,
            "errore":
                "Errore durante l'importazione dell'annuncio."
        }