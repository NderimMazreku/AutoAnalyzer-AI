import os
import json
import socket
import ipaddress
from datetime import datetime
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================================================
# CONFIGURAZIONE
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(title="AutoAnalyzer AI")


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

Every user-facing sentence in your response MUST be written in {language}.
Do not use Italian unless the selected language is Italian.

Your task is to help the buyer understand what should be checked
before purchasing this vehicle.

Do not claim that this specific vehicle has a defect unless that
information was explicitly provided.

When mentioning known or commonly reported model-specific issues,
describe them only as potential points worth checking.

If modifications or tuning are provided, specifically explain what
should be checked because of those modifications.

If Stage 1, Stage 2 or Stage 3 tuning is present, consider appropriate
checks concerning the engine, turbocharger, drivetrain,
clutch/transmission and emissions equipment.

If modifications are listed, consider whether documentation,
approval, registration, ABE or technical inspection may need to be
verified where relevant.

Use TÜV/HU, accident information, fuel type and transmission
information when generating checks and seller questions.

RISK CHECK:

For each risk category return one of these exact values:

"low"
"medium"
"high"

The risk level represents how important it is to verify that area
before purchase. It is NOT a diagnosis of the actual condition
of the vehicle.

Use these principles:

- "low": the available information does not show an obvious reason
  for special attention beyond normal used-car checks.

- "medium": there are elements or missing information that should
  be specifically verified before purchase.

- "high": the listing contains significant elements that deserve
  particularly careful verification before purchase.

Do not claim that a component is defective merely because the vehicle
has high mileage, tuning or modifications.

If information is missing, explain that it should be verified instead
of inventing facts.

Keep every risk explanation concise and practical.

Return ONLY valid JSON.
Do not use Markdown.
Do not use ```json code fences.

Use exactly this JSON structure:

{{
    "riassunto": "short analysis",

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

        # Strutture di sicurezza
        if not isinstance(dati_ai, dict):
            raise ValueError("AI response is not a JSON object.")

        if not isinstance(dati_ai.get("rischi"), dict):
            dati_ai["rischi"] = {}

        for key in [
            "punti_controllo",
            "problemi_modello",
            "test_drive",
            "domande_venditore"
        ]:
            if not isinstance(dati_ai.get(key), list):
                dati_ai[key] = []

        dati_ai["disponibile"] = True

        return dati_ai

    except Exception as e:

        print("AI analysis error:", repr(e))

        return {
            "disponibile": False,
            "errore": "AI analysis temporarily unavailable.",
            "rischi": {}
        }


# =========================================================
# ENDPOINT ANALISI AUTO
# =========================================================

@app.post("/analyze")
def analyze_auto(auto: AutoData):

    if auto.lingua not in ["de", "en", "it"]:
        auto.lingua = "de"

    local = get_local_text(auto.lingua)

    anno_corrente = datetime.now().year

    eta = max(anno_corrente - auto.anno, 0)

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

    if auto.km > 0:
        prezzo_per_km = auto.prezzo / auto.km
    else:
        prezzo_per_km = 0

    # =====================================================
    # FINANZIAMENTO
    # =====================================================

    capitale = max(auto.prezzo - auto.anticipo, 0)

    durata = max(auto.durata_mesi, 1)

    tasso_mensile = (auto.tasso_annuo / 100) / 12

    if capitale <= 0:

        rata_mensile = 0

    elif tasso_mensile == 0:

        rata_mensile = capitale / durata

    else:

        rata_mensile = (
            capitale
            * tasso_mensile
            * (1 + tasso_mensile) ** durata
        ) / (
            (1 + tasso_mensile) ** durata - 1
        )

    totale_rate = rata_mensile * durata

    interessi_totali = max(totale_rate - capitale, 0)

    costo_totale_con_anticipo = totale_rate + auto.anticipo

    # =====================================================
    # AI
    # =====================================================

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

        rischi = analisi_ai.get(
            "rischi",
            {}
        )

        controlli_specifici = []

        controlli_specifici.extend(checklist)

        for item in problemi_modello:
            if item not in controlli_specifici:
                controlli_specifici.append(item)

        for item in test_drive:
            if item not in controlli_specifici:
                controlli_specifici.append(item)

        controlli_specifici = controlli_specifici[:8]

    else:

        checklist = local["checklist"]

        domande_venditore = local["fallback_questions"]

        controlli_specifici = local["fallback_checks"]

        problemi_modello = []

        test_drive = []

        riassunto_ai = ""

        rischi = {}

    # =====================================================
    # RISPOSTA
    # =====================================================

    return {

        "auto": f"{auto.marca} {auto.modello}",

        "lingua": auto.lingua,

        "anno": auto.anno,

        "eta_anni": eta,

        "prezzo": round(auto.prezzo, 2),

        "km": auto.km,

        "potenza_cv": auto.potenza_cv,

        "prezzo_per_km": round(prezzo_per_km, 4),

        "finanziamento": {

            "capitale_finanziato": round(capitale, 2),

            "rata_mensile": round(rata_mensile, 2),

            "totale_rate": round(totale_rate, 2),

            "interessi_totali": round(interessi_totali, 2),

            "costo_totale_con_anticipo":
                round(costo_totale_con_anticipo, 2)
        },

        "analisi": {

            "km_annui": round(km_annui),

            "valutazione_km": valutazione_km,

            "checklist": checklist,

            "domande_venditore": domande_venditore,

            "controlli_specifici": controlli_specifici,

            "problemi_modello": problemi_modello,

            "test_drive": test_drive,

            "riassunto_ai": riassunto_ai,

            "rischi": rischi
        },

        "ai": analisi_ai
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

Important rules:

- Price must be the actual vehicle price in EUR.
- Do not confuse monthly financing payments with the vehicle price.
- Mileage must be an integer representing kilometres.
- Power must be PS/HP, not kW.
- Do not confuse engine displacement, wheel size or other numbers
  with power, mileage, year or price.
- Year should represent first registration or manufacturing year
  when clearly available.
- If information is unavailable, return null.
- Keep the model concise.
- Detect tuning such as Stage 1, Stage 2 or Stage 3.
- Detect relevant technical or cosmetic modifications.
- Detect exhaust, downpipe, suspension, wheels and emissions changes.
- Accident information must only be returned if explicitly stated.
- Extract TÜV/HU information if available.
- Extract transmission and fuel type if available.

The selected language is {language}.

Write the entries in "modifiche" and "informazioni_extra"
in {language}.

Return ONLY valid JSON.

Do not use Markdown or code fences.

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
def extract_listing(data: ListingURL):

    url = data.url.strip()

    if data.lingua not in ["de", "en", "it"]:
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
            titolo = soup.title.get_text(" ", strip=True)
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
                    "Non è stato possibile leggere il contenuto dell'annuncio."
            }

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
            "errore": "Timeout durante il caricamento dell'annuncio."
        }

    except requests.RequestException as e:

        print("Listing request error:", repr(e))

        return {
            "success": False,
            "errore": "Impossibile caricare l'annuncio."
        }

    except Exception as e:

        print("Listing import error:", repr(e))

        return {
            "success": False,
            "errore": "Errore durante l'importazione dell'annuncio."
        }