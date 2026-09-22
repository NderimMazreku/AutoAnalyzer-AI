import os
import json
import socket
import ipaddress
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel


# ==========================================
# CONFIGURAZIONE
# ==========================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(title="AutoAnalyzer AI")


# ==========================================
# MODELLI
# ==========================================

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


class ListingURL(BaseModel):
    url: str
    lingua: str = "de"


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return FileResponse("index.html")


# ==========================================
# LINGUE
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
# ANALISI AI DEL VEICOLO
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
# ANALISI PRINCIPALE
# ==========================================

@app.post("/analyze")
def analyze_auto(auto: AutoData):

    if auto.lingua not in ["de", "en", "it"]:
        auto.lingua = "de"

    local = get_local_text(auto.lingua)

    anno_corrente = datetime.now().year

    eta = max(
        anno_corrente - auto.anno,
        0
    )

    # KM annui

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
            * (1 + tasso_mensile) ** auto.durata_mesi
            /
            (
                (1 + tasso_mensile) ** auto.durata_mesi
                - 1
            )
        )

    totale_rate = (
        rata * auto.durata_mesi
    )

    interessi = (
        totale_rate - capitale
    )

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

        "ai":
            analisi_ai
    }


# ==========================================
# ESTRAZIONE DATI ANNUNCIO CON AI
# ==========================================

def estrai_dati_annuncio_ai(
    titolo,
    testo,
    lingua="de"
):

    if client is None:

        return {
            "success": False,
            "errore": "OpenAI API not configured."
        }

    if lingua not in ["de", "en", "it"]:
        lingua = "de"

    language = get_language_name(lingua)

    prompt = f"""
You extract structured vehicle information from a used-car listing.

LISTING TITLE:
{titolo}

LISTING TEXT:
{testo}

Extract ONLY information that is actually present or can be clearly
identified from the listing.

IMPORTANT RULES:

- Never invent missing information.
- Prices must be numbers only, expressed in EUR.
- Mileage must be an integer representing kilometers.
- Power must be PS/HP, not kW.
- Year should represent first registration or vehicle year when clearly identifiable.
- If a numeric value cannot be identified, return null.
- If a text value cannot be identified, return null.
- Keep the model name concise.
- Do not mistake financing payments for the vehicle purchase price.
- Do not mistake engine displacement for mileage.
- Do not mistake kW for PS.
- Detect modifications such as Stage 1, Stage 2, Stage 3, tuning,
  aftermarket exhaust, suspension modifications and aftermarket wheels.
- Detect accident information only when explicitly stated.
- Detect TÜV/HU information when explicitly present.
- Detect transmission when present.
- Detect fuel type when present.

Return ONLY valid JSON.

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

The content of "modifiche" and "informazioni_extra"
must be written in {language}.

JSON only.
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            reasoning={"effort": "low"},
            input=prompt
        )

        contenuto = response.output_text.strip()

        if contenuto.startswith("```"):

            contenuto = contenuto.replace(
                "```json",
                ""
            )

            contenuto = contenuto.replace(
                "```",
                ""
            )

            contenuto = contenuto.strip()

        dati = json.loads(contenuto)

        return {
            "success": True,
            "dati": dati
        }

    except Exception as e:

        print(
            "ERRORE ESTRAZIONE AI:",
            str(e)
        )

        return {
            "success": False,
            "errore": "AI extraction failed."
        }


# ==========================================
# CONTROLLO URL
# ==========================================

def url_pubblico_valido(url):

    try:

        parsed = urlparse(url)

        if parsed.scheme not in [
            "http",
            "https"
        ]:
            return False, "URL non valido."

        if not parsed.hostname:
            return False, "URL non valido."

        hostname = parsed.hostname.lower()

        if hostname in [
            "localhost",
            "localhost.localdomain"
        ]:
            return False, "Indirizzo non consentito."

        addresses = socket.getaddrinfo(
            hostname,
            None
        )

        for address in addresses:

            ip_string = address[4][0]

            ip = ipaddress.ip_address(
                ip_string
            )

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):

                return (
                    False,
                    "Indirizzo non consentito."
                )

        return True, None

    except Exception as e:

        print(
            "ERRORE CONTROLLO URL:",
            str(e)
        )

        return (
            False,
            "Impossibile verificare l'indirizzo."
        )


# ==========================================
# ESTRAZIONE ANNUNCIO DA URL
# ==========================================

@app.post("/extract-listing")
def extract_listing(data: ListingURL):

    url = data.url.strip()

    if data.lingua not in [
        "de",
        "en",
        "it"
    ]:
        data.lingua = "de"

    valido, errore = url_pubblico_valido(
        url
    )

    if not valido:

        return {
            "success": False,
            "errore": errore
        }

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36",

        "Accept":
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8",

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

        # Non seguiamo automaticamente redirect,
        # perché prima dobbiamo verificare il nuovo URL.

        if response.status_code in [
            301,
            302,
            303,
            307,
            308
        ]:

            redirect_url = response.headers.get(
                "Location"
            )

            if not redirect_url:

                return {
                    "success": False,
                    "errore":
                        "Redirect senza destinazione."
                }

            from urllib.parse import urljoin

            redirect_url = urljoin(
                url,
                redirect_url
            )

            valido_redirect, errore_redirect = (
                url_pubblico_valido(
                    redirect_url
                )
            )

            if not valido_redirect:

                return {
                    "success": False,
                    "errore":
                        errore_redirect
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
                    "La pagina ha risposto con "
                    f"codice {response.status_code}."
            }

        # Evita download troppo grandi

        content_length = (
            response.headers.get(
                "Content-Length"
            )
        )

        if content_length:

            try:

                if int(content_length) > 5_000_000:

                    return {
                        "success": False,
                        "errore":
                            "Pagina troppo grande."
                    }

            except ValueError:
                pass

        if len(response.content) > 5_000_000:

            return {
                "success": False,
                "errore":
                    "Pagina troppo grande."
            }

        # Accettiamo solo HTML

        content_type = (
            response.headers
            .get("Content-Type", "")
            .lower()
        )

        if (
            "text/html" not in content_type
            and
            "application/xhtml+xml"
            not in content_type
        ):

            return {
                "success": False,
                "errore":
                    "Il link non contiene una pagina HTML."
            }

        # ==========================================
        # PARSING HTML
        # ==========================================

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg"
            ]
        ):
            element.decompose()

        title = ""

        if soup.title:

            title = soup.title.get_text(
                " ",
                strip=True
            )

        text = soup.get_text(
            " ",
            strip=True
        )

        # Limitiamo il testo inviato all'AI

        text = text[:15000]

        if not title and not text:

            return {
                "success": False,
                "errore":
                    "Non è stato possibile leggere "
                    "il contenuto dell'annuncio."
            }

        # ==========================================
        # ESTRAZIONE AI
        # ==========================================

        estrazione_ai = (
            estrai_dati_annuncio_ai(
                title,
                text,
                data.lingua
            )
        )

        if not estrazione_ai.get(
            "success"
        ):

            return {
                "success": False,
                "errore":
                    estrazione_ai.get(
                        "errore",
                        "AI extraction failed."
                    )
            }

        return {

            "success": True,

            "url": url,

            "titolo": title,

            "dati_estratti":
                estrazione_ai["dati"]
        }

    except requests.Timeout:

        return {
            "success": False,
            "errore":
                "Timeout durante la lettura dell'annuncio."
        }

    except requests.RequestException as e:

        print(
            "ERRORE DOWNLOAD ANNUNCIO:",
            str(e)
        )

        return {
            "success": False,
            "errore":
                "Non è stato possibile leggere l'annuncio."
        }

    except Exception as e:

        print(
            "ERRORE EXTRACT LISTING:",
            str(e)
        )

        return {
            "success": False,
            "errore":
                "Errore durante l'analisi dell'annuncio."
        }