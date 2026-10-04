import json
import multiprocessing
import os

from dotenv import load_dotenv
from uvicorn import run
from fastapi import FastAPI, File, Form, UploadFile
from openai import OpenAI
from typing import Annotated, Any, Dict


def parse_json_from_response(response) -> Dict[str, Any]:
    # Chat Completions JSON-schema responses are normally valid JSON in message.content.
    content = response.choices[0].message.content
    return json.loads(content)

PORT = os.getenv("PORT", 8081)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

JOINT_PROMPT = """Rol:
Je fungeert als digitale assistent die samen met de patiënt de huidige medicatielijst opstelt.

Doel:
Het controleren of de door de patiënt ingevoerde geneesmiddelnaam correct is gespeld en ervoor zorgen dat de juiste naam wordt genoteerd.

Instructies:
    Voer achtereenvolgens Taak 1 en 2 uit. 
    
    Taak 1 - interne validatie:
        * Beoordeel de ingevoerde geneesmiddelnaam en stel vast of de spelling juist is.
        
    Gebruik de uitkomst van Taak 1 als basis voor Taak 2.
    
    Taak 2 - interne validatie:
        * Indien bij Taak 1 vastgesteld is dat de spelling onjuist is: geef de juiste schrijfwijze. Als een geneesmiddel zonder Nederlandse naam wordt ingevoerd, gebruik dan de internationaal gangbare naam. Wanneer zowel een Nederlandse als internationale naam bestaat, heeft de Nederlandse naam de voorkeur. Kies bij twijfel de naam die het beste aansluit bij de invoer van de patiënt. Houd rekening met een patiëntenpopulatie die doorgaans geen internationale achtergrond heeft.
        * Indien bij Taak 1 vastgesteld is dat de spelling correct is: geef de correct gespelde medicijnnaam terug.
        * Bij twijfel of geen herkenning van het geneesmiddel: geef geen output voor Taak 2.        
    
Beperkingen:
    • Vermijd aannames.
    • Trek geen conclusies op basis van vermoedens of context.
    • Elke spelfout, hoe klein ook, betekent dat de naam onjuist is gespeld.
    • Stel geen geneesmiddelen voor die niet meer op de markt zijn.

Verwachtingen:
    • Houd de output zo beknopt mogelijk en volg strikt de instructies.
    • Geef geen inhoudelijke of farmacologische informatie over het geneesmiddel.
    • Vermeld geen gebruikte bronnen.
    • Verwaarloos fouten in hoofdlettergebruik

Context:
Bij het opstellen van een medicatielijst is het cruciaal dat geneesmiddelnamen correct worden vastgelegd. Als er twijfel is over de spelling, moet dit worden herkend en moet de patiënt worden gevraagd de naam opnieuw in te voeren, in plaats van een mogelijk onjuist geneesmiddel over te nemen."""

RESPONSE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "medicine_spellcheck_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task1_label": {
                    "type": "string",
                    "enum": ["Ja", "Nee"],
                    "description": "Taak 1: 'Ja' als de invoer correct gespeld is, anders 'Nee'."
                },
                "task2_corrected": {
                    "type": "string",
                    "description": "Taak 2: correcte geneesmiddelnaam. Laat leeg indien van toepassing volgens instructie."
                }
            },
            "required": ["task1_label", "task2_corrected"]
        }
    }
}

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}

@app.post("/")
async def check_spelling(image: UploadFile = File(None),
    user_input: str = Form(""),
    prompt: str | None = Form(None),
    intent_options: str | None = Form(None),
    key_user_account: str | None = Form(None),
):
    user_input = user_input

    print(user_input)
    
    message = client.responses.parse(
        model="gpt-5.5-2026-04-23",
        instructions=JOINT_PROMPT,
        text=RESPONSE_FORMAT,
        input="Voer Taak 1 en Taak 2 uit voor deze invoer. "
        "Gebruik de gestructureerde velden voor de interne resultaten en de patiëntboodschap.\n\n"
        f"Geneesmiddelnaam: {user_input.lower()}"
    )

    results = json.loads(message.output_text)

    if results["task1_label"] == "Ja":
        return {"intent": "correct", "connector_label": results["task2_corrected"]}
    else:
        if results["task2_corrected"] == "":
            return {"intent": "unknown", "connector_label": user_input}
        else:
            return {"intent": "suggestion", "connector_label": results["task2_corrected"]}

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    run(app, host="0.0.0.0", port=PORT, reload=False, workers=1)
