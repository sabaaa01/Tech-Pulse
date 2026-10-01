import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
import sys
import time
import asyncio
import threading


sys.stdout.reconfigure(line_buffering=True)

from flask import Flask, request, render_template
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from newsletter_agent.agent import root_agent

load_dotenv()

app = Flask(__name__)

#
SESSION_SERVICE = InMemorySessionService()
RUNNER = Runner(
    agent=root_agent,
    app_name="newsletter_bot",
    session_service=SESSION_SERVICE,
)

#
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = "whatsapp:+14155238886"

if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
else:
    twilio_client = None
    print("Warning: Twilio credentials not found.", flush=True)


async def _generate_newsletter(user_message: str, sender_number: str) -> str:
    """Async: creates ADK session and runs the newsletter pipeline."""
    session = await SESSION_SERVICE.create_session(
        app_name="newsletter_bot",
        user_id=sender_number,
    )

    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_message)],
    )

    response_text = ""
    async for event in RUNNER.run_async(
        user_id=sender_number,
        session_id=session.id,
        new_message=user_content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    return response_text or "Sorry, I couldn't generate a newsletter. Please try again."


def _send_whatsapp(text: str, to: str):
    """Splits and sends a message via Twilio. Stays under 1600-char Twilio limit."""
    if not twilio_client:
        print(f"[!] No Twilio client. Response:\n{text[:200]}", flush=True)
        return

    MAX = 1400  # safe buffer under Twilio's 1600-char hard limit

    if len(text) <= MAX:
        twilio_client.messages.create(from_=TWILIO_PHONE_NUMBER, body=text, to=to)
        print(f"[+] Message sent to {to}.", flush=True)
    else:
        parts, remaining = [], text
        while remaining:
            if len(remaining) > MAX:
                idx = remaining.rfind("\n", 0, MAX)
                if idx <= 0:
                    idx = MAX
                parts.append(remaining[:idx])
                remaining = remaining[idx:].lstrip("\n")
            else:
                parts.append(remaining)
                break
        for i, part in enumerate(parts):
            twilio_client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                body=f"[{i+1}/{len(parts)}]\n{part}",
                to=to,
            )
            time.sleep(1)
        print(f"[+] Sent {len(parts)} parts to {to}.", flush=True)


def run_adk_and_send(user_message: str, sender_number: str):
    """Background thread: runs async ADK pipeline then sends result via Twilio."""
    print(f"[*] Background job started for {sender_number} | topic: {user_message}", flush=True)
    try:
        response_text = asyncio.run(_generate_newsletter(user_message, sender_number))
        _send_whatsapp(response_text, sender_number)
    except Exception as e:
        print(f"[-] Error in background job: {e}", flush=True)
        if twilio_client:
            twilio_client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                body="Sorry, I encountered an error generating your newsletter.",
                to=sender_number,
            )


def run_adk_for_email(topic: str, recipient_email: str):
    """Background thread: runs ADK pipeline for email delivery."""
    print(f"[*] Background Email job started for {recipient_email} | topic: {topic}", flush=True)
    try:
        message_with_email = f"Topic: {topic}. Deliver to email: {recipient_email}"
        asyncio.run(_generate_newsletter(message_with_email, recipient_email))
        print(f"[+] Email processing complete for {recipient_email}", flush=True)
    except Exception as e:
        print(f"[-] Error in email background job: {e}", flush=True)


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Handles incoming WhatsApp messages from Twilio."""
    user_message = request.values.get("Body", "").strip()
    sender_number = request.values.get("From", "")

    if not user_message:
        return "No message found", 400

    print(f"[Webhook] From {sender_number}: {user_message}", flush=True)

    if sender_number:
        thread = threading.Thread(
            target=run_adk_and_send,
            args=(user_message, sender_number),
            daemon=True,
        )
        thread.start()

    resp = MessagingResponse()
    resp.message("🔍 Searching for the latest tech news... This may take ~30 seconds.")
    return str(resp)


@app.route("/email", methods=["POST"])
def email_newsletter():
    """Endpoint for requesting an email newsletter.
    Expects JSON: {"topic": "AI", "email": "user@example.com"}
    """
    data = request.get_json() or request.form
    topic = data.get("topic", "").strip()
    email = data.get("email", "").strip()

    if not topic or not email:
        return "Topic and Email are required", 400

    print(f"[Email Request] To {email}: {topic}", flush=True)

    thread = threading.Thread(
        target=run_adk_for_email,
        args=(topic, email),
        daemon=True,
    )
    thread.start()

    return f"Accepted! Newsletter for '{topic}' will be sent to {email} shortly."


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print(">>> APP IS STARTING — logs will appear here <<<", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
