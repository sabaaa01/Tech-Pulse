# Tech Pulse Newsletter Agent

An agentic newsletter service that fetches current TechCrunch stories, ranks them with OpenAI, writes short teasers, and delivers newsletters by email or WhatsApp.

## Requirements

- Python 3.10+
- OpenAI API key
- SMTP credentials for email delivery
- Twilio credentials for WhatsApp delivery (optional)

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` with your credentials. Never commit that file.

## Run

```powershell
python app.py
```

Open `http://localhost:5000` to request a newsletter by email.

### API endpoints

- `POST /email` with JSON `{ "topic": "AI", "email": "you@example.com" }`
- `POST /whatsapp` for a Twilio WhatsApp webhook

## Security note

If credentials have ever been exposed outside your local machine, revoke or rotate them before deploying this project.