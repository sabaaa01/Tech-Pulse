"""
tools.py — Plain Python functions that ADK agents call as tools.

In ADK 1.x, tools are just regular functions with type-annotated args and
a proper docstring. ADK auto-wraps them as FunctionTool.
All actual LLM calls (ranking, teaser generation) use the openai client directly.
"""

import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

CATEGORY_MAP = {
    "ai": "artificial-intelligence",
    "artificial intelligence": "artificial-intelligence",
    "startups": "startups",
    "security": "security",
    "apps": "apps",
    "fintech": "fintech",
    "crypto": "cryptocurrency",
    "venture": "venture",
}

BASE_URL = "https://techcrunch.com/category/"

def _fetch_content_raw(url: str) -> str:
    """Internal helper: fetches article body text (not exposed as an ADK tool)."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.select_one("div.entry-content, div.article-content")
        if content_div:
            return content_div.get_text(strip=True, separator="\n")[:8000]
        return ""
    except Exception as e:
        print(f"[Content Fetch Internal Error] {url}: {e}", flush=True)
        return ""


# ── ADK Tool Functions ───────────────────────────────────────────────────────
# No decorator needed — ADK auto-wraps plain functions passed to LlmAgent(tools=[...])

def fetch_tech_articles(topic: str, limit: int = 5) -> str:
    """Fetches article metadata from TechCrunch for a given topic.

    Args:
        topic: Tech topic to search, e.g. 'ai', 'startups', 'security'.
        limit: Maximum number of articles to return (default 10).

    Returns:
        JSON string — list of dicts with keys: title, link, image_url.
    """
    topic_slug = CATEGORY_MAP.get(topic.lower().strip())
    url = f"{BASE_URL}{topic_slug}/" if topic_slug else f"{BASE_URL}startups/"
    print(f"[Scraper] Fetching: {url}", flush=True)

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        articles = []
        cards = soup.select("div.loop-card") or soup.select("div.post-block")

        for card in cards:
            if len(articles) >= limit:
                break
            title_tag = card.select_one("a.loop-card__title-link, h2.post-block__title a")
            if not title_tag:
                continue
            articles.append({
                "title": title_tag.get_text().strip()[:100],
                "link": title_tag["href"],
            })

        print(f"[Scraper] Found {len(articles)} articles.", flush=True)
        return json.dumps(articles)

    except Exception as e:
        print(f"[Scraper Error] {e}", flush=True)
        return json.dumps([])

def fetch_article_content(url: str) -> str:
    """Fetches the full text content of a single article URL.

    Args:
        url: Full URL of the article to fetch.

    Returns:
        Article body text (up to 8000 characters).
    """
    return _fetch_content_raw(url)


def rank_articles(articles_json: str, topic: str, top_n: int = 3) -> str:
    """Selects the top N most relevant articles for a topic using OpenAI gpt-4o-mini.

    Args:
        articles_json: JSON string list of article dicts (title, link, image_url).
        topic: Newsletter topic to rank against.
        top_n: Number of top articles to select (default 3).

    Returns:
        JSON string — list of the selected top article dicts.
    """
    articles = json.loads(articles_json)
    if not articles:
        return json.dumps([])

    articles_str = "\n".join(
        f"{i}. Title: {a['title']}\n   Link: {a['link']}"
        for i, a in enumerate(articles)
    )

    prompt = f"""You are an expert tech newsletter editor.
Select the TOP {top_n} most suitable articles for the topic below.

Topic: {topic}

Criteria:
1. Direct relevance to the topic
2. Significance (major events > opinions)
3. Novelty (recent news preferred)
4. Diversity (avoid duplicate stories)

Articles:
{articles_str}

Return ONLY a JSON array of the selected indices, e.g. [0, 4, 2].
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        start, end = content.find("["), content.rfind("]") + 1
        indices = json.loads(content[start:end] if start >= 0 else content)
        selected = [articles[i] for i in indices if 0 <= i < len(articles)]
        print(f"[Ranker] Selected {len(selected)} articles.", flush=True)
        return json.dumps(selected[:top_n])
    except Exception as e:
        print(f"[Ranker Error] {e}", flush=True)
        return json.dumps(articles[:top_n])


def generate_teaser(article_json: str, topic: str) -> str:
    """Generates a short WhatsApp-optimised teaser for a single article using OpenAI gpt-4o-mini.

    Args:
        article_json: JSON string of an article dict (title, link, image_url).
        topic: Newsletter topic for context.

    Returns:
        Teaser string — max 25 words, no emojis.
    """
    article = json.loads(article_json, strict=False)
    full_text = _fetch_content_raw(article["link"])

    prompt = f"""You are a WhatsApp newsletter copywriter.
Write one short, punchy teaser sentence (max 25 words, no emojis) for this article.

Topic: {topic}
Article: {article['title']}
Content: {full_text[:4000]}

Output the teaser text only.
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Copywriter Error] {e}", flush=True)
        return "Check out this update."


def format_newsletter(items_json: str, topic: str) -> str:
    """Assembles the final newsletter string formatted for WhatsApp.

    Args:
        items_json: JSON string list of dicts with keys: title, link, image_url, teaser.
        topic: Newsletter topic used in the header.

    Returns:
        Formatted WhatsApp-ready newsletter string.
    """
    items = json.loads(items_json, strict=False)
    lines = [f"📰 *Tech Pulse: {topic}*\n"]
    for i, item in enumerate(items, 1):
        lines.append(f"*{i}. {item['title']}*")
        lines.append(f"_{item.get('teaser', '')}_")
        lines.append(f"🔗 {item['link']}\n")
    lines.append("------------------")
    lines.append("💡 _Reply with a new topic to explore!_")
    return "\n".join(lines)


def format_email_newsletter(items_json: str, topic: str) -> str:
    """Assembles the final newsletter string formatted as HTML for Email.

    Args:
        items_json: JSON string list of dicts with keys: title, link, teaser.
        topic: Newsletter topic used in the subject/header.

    Returns:
        Clean HTML string for the email body.
    """
    items = json.loads(items_json, strict=False)
    html = [
        f"<html><body style='font-family: Arial, sans-serif; color: #333;'>",
        f"<h1 style='color: #2c3e50;'>📰 Tech Pulse: {topic}</h1>",
        "<hr style='border: 0; border-top: 1px solid #eee; margin: 20px 0;'>"
    ]
    
    for i, item in enumerate(items, 1):
        html.append(f"<div style='margin-bottom: 25px;'>")
        html.append(f"<h3 style='margin: 0;'>{i}. {item['title']}</h3>")
        html.append(f"<p style='font-style: italic; color: #666;'>{item.get('teaser', '')}</p>")
        html.append(f"<a href='{item['link']}' style='color: #3498db; text-decoration: none;'>Read Full Story &rarr;</a>")
        html.append(f"</div>")

    html.append("<hr style='border: 0; border-top: 1px solid #eee;'>")
    html.append("<p style='font-size: 12px; color: #999;'>Generated by your Automated Newsletter Agent.</p>")
    html.append("</body></html>")
    return "".join(html)


def send_email(to_email: str, subject: str, body_html: str) -> str:
    """Sends an HTML email using SMTP credentials from environment variables.

    Args:
        to_email: Recipient email address.
        subject: Email subject line.
        body_html: HTML content of the email.

    Returns:
        Success or error message string.
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_pass]):
        return "Error: SMTP credentials missing in .env"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        part = MIMEText(body_html, "html")
        msg.attach(part)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        print(f"[Email] Successfully sent to {to_email}", flush=True)
        return f"Done! Newsletter sent to {to_email}."
    except Exception as e:
        print(f"[Email Error] {e}", flush=True)
        return f"Error sending email: {e}"
