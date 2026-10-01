"""
agent.py — Google ADK agent definitions for the Newsletter Curator.

Model: openai/gpt-4o-mini via ADK's bundled LiteLlm adapter.
       All tool-level LLM calls (ranking, teasers) use the openai client directly.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm

from . import tools

# LiteLlm routes "openai/..." calls straight to OpenAI using your OPENAI_API_KEY.
# It is bundled inside google-adk[extensions] — not a standalone dependency.
MODEL = LiteLlm(model="openai/gpt-4o-mini")

# ── Sub-agent 1: Scraper ─────────────────────────────────────────────────────
scraper_agent = LlmAgent(
    name="scraper_agent",
    model=MODEL,
    description="Fetches the latest TechCrunch articles for a topic.",
    tools=[tools.fetch_tech_articles],
    output_key="raw_articles",
    instruction="""You are a web scraper agent.

1. The user provides a topic (e.g. 'AI', 'startups', 'fintech').
2. Call `fetch_tech_articles` with that topic and limit=10.
3. Output only: "Scraped <N> articles for topic '<topic>'."
""",
)


ranker_agent = LlmAgent(
    name="ranker_agent",
    model=MODEL,
    description="Picks the 3 most relevant articles from the scraped list.",
    tools=[tools.rank_articles],
    output_key="ranked_articles",
    instruction="""You are a content ranking agent.

1. Read the scraped articles from session state key 'raw_articles'.
2. Call `rank_articles` with:
   - articles_json = value of 'raw_articles'
   - topic = original user topic
   - top_n = 3
3. Output only: "Ranked and selected <N> articles."
""",
)


copywriter_agent = LlmAgent(
    name="copywriter_agent",
    model=MODEL,
    description="Writes teasers and handles delivery (WhatsApp or Email).",
    tools=[
        tools.generate_teaser, 
        tools.format_newsletter, 
        tools.format_email_newsletter, 
        tools.send_email
    ],
    instruction="""You are a newsletter copywriter and delivery agent.

1. Read ranked articles from 'ranked_articles' (JSON list).
2. For EACH article, call `generate_teaser` with:
   - article_json = JSON string of that article
   - topic = original user topic
   Add the returned teaser as the 'teaser' key in the article dict.
3. DELIVERY LOGIC:
   - IF the user provided an EMAIL address (in the topic or message context), call `format_email_newsletter` then `send_email`.
   - ALWAYS return the formatted text (WhatsApp or HTML) as your final response so it can be handled by the caller.
   - For email, use Subject: "Tech Pulse: <topic>".
""",
)


root_agent = SequentialAgent(
    name="newsletter_root",
    description=(
        "Generates a WhatsApp-ready tech newsletter. "
        "Input: topic only (e.g. 'AI' or 'startups'). "
        "Pipeline: scrape → rank → write teasers → format."
    ),
    sub_agents=[scraper_agent, ranker_agent, copywriter_agent],
)