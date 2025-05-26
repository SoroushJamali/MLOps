import os, requests, logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_URL = os.getenv("SLACK_WEBHOOK_URL")
PD_KEY    = os.getenv("PAGERDUTY_ROUTING_KEY")

def send_slack(msg):
    if not SLACK_URL:
        logger.warning("No Slack URL configured.")
        return
    requests.post(SLACK_URL, json={"text": msg})

def send_pagerduty(msg):
    if not PD_KEY:
        logger.warning("No PagerDuty key configured.")
        return
    payload = {
        "routing_key": PD_KEY,
        "event_action": "trigger",
        "payload": {"summary": msg, "source": "mlops-monitor"}
    }
    requests.post("https://events.pagerduty.com/v2/enqueue", json=payload)
