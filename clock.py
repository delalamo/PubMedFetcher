from email.mime.text import MIMEText
from datetime import datetime
import os
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
import requests

# from urllib2 import Request, urlopen
import urllib.request

url = os.environ["TRUSTIFI_URL"] + "/api/i/v1/email"
payload = '{"recipients":[{"email":"diego.delalamo@gmail.com"}],"title":"Title","html":"Body"}'
headers = {
    "x-trustifi-key": os.environ["TRUSTIFI_KEY"],
    "x-trustifi-secret": os.environ["TRUSTIFI_SECRET"],
    "Content-Type": "application/json",
}

# Load environment variables
load_dotenv()

# Email configuration (replace with your actual details)
# SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
# SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
# RECEIVER_EMAIL = "123@test.com"

sched = BlockingScheduler()


@sched.scheduled_job("interval", minutes=1)
def send_email():
    """Sends an email with the current time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = MIMEText(f"The current time is: {now}")

    request = urllib.request.Request(
        "https://realemail.expeditedaddons.com/?api_key="
        + os.environ["REALEMAIL_API_KEY"]
        + "&email=diego.delalamo%40example.org&fix_typos=false"
    )

    response_body = urllib.request.urlopen(request).read()
    print(response_body)

    # print(message)
    # response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.json())


sched.start()
