from datetime import datetime
import os
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

sched = BlockingScheduler()


@sched.scheduled_job("interval", minutes=1)
def send_email():
    """Sends an email with the current time."""

    print(os.environ.get("MY_EMAIL"), os.environ.get("MY_PW"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = MIMEText(f"The current time is: {now}")

    message = MIMEMultipart()
    message["From"] = os.environ.get("MY_EMAIL")
    message["To"] = os.environ.get("MY_EMAIL")
    message["Subject"] = "TEST TEST"
    message.attach(MIMEText("TEST TEST TEST ", "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.environ.get("MY_EMAIL"), os.environ.get("MY_PW"))
        server.sendmail(
            os.environ.get("MY_EMAIL"), os.environ.get("MY_EMAIL"), message.as_string()
        )


sched.start()
