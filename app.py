import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler

# Load environment variables
load_dotenv()

# Email configuration (replace with your actual details)
# SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
# SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
# RECEIVER_EMAIL = "123@test.com"


def send_email():
    """Sends an email with the current time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = MIMEText(f"The current time is: {now}")
    # message["Subject"] = "Hourly Time Update"
    # message["From"] = SENDER_EMAIL
    # message["To"] = RECEIVER_EMAIL

    try:
        # with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # Use your email provider's SMTP server
        #     server.login(SENDER_EMAIL, SENDER_PASSWORD)
        #     server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
        print(f"Email sent successfully at {now}: {message}")
    except Exception as e:
        print(f"Error sending email: {e}")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(send_email, "interval", minutes=1)
    print("Press Ctrl+C to exit")
    scheduler.start()
