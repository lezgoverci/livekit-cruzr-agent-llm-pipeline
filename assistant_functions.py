import aiohttp
from typing import Annotated
from livekit.agents import llm

class AssistantFnc(llm.FunctionContext):
    @llm.ai_callable()
    async def send_email(
        self,
        to: Annotated[str, llm.TypeInfo(description="Email address of the recipient")],
        subject: Annotated[str, llm.TypeInfo(description="Subject of the email")],
        body: Annotated[str, llm.TypeInfo(description="Body of the email")]
    ):
        """
        Sends an email using n8n webhook.
        """
        webhook_url = "https://n8n.xponent.ph/webhook/livekit-cruzr-gmail"
        payload = {
            "to": to,
            "subject": subject,
            "body": body
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    return "Email sent successfully."
                else:
                    raise Exception(f"Failed to send email. Status code: {response.status}")
                
    @llm.ai_callable()
    async def add_google_sheets_row(
        self,
        sheet_name: Annotated[str, llm.TypeInfo(description="Name of the Google Sheet")],
        row_data: Annotated[str, llm.TypeInfo(description="Comma-separated string of values to be added as a new row")]
    ):
        """
        Adds a new row to a Google Sheet using n8n webhook.
        """
        webhook_url = "https://n8n.xponent.ph/webhook/livekit-cruzr-sheets"
        
        # Convert the comma-separated string to a list
        row_data_list = [item.strip() for item in row_data.split(',')]
        
        payload = {
            "sheet_name": sheet_name,
            "row_data": row_data_list
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    return "Row added to Google Sheet successfully."
                else:
                    raise Exception(f"Failed to add row to Google Sheet. Status code: {response.status}")



# Create an instance of AssistantFnc
fnc_ctx = AssistantFnc()