import os
from azure.communication.email import EmailClient
from azure.core.credentials import AzureKeyCredential

conn = os.environ.get("ACS_EMAIL_CONNECTION_STRING", "")
if not conn:
    raise SystemExit("ACS_EMAIL_CONNECTION_STRING no está definido")

sender = os.environ.get("ACS_EMAIL_FROM", "DoNotReply@gotogym.store")
recipient = os.environ.get("ACS_EMAIL_TO", "contacto@gotogym.store")

# Parse manual para evitar errores de formato en la URL
endpoint = None
access_key = None
for part in conn.split(";"):
    if not part:
        continue
    key, _, val = part.partition("=")
    if key.lower() == "endpoint":
        endpoint = val.strip()
    elif key.lower() == "accesskey":
        access_key = val.strip()

if not endpoint or not access_key:
    raise SystemExit("Connection string inválida: falta endpoint o accesskey")

if not endpoint.endswith("/"):
    endpoint = endpoint + "/"

client = EmailClient(endpoint, AzureKeyCredential(access_key))

message = {
    "senderAddress": sender,
    "recipients": {"to": [{"address": recipient}]},
    "content": {
        "subject": "Prueba ACS Email - GoToGym",
        "plainText": "Este es un correo de prueba enviado por Azure Communication Services.",
        "html": "<p>Este es un correo de prueba enviado por <strong>Azure Communication Services</strong>.</p>",
    },
}

poller = client.begin_send(message)
result = poller.result()

print("Result:", result)
print("MessageId:", getattr(result, "message_id", None))
print("Status:", getattr(result, "status", None))
