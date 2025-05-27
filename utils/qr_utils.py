import qrcode
import base64
from io import BytesIO

def generate_qr_base64(user_id):
    url = f"http://localhost:5000/?user_id={user_id}"
    img = qrcode.make(url)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
