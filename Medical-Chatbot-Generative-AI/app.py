"""Flask app for the medical chatbot."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request

load_dotenv()

app = Flask(__name__)

CHATBOT = None
INIT_ERROR = None


def _init_chatbot():
    global CHATBOT, INIT_ERROR

    if CHATBOT is not None or INIT_ERROR is not None:
        return

    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        INIT_ERROR = "Missing environment variable: PINECONE_INDEX_NAME"
        return

    try:
        from src.helper import build_qa_chain

        CHATBOT = build_qa_chain(index_name=index_name, namespace=os.getenv("PINECONE_NAMESPACE"))
    except Exception as exc:  # pragma: no cover - surfacing startup errors
        INIT_ERROR = f"Failed to initialize chatbot: {exc}"


@app.route("/")
def home():
    return render_template_string(
        """
        <!doctype html>
        <html>
          <head><title>Medical Chatbot</title></head>
          <body>
            <h2>Medical Chatbot</h2>
            <form id="chat-form">
              <input id="msg" placeholder="Ask a medical question" style="width:360px" />
              <button type="submit">Send</button>
            </form>
            <pre id="out"></pre>
            <script>
              const form = document.getElementById('chat-form');
              const out = document.getElementById('out');
              form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const message = document.getElementById('msg').value;
                const res = await fetch('/chat', {
                  method: 'POST',
                  headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({message})
                });
                const data = await res.json();
                out.textContent = data.response || data.error;
              });
            </script>
          </body>
        </html>
        """
    )


@app.route("/health")
def health():
    _init_chatbot()
    if INIT_ERROR:
        return jsonify({"status": "error", "detail": INIT_ERROR}), 500
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    _init_chatbot()

    if INIT_ERROR:
        return jsonify({"error": INIT_ERROR}), 500

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Please provide a non-empty `message`."}), 400

    try:
        result = CHATBOT.invoke({"query": message})
        return jsonify({"response": result.get("result", "")})
    except Exception as exc:  # pragma: no cover - runtime errors returned to client
        return jsonify({"error": f"Chatbot failed to respond: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
