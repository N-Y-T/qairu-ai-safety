"""Live Qwen bridge. The access-control policy is imported unchanged."""
import datetime
import hmac
import secrets
import threading
import uuid
from pathlib import Path


class WebBridge:
    def __init__(self, lab, modes, token=None):
        self.lab = lab
        self.modes = tuple(modes)
        self.token = token or secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.session_id = str(uuid.uuid4())

    def authorized(self, token):
        return isinstance(token, str) and hmac.compare_digest(
            token.encode(), self.token.encode())

    def health(self, token):
        if not self.authorized(token):
            return {"ok": False, "error": "Неверный ключ подключения Colab."}
        return {"ok": True, "model_id": self.lab.model_id,
                "revision": self.lab.revision, "device": str(self.lab.device),
                "session_id": self.session_id, "modes": self.modes,
                "max_new_tokens": self.lab.max_new_tokens, "seed": self.lab.seed,
                "independent_requests": True}

    def chat(self, token, mode, supplied_code, user_message, compare=False):
        if not self.authorized(token):
            return {"ok": False, "error": "Неверный ключ подключения Colab."}
        if mode not in self.modes or not isinstance(compare, bool):
            return {"ok": False, "error": "Неизвестный режим."}
        if not isinstance(supplied_code, str) or not isinstance(user_message, str):
            return {"ok": False, "error": "Ожидался текст."}
        if len(supplied_code) > 128 or len(user_message) > 6000:
            return {"ok": False, "error": "Лимит: код 128 символов, сообщение 6000 символов; полный ввод до 2048 токенов."}
        request_id = str(uuid.uuid4())
        selected = self.modes if compare else (mode,)
        records = []
        try:
            with self.lock:
                for selected_mode in selected:
                    r = self.lab.run_case(selected_mode, supplied_code, user_message,
                        phase="website", case_id="WEB-" + request_id)
                    r["credential_valid"] = self.lab.backend_verify(supplied_code)
                    r["unauthorized_grant"] = r["access_granted"] and not r["credential_valid"]
                    # Full researcher log remains in Colab; no tunnel credential is logged.
                    records.append(r)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "records": records}
        except Exception:
            return {"ok": False, "error": "Ошибка генерации в Colab. Проверь вывод ячейки и состояние GPU.", "records": records}
        return {"ok": True, "request_id": request_id, "session_id": self.session_id,
                "records": records, "inference": "live_colab_qwen",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def launch_bridge(bridge, share=True, port=None):
    import gradio as gr
    import inspect
    with gr.Blocks(title="QAIRU · Colab Qwen bridge", analytics_enabled=False) as app:
        gr.Markdown("# Qwen в Colab\nМост к сайту QAIRU Prompt Injection Lab. "
                    "Модель выполняется в этой среде. Ключ вводится на сайте и хранится только в памяти вкладки.")
        with gr.Row(visible=False):
            key = gr.Textbox(type="password")
            mode = gr.Textbox()
            code = gr.Textbox()
            message = gr.Textbox()
            compare = gr.Checkbox()
            result = gr.JSON()
            health_result = gr.JSON()
            send = gr.Button()
            status = gr.Button()
        visibility = ({"show_api": False} if "show_api" in inspect.signature(send.click).parameters
                      else {"api_visibility": "undocumented"})
        send.click(bridge.chat, [key, mode, code, message, compare], result,
                   api_name="chat", concurrency_limit=1, concurrency_id="qwen", **visibility)
        status.click(bridge.health, [key], health_result, api_name="health", **visibility)
    app.queue(max_size=16)
    launch_visibility = ({"show_api": False} if "show_api" in inspect.signature(app.launch).parameters
                         else {"footer_links": []})
    _, local_url, share_url = app.launch(share=share, server_port=port,
        prevent_thread_lock=True, inline=False, show_error=False,
        allowed_paths=[], blocked_paths=["/content/drive"],
        app_kwargs={"docs_url": None, "redoc_url": None}, **launch_visibility)
    return app, share_url or local_url
