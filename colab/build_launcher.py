import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
app_source = (root / 'colab/midterm_app.py').read_text()
bridge_source = (root / 'colab/bridge.py').read_text()
launcher = '''# @title Запустить Qwen и подключение к веб-чату QAIRU
# Выполнять в Google Colab. Исходные логи 288 тестов не изменяются.
import importlib.util, subprocess, sys, os, json, datetime, uuid
from pathlib import Path
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
if importlib.util.find_spec("gradio") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio==5.49.1"])
import torch
WEB_ROOT = Path("/content/qairu_web_chat")
WEB_ROOT.mkdir(parents=True, exist_ok=True)
'''
launcher += 'APP_SOURCE = ' + repr(app_source) + '\n'
launcher += 'BRIDGE_SOURCE = ' + repr(bridge_source) + '\n'
launcher += '''
(WEB_ROOT / "midterm_app.py").write_text(APP_SOURCE, encoding="utf-8")
(WEB_ROOT / "bridge.py").write_text(BRIDGE_SOURCE, encoding="utf-8")
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
WEB_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
WEB_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
if "MODEL" not in globals() or "TOKENIZER" not in globals():
    if not torch.cuda.is_available():
        raise RuntimeError("Выбери в Colab: Среда выполнения → Сменить среду выполнения → T4 GPU. Затем запусти ячейку снова.")
    if importlib.util.find_spec("transformers") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.1"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Загружаю Qwen2.5-0.5B-Instruct на GPU...")
    TOKENIZER = AutoTokenizer.from_pretrained(WEB_MODEL_ID, revision=WEB_REVISION)
    MODEL = AutoModelForCausalLM.from_pretrained(WEB_MODEL_ID, revision=WEB_REVISION,
        torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
else:
    actual_id = getattr(MODEL.config, "_name_or_path", "")
    if actual_id != WEB_MODEL_ID and "Qwen2.5-0.5B-Instruct" not in actual_id:
        raise RuntimeError("Уже загружена другая модель. Открой отдельную среду для Qwen2.5-0.5B-Instruct.")
    print("Использую уже загруженную Qwen в этой среде.")
import importlib
import midterm_app as _web_policy
import bridge as _web_bridge_module
# Reload the bridge when the cell is rerun after a source update.
importlib.reload(_web_bridge_module)
AccessControlLab, MODES = _web_policy.AccessControlLab, _web_policy.MODES
WebBridge, launch_bridge = _web_bridge_module.WebBridge, _web_bridge_module.launch_bridge
if "WEB_GRADIO" in globals():
    WEB_GRADIO.close()
WEB_LOG_DIR = WEB_ROOT / "sessions" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
WEB_ACTUAL_REVISION = getattr(MODEL.config, "_commit_hash", None) or globals().get("MODEL_REVISION", "unverified")
WEB_LAB = AccessControlLab(MODEL, TOKENIZER, WEB_MODEL_ID, WEB_ACTUAL_REVISION, WEB_LOG_DIR,
                           seed=2026, max_new_tokens=64)
WEB_BRIDGE = WebBridge(WEB_LAB, MODES)
WEB_GRADIO, WEB_URL = launch_bridge(WEB_BRIDGE, share=True)
print("\\nПодключение к сайту: https://qairu-injection-lab.brown-sky-2597.chatgpt.site")
print("Адрес Colab:", WEB_URL)
print("Ключ подключения:", WEB_BRIDGE.token)
print("Модель:", WEB_MODEL_ID, "|", WEB_LAB.device, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Новые запросы: отдельная фаза website, журнал", str(WEB_LOG_DIR / "requests.jsonl"))
print("Скопируй адрес и ключ в блок подключения на сайте. Не закрывай среду Colab.")
'''
(root / 'dist/colab_launch.py').write_text(launcher)
notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'QAIRU_Live_Web_Chat.ipynb'},'accelerator':'GPU','kernelspec':{'display_name':'Python 3','name':'python3'}},'cells':[
 {'cell_type':'markdown','metadata':{},'source':['# QAIRU · Веб-чат с живой Qwen\nВыберите T4 GPU и запустите ячейку. Затем скопируйте адрес и ключ подключения на [сайт](https://qairu-injection-lab.brown-sky-2597.chatgpt.site).\n\nРаботает с уже загруженной моделью в исходном Colab либо загружает Qwen в новой среде. Это независимые одиночные запросы; история на сайте не отправляется модели. Ключ временный; при перезапуске подключитесь снова.']},
 {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':launcher.splitlines(keepends=True)}]}
(root / 'dist/QAIRU_Live_Web_Chat.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
print('Launcher and notebook built')
