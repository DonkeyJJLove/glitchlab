import os, json, subprocess, base64, sys
from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage

# --- ENV / STATE ---
ENV = Path(".env")
STATE = Path(".glx/state.json")

def load_env():
    data = {}
    if ENV.exists():
        for ln in ENV.read_text(encoding="utf-8").splitlines():
            ln=ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln: continue
            k,v = ln.split("=",1); data[k.strip()] = v.strip()
    for k,v in data.items():
        os.environ.setdefault(k,v)

def git(*args) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()

def short7(sha:str) -> str: return sha[:7]

def build_patch(base_sha:str, attach_path:Path):
    patch = subprocess.check_output(["git","format-patch","--stdout", f"{base_sha}..HEAD"], text=True)
    attach_path.write_text(patch, encoding="utf-8")

def build_message(subject:str, body:str, attach_path:Path, sender:str, to:str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    data = attach_path.read_bytes()
    msg.add_attachment(data, maintype="text", subtype="plain", filename=attach_path.name)
    return msg

def gmail_send(msg: EmailMessage, credentials_path:str, token_path:str):
    # Lazy import (szybciej ładuje w hookach)
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import google.oauth2.credentials as oauth2

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

    creds = None
    if os.path.exists(token_path):
        creds = oauth2.Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # refresh
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # save token
        Path(token_path).parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return service.users().messages().send(userId="me", body={"raw": raw}).execute()

def main():
    load_env()
    if not STATE.exists():
        print("Missing .glx/state.json (need base_sha)", file=sys.stderr); sys.exit(1)

    st = json.loads(STATE.read_text(encoding="utf-8"))
    base_sha = st.get("base_sha")
    if not base_sha:
        print("state.json missing base_sha", file=sys.stderr); sys.exit(1)

    cred_path = os.environ["GLX_GOOGLE_CREDENTIALS"]
    token_path = os.environ["GLX_GOOGLE_TOKEN"]
    to_addr   = os.environ["GLX_MAIL_TO"]
    project   = os.environ.get("GLX_PROJECT","glitchlab")
    branch    = os.environ.get("GLX_BRANCH","main")

    head_full  = git("rev-parse","HEAD")
    base_short = short7(git("rev-parse","--short=7", base_sha))
    head_short = short7(head_full)
    seq = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    Path(".glx").mkdir(exist_ok=True, parents=True)
    attach_name = f"glx-{project}-{branch}-SEQ-{seq}-{base_short}..{head_short}.patch"
    attach_path = Path(".glx")/attach_name
    build_patch(base_sha, attach_path)

    subject = f"GLX DIFF | {project} | {branch} | SEQ:{seq} | RANGE:{base_short}..{head_short}"
    body = f"Project: {project}\nBranch: {branch}\nBase..Head: {base_sha}..{head_full}\nSEQ: {seq}\nAttachment: {attach_name}\n"

    sender = "me"  # Gmail API przyjmie 'me' (konto z tokena)
    msg = build_message(subject, body, attach_path, sender, to_addr)

    resp = gmail_send(msg, cred_path, token_path)
    print(f"Sent via Gmail API: id={resp.get('id')}  subject={subject}")

if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"Missing .env key: {e}", file=sys.stderr); sys.exit(2)
    except subprocess.CalledProcessError as e:
        print(f"git error: {e}", file=sys.stderr); sys.exit(3)
