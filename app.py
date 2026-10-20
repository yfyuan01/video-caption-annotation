import html
import hmac
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.parse import quote

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Video Caption Annotation", layout="wide")
ANNOTATORS = ("A", "B", "C")
DISPLAY_NAMES = {"A": "Laurence", "B": "Sophia", "C": "Extra Annotator", "Admin": "Admin"}
LABELS = ("Matched", "Unmatched", "Not sure")


def setting(name):
    value = st.secrets.get(name)
    if not value:
        st.error(f"Missing configuration: {name}. Contact the administrator.")
        st.stop()
    return str(value)


HF_REPO = setting("HF_REPO")
HF_REVISION = st.secrets.get("HF_REVISION", "main")
BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{quote(str(HF_REVISION), safe='')}"


def remote_url(path):
    # Only use filenames: the source index may contain absolute lab paths.
    name = str(path).replace("\\", "/").split("/")[-1]
    if not name or name in (".", ".."):
        raise ValueError("Invalid dataset filename")
    return f"{BASE}/{quote(name, safe='')}"


@st.cache_resource
def db_client():
    return create_client(setting("SUPABASE_URL"), setting("SUPABASE_SECRET_KEY"))


def authenticate():
    if "role" in st.session_state:
        with st.sidebar:
            st.write(f"Signed in as **{DISPLAY_NAMES[st.session_state.role]}**")
            if st.button("Sign out"):
                del st.session_state["role"]
                st.rerun()
        return st.session_state.role
    st.title("Video-Caption Consistency Annotation")
    with st.form("login"):
        role = st.selectbox(
            "Account", [*ANNOTATORS, "Admin"],
            format_func=lambda value: DISPLAY_NAMES[value],
        )
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in", type="primary")
    if submit:
        expected = setting(f"PASSWORD_{role.upper()}")
        if hmac.compare_digest(password, expected):
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


@st.cache_data(ttl=3600)
def load_index():
    df = pd.read_csv(remote_url("annotation_index.csv"), dtype={"video_id": str})
    required = {"video_id", "clip_path", "caption_path", "start_time", "end_time"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing index columns: {sorted(required - set(df.columns))}")
    if df["video_id"].duplicated().any():
        raise ValueError("Duplicate video_id values in annotation index")
    df = df.sort_values("video_id", key=lambda s: pd.to_numeric(s, errors="coerce"), kind="stable")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["annotator"] = [ANNOTATORS[i % len(ANNOTATORS)] for i in range(len(df))]
    return df


@st.cache_data(ttl=3600)
def load_captions(filename):
    df = pd.read_csv(remote_url(filename))
    required = {"ts", "te", "text", "speaker_label", "speaker_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing caption columns: {sorted(required - set(df.columns))}")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["te"] = pd.to_numeric(df["te"], errors="coerce")
    df = df.dropna(subset=["ts", "te"])
    return df[df["te"] >= df["ts"]].sort_values(["ts", "te"]).reset_index(drop=True)


def load_annotations():
    # Pagination avoids Supabase's default 1000-row API limit if the dataset grows.
    rows, offset, page_size = [], 0, 500
    while True:
        batch = db_client().table("annotations").select("video_id,annotator,label,updated_at").range(offset, offset + page_size - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return pd.DataFrame(rows, columns=["video_id", "annotator", "label", "updated_at"])


def save_annotation(video_id, annotator, label):
    db_client().table("annotations").upsert({
        "video_id": str(video_id), "annotator": annotator, "label": label,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, on_conflict="video_id").execute()


def format_time(seconds):
    value = abs(float(seconds))
    return f"{'-' if seconds < 0 else ''}{int(value // 60):02d}:{value % 60:06.3f}"


def display_captions(df, clip_start, clip_end):
    if df.empty:
        st.warning("No matching captions found.")
        return
    cards = []
    for _, row in df.iterrows():
        ts, te = float(row["ts"]), float(row["te"])
        overlap = ts < clip_end and te > clip_start
        speaker = row["speaker_name"] if pd.notna(row["speaker_name"]) and str(row["speaker_name"]).strip() else row["speaker_label"]
        speaker = "" if pd.isna(speaker) else str(speaker)
        caption = "" if pd.isna(row["text"]) else str(row["text"])
        timing = f"{format_time(ts - clip_start)} – {format_time(te - clip_start)}"
        cards.append(
            '<div style="padding:7px 10px;margin-bottom:5px;border-radius:5px;'
            f'background:{"#fff1f2" if overlap else "#f7f7f7"};'
            f'border-left:3px solid {"#d62828" if overlap else "#cccccc"};">'
            '<div style="font-size:11px;color:#777;margin-bottom:3px;">'
            f'{html.escape(timing)}{(" | " + html.escape(speaker)) if speaker else ""}</div>'
            f'<div style="font-size:13px;line-height:1.4;color:{"#d62828" if overlap else "#444"};'
            f'font-weight:{"700" if overlap else "400"};">{html.escape(caption)}</div></div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


role = authenticate()
st.title("Video-Caption Consistency Annotation")
st.caption("Watch the 30-second video and determine whether the captions accurately match its content. Submit one overall judgment per video.")

try:
    index = load_index()
    records = load_annotations()
except Exception as exc:
    st.error(f"Unable to load annotation data: {exc}")
    st.stop()

if role == "Admin":
    st.subheader("Administrator Dashboard")
    merged = index.merge(records, on="video_id", how="left", suffixes=("_assigned", ""))
    st.metric("Completed annotations", f"{len(records)} / {len(index)}")
    st.download_button("Download All Annotations (CSV)", merged.to_csv(index=False).encode("utf-8-sig"), "annotation_results.csv", "text/csv")
    st.dataframe(merged, use_container_width=True)
    st.stop()

my = index[index["annotator"] == role].reset_index(drop=True)
records = records.copy()
records["video_id"] = records["video_id"].astype(str)
finished = set(records.loc[records["annotator"] == role, "video_id"])
completed = int(my["video_id"].isin(finished).sum())
st.sidebar.progress(completed / len(my) if len(my) else 0)
st.sidebar.write(f"**Progress: {completed} / {len(my)}**")

if my.empty:
    st.info("No videos assigned to this annotator.")
    st.stop()

# Unlock review as soon as this annotator finishes their own assigned videos.
my_complete = completed == len(my)

if my_complete:
    mode = st.sidebar.radio(
        "Annotation mode",
        ["Review My Not Sure", "Review All My Annotations"],
        key=f"mode_{role}",
    )
    if mode == "Review My Not Sure":
        unsure_ids = set(records.loc[
            (records["annotator"] == role) & (records["label"] == "Not sure"),
            "video_id",
        ])
        choices = my[my["video_id"].isin(unsure_ids)]
        st.info(f"Your remaining Not sure annotations: {len(choices)}")
        if choices.empty:
            st.success("Your review is complete!")
            st.stop()
    else:
        choices = my
else:
    mode = "Annotate Remaining"
    choices = my[~my["video_id"].isin(finished)]
    st.sidebar.caption("Review unlocks when you finish all your assigned videos.")

selected_id = st.selectbox("Choose a video", choices["video_id"].tolist(), key=f"video_{role}_{mode}")
item = my.loc[my["video_id"] == selected_id].iloc[0]
clip_start, clip_end = float(item["start_time"]), float(item["end_time"])
st.subheader(f"Video {selected_id}")
st.caption(f"Original video time: {format_time(clip_start)} - {format_time(clip_end)}")
left, right = st.columns([1.2, 1], gap="large")
ready = True
with left:
    st.markdown("### Video")
    try:
        st.video(remote_url(item["clip_path"]))
    except Exception as exc:
        st.error(f"Video could not be displayed: {exc}")
        ready = False
with right:
    st.markdown("### Caption Records")
    st.caption("Red bold = overlaps with video. Scroll to view all captions.")
    with st.container(height=350, border=True):
        try:
            captions = load_captions(str(item["caption_path"]))
            display_captions(captions, clip_start, clip_end)
        except Exception as exc:
            st.error(f"Caption loading failed: {exc}")
            ready = False

st.divider()
st.markdown("### Does this video match the corresponding captions?")
existing = records.loc[records["video_id"] == selected_id]
old_label = existing.iloc[0]["label"] if not existing.empty else None
with st.form(f"form_{role}_{selected_id}"):
    label = st.radio("Please choose a label", LABELS, index=LABELS.index(old_label) if old_label in LABELS else None, horizontal=True)
    submitted = st.form_submit_button("Save and Continue →", type="primary", disabled=not ready)
if submitted:
    if label is None:
        st.warning("Please select a label.")
    else:
        try:
            save_annotation(selected_id, role, label)
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to save annotation: {exc}")
