import json
import os
import pickle
from pathlib import Path
from typing import List, Dict

MAX_WORDS = 200


def preprocess_all(transcripts_dir: str) -> List[Dict]:
    """
    Read all transcript JSON files and convert them into clean chunks.
    """
    all_chunks = []

    for path in Path(transcripts_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_chunks = build_chunks_from_segments(data)
        all_chunks.extend(video_chunks)

    return all_chunks


def build_chunks_from_segments(video: Dict) -> List[Dict]:
    chunks = []

    segments = video.get("segments", [])
    if not segments:
        return chunks

    buffer = []
    word_count = 0
    chunk_index = 0

    start_sec = segments[0]["start_sec"]
    start_hhmmss = segments[0]["start_hhmmss"]

    for seg in segments:
        text = seg.get("text_roman", "").strip()
        if not text:
            continue

        words = text.split()

        if word_count + len(words) <= MAX_WORDS:
            buffer.append(text)
            word_count += len(words)
            end_sec = seg["end_sec"]
            end_hhmmss = seg["end_hhmmss"]
        else:
            chunks.append(
                make_chunk(
                    video,
                    buffer,
                    chunk_index,
                    start_sec,
                    end_sec,
                    start_hhmmss,
                    end_hhmmss,
                    seg.get("play_url")
                )
            )

            chunk_index += 1
            buffer = [text]
            word_count = len(words)
            start_sec = seg["start_sec"]
            start_hhmmss = seg["start_hhmmss"]
            end_sec = seg["end_sec"]
            end_hhmmss = seg["end_hhmmss"]

    if buffer:
        chunks.append(
            make_chunk(
                video,
                buffer,
                chunk_index,
                start_sec,
                end_sec,
                start_hhmmss,
                end_hhmmss,
                segments[-1].get("play_url")
            )
        )

    return chunks


def make_chunk(
    video: Dict,
    buffer: List[str],
    index: int,
    start_sec: int,
    end_sec: int,
    start_hhmmss: str,
    end_hhmmss: str,
    play_url: str
) -> Dict:
    return {
        "chunk_id": f"{video['video_id']}_{index:04d}",
        "video_id": video["video_id"],
        "title": video.get("title"),
        "playlist_id": video.get("playlist_id"),
        "chunk_index": index,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "start_hhmmss": start_hhmmss,
        "end_hhmmss": end_hhmmss,
        "text_roman": " ".join(buffer),
        "play_url": play_url
    }


# =========================
# SCRIPT ENTRY POINT
# =========================
if __name__ == "__main__":
    TRANSCRIPTS_DIR = "data/transcripts"

    print("Preprocessing transcripts...")
    all_chunks = preprocess_all(TRANSCRIPTS_DIR)

    print(f"Total chunks created: {len(all_chunks)}")

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Chunks saved to data/processed/chunks.pkl")
