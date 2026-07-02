"""
Parse Combined_Conversations.md (10 full reference transcripts) into
structured eval fixtures.

These traces are full dialogues, not a bare "persona + fact sheet"
format -- so we derive what we need from them:
  - `persona_facts`: every User turn, in order. This is the script a
    simulated-user LLM (in eval/run_eval.py) draws on when it plays this
    persona against our own /chat -- exactly mirroring how SHL describes
    their own grading harness ("simulates a user... given the trace's
    persona and facts").
  - `expected_shortlist`: the recommendation table attached to the FINAL
    turn (the one with end_of_conversation: true) -- our Recall@10
    ground truth.
  - `full_transcript`: every turn's user/agent/recommendations/eoc,
    kept for reference and for spot-checking that our own agent lands in
    a similar place, not just the same final answer.

Run: python3 scripts/parse_traces.py
Reads:  <source markdown, path given via --source, default Combined_Conversations.md>
Writes: eval/traces/conversation_<N>.json (one file per trace)
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_table(table_text: str):
    """Markdown table -> list of {name, test_type, url}. Duration/Keys/
    Languages columns exist for human readability in the source doc but
    aren't part of our API schema, so we don't carry them into the
    eval fixture."""
    rows = []
    lines = [l for l in table_text.strip().splitlines() if l.strip().startswith("|")]
    for line in lines[2:]:  # skip header + separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue
        _, name, test_type, _keys, _duration, _languages, url_cell = cells[:7]
        url_match = re.search(r"<(https?://[^>]+)>", url_cell)
        url = url_match.group(1) if url_match else url_cell.strip()
        rows.append({"name": name.strip(), "test_type": test_type.strip(), "url": url})
    return rows


def parse_turn(turn_text: str):
    user_match = re.search(r"\*\*User\*\*\s*\n((?:^>.*\n?)+)", turn_text, re.MULTILINE)
    user_lines = user_match.group(1) if user_match else ""
    user_msg = "\n".join(
        l.lstrip(">").strip() for l in user_lines.splitlines() if l.strip().startswith(">")
    ).strip()

    agent_match = re.search(r"\*\*Agent\*\*\s*\n(.*?)(?=\n_`end_of_conversation`)", turn_text, re.DOTALL)
    agent_block = agent_match.group(1).strip() if agent_match else ""

    table_match = re.search(r"(\|.*\|(?:\n\|.*\|)+)", agent_block, re.DOTALL)
    recommendations = parse_table(table_match.group(1)) if table_match else []
    reply_text = agent_block[: table_match.start()].strip() if table_match else agent_block
    # drop the boilerplate "_No recommendations this turn..._" note from the reply text
    reply_text = re.sub(r"_No recommendations this turn.*?_\.?", "", reply_text).strip()

    eoc_match = re.search(r"_`end_of_conversation`:\s*\*\*(true|false)\*\*_", turn_text)
    end_of_conversation = eoc_match.group(1) == "true" if eoc_match else False

    return {
        "user": user_msg,
        "agent_reply": reply_text,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation,
    }


def parse_conversation(block_text: str):
    turn_blocks = re.split(r"### Turn \d+\s*\n", block_text)[1:]  # [0] is the header/preamble
    return [parse_turn(t) for t in turn_blocks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT.parent / "Combined_Conversations.md"))
    args = ap.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        # fall back to the uploads copy used during development
        source_path = Path("/mnt/user-data/uploads/Combined_Conversations.md")
    text = source_path.read_text(encoding="utf-8")

    blocks = re.split(r"^# Conversation (\d+)\s*$", text, flags=re.MULTILINE)[1:]
    # re.split with a capturing group interleaves [num, block, num, block, ...]
    conversations = list(zip(blocks[0::2], blocks[1::2]))

    out_dir = ROOT / "eval" / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for conv_num, block in conversations:
        turns = parse_conversation(block)
        persona_facts = [t["user"] for t in turns]
        final_turn = turns[-1]
        fixture = {
            "conversation_id": int(conv_num),
            "persona_facts": persona_facts,
            "expected_shortlist": final_turn["recommendations"],
            "num_turns_in_reference": len(turns),
            "full_transcript": turns,
        }
        out_path = out_dir / f"conversation_{conv_num}.json"
        out_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append((int(conv_num), len(turns), len(final_turn["recommendations"]), final_turn["end_of_conversation"]))

    print(f"Parsed {len(conversations)} conversations -> {out_dir}/")
    print(f"\n{'conv':>5} {'turns':>6} {'shortlist_size':>15} {'ends_true':>10}")
    for conv_id, n_turns, shortlist_size, ends_true in sorted(summary):
        flag = "OK" if ends_true else "!! missing final end_of_conversation=true"
        print(f"{conv_id:>5} {n_turns:>6} {shortlist_size:>15} {flag:>10}")


if __name__ == "__main__":
    main()
