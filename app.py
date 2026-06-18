"""
スポーツ記事 固有名詞チェッカー
名簿データと照合して記事内の固有名詞の誤りを検出します。
"""

import json
import html
import re
import os
from io import StringIO

import streamlit as st
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="記事チェッカー",
    page_icon="📰",
    layout="wide",
)

# ════════════════════════════════════════════════
# 名簿パーサー
# ════════════════════════════════════════════════

GRADE_CHARS = set("１２３４５６123456①②③④⑤⑥")
ROLE_KANTOUKU = {"監", "監督"}
ROLE_COACH = {"コ", "コーチ"}


def parse_roster_text(text: str) -> list:
    """
    テキスト形式の名簿を解析。
    フォーマット例:
        チーム名
        （地域名）
        氏名　監  or  氏名　コ
        氏名　６
    """
    roster = []
    current_team = None
    current_area = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 地域行: （xxx） or (xxx)
        m = re.fullmatch(r"[（(](.+)[)）]", line)
        if m:
            current_area = m.group(1)
            continue

        # 全角・半角スペースで分割
        parts = re.split(r"[\s　]+", line)
        last = parts[-1] if parts else ""

        if len(parts) >= 2 and current_team is not None:
            if last in ROLE_KANTOUKU:
                roster.append({
                    "チーム名": current_team,
                    "地域": current_area or "",
                    "氏名": "".join(parts[:-1]),
                    "役職": "監督",
                    "学年": "",
                })
                continue
            if last in ROLE_COACH:
                roster.append({
                    "チーム名": current_team,
                    "地域": current_area or "",
                    "氏名": "".join(parts[:-1]),
                    "役職": "コーチ",
                    "学年": "",
                })
                continue
            if last in GRADE_CHARS:
                roster.append({
                    "チーム名": current_team,
                    "地域": current_area or "",
                    "氏名": "".join(parts[:-1]),
                    "役職": "選手",
                    "学年": last,
                })
                continue

        # チーム名行
        current_team = line

    return roster


def parse_roster_csv(source) -> list:
    """CSV（ファイルまたはテキスト）の名簿を解析"""
    try:
        if hasattr(source, "read"):
            df = pd.read_csv(source, dtype=str)
        else:
            df = pd.read_csv(StringIO(source), dtype=str)
        df = df.fillna("")
        return df.to_dict("records")
    except Exception as exc:
        st.error(f"CSVの読み込みエラー: {exc}")
        return []


# ════════════════════════════════════════════════
# Claude API による照合
# ════════════════════════════════════════════════

SYSTEM_PROMPT = """\
あなたはスポーツ記事の校正アシスタントです。
提供された名簿と記事を照合し、固有名詞の誤りを検出します。
応答は必ずJSON形式のみで返してください。余計な説明文は不要です。"""

USER_PROMPT_TMPL = """\
## 名簿データ（JSON）
```json
{roster}
```

## 記事テキスト
```
{article}
```

## 照合タスク
記事テキスト内に登場する以下の固有名詞を**全て**抽出し、名簿と照合してください。
- 人名（選手名・監督名・コーチ名）
- チーム名
- 地域名（市名・町村名）
- 選手に紐づく学年（記事内で学年が言及されている場合）

## 注意事項
- 名簿に登録されていない大会来賓・連盟役員・記者名などは対象外です（status="scope_out"）。
- "text" フィールドは記事内に**そのまま存在する文字列**としてください（ハイライト検索に使用します）。
- 同じ固有名詞が複数回出現する場合、最初の1回だけを報告してください。

## 出力フォーマット（JSONのみ）
{{
  "entities": [
    {{
      "text": "記事内の表記（完全一致する文字列）",
      "type": "人名|チーム名|地域名|学年",
      "status": "ok|mismatch|not_found|warning|scope_out",
      "suggestion": "修正候補（不一致時のみ、なければ省略）",
      "issue": "問題の説明（不一致時のみ、なければ省略）"
    }}
  ],
  "summary": {{
    "total": 0,
    "ok": 0,
    "mismatch": 0,
    "not_found": 0,
    "warning": 0
  }}
}}

statusの意味：
- ok        : 名簿と一致、または問題なし
- mismatch  : 名簿に類似エントリがあるが表記が異なる（漢字誤り等）
- not_found : 名簿に該当エントリが見当たらない
- warning   : 類似するものがあるが確信が持てない
- scope_out : 名簿対象外（集計・ハイライトから除外）
"""


def check_with_claude(article: str, roster: list, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)

    roster_json = json.dumps(roster, ensure_ascii=False, indent=2)
    user_msg = USER_PROMPT_TMPL.format(roster=roster_json, article=article)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    # Markdown コードブロック除去
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    return json.loads(raw.strip())


# ════════════════════════════════════════════════
# ハイライト HTML 生成
# ════════════════════════════════════════════════

COLORS = {
    "mismatch":  "#ff9999",
    "not_found": "#ffcc88",
    "warning":   "#ffff88",
}
LABELS = {
    "mismatch":  "誤字",
    "not_found": "名簿なし",
    "warning":   "要確認",
}


def build_html(article: str, entities: list) -> str:
    issues = [e for e in entities if e.get("status") in COLORS]

    # テキスト位置を特定
    annotations = []
    seen_texts = set()
    for ent in issues:
        target = ent.get("text", "")
        if not target or target in seen_texts:
            continue
        seen_texts.add(target)
        pos = 0
        while True:
            idx = article.find(target, pos)
            if idx == -1:
                break
            annotations.append({"start": idx, "end": idx + len(target), "ent": ent})
            pos = idx + 1

    annotations.sort(key=lambda a: a["start"])

    # 重複範囲を除去
    merged = []
    for ann in annotations:
        if merged and ann["start"] < merged[-1]["end"]:
            continue
        merged.append(ann)

    def escape(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

    # HTML 組み立て
    parts = [
        '<div style="font-size:0.95rem;line-height:1.9;font-family:sans-serif;'
        'background:#fafafa;padding:20px;border-radius:8px;border:1px solid #ddd">'
    ]
    cursor = 0
    for ann in merged:
        parts.append(escape(article[cursor:ann["start"]]))
        ent = ann["ent"]
        color = COLORS.get(ent["status"], "#eee")
        label = LABELS.get(ent["status"], "?")
        tip = ent.get("issue", "")
        if ent.get("suggestion"):
            tip += f"　→ 候補: {ent['suggestion']}"
        parts.append(
            f'<mark style="background:{color};padding:1px 4px;border-radius:3px;'
            f'cursor:help" title="{html.escape(tip)}">'
            f"{html.escape(article[ann['start']:ann['end']])}"
            f'<sup style="font-size:0.65em;vertical-align:super;color:#333">'
            f"{html.escape(label)}</sup></mark>"
        )
        cursor = ann["end"]
    parts.append(escape(article[cursor:]))
    parts.append("</div>")
    return "".join(parts)


# ════════════════════════════════════════════════
# CSV テンプレート
# ════════════════════════════════════════════════

CSV_TEMPLATE = (
    "チーム名,地域,氏名,役職,学年\n"
    "基山ジュニア,基山町,平野芳継,監督,\n"
    "基山ジュニア,基山町,竹村悠希,選手,6\n"
    "基山ジュニア,基山町,竹田幸司,選手,4\n"
)

TEXT_TEMPLATE = (
    "基山ジュニア\n（基山町）\n平野　芳継　監\n竹村　悠希　６\n竹田　幸司　４\n"
)


# ════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════


def main():
    st.title("📰 記事固有名詞チェッカー")
    st.caption("名簿データと記事を照合し、人名・チーム名・地域名の誤りを検出します。")

    # ── API キー ────────────────────────────────
    # .env（ローカル）とStreamlit Secrets（クラウド）の両方に対応
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            api_key = ""
    if not api_key:
        with st.sidebar:
            api_key = st.text_input(
                "🔑 Anthropic APIキー",
                type="password",
                help=".envファイルにANTHROPIC_API_KEYを設定すると毎回入力不要です。",
            )

    # ── サイドバー: 名簿 ─────────────────────────
    with st.sidebar:
        st.header("📋 名簿の読み込み")

        with st.expander("CSVテンプレートを確認・ダウンロード"):
            st.code(CSV_TEMPLATE, language="text")
            st.download_button(
                "⬇️ CSVテンプレートをダウンロード",
                data=CSV_TEMPLATE.encode("utf-8-sig"),
                file_name="名簿テンプレート.csv",
                mime="text/csv",
            )

        roster_mode = st.radio("入力方式", ["CSVファイル", "テキスト貼り付け"], horizontal=True)
        roster = []

        if roster_mode == "CSVファイル":
            uploaded = st.file_uploader(
                "名簿CSVをアップロード",
                type=["csv"],
                help="文字コードはUTF-8またはShift-JIS（BOM付き推奨）",
            )
            if uploaded:
                roster = parse_roster_csv(uploaded)
        else:
            t = st.text_area(
                "名簿テキストを貼り付け",
                height=280,
                placeholder=TEXT_TEMPLATE,
            )
            if t.strip():
                roster = parse_roster_text(t)

        if roster:
            st.success(f"✅ {len(roster)} 件読み込み完了")
            with st.expander("名簿プレビュー（先頭10件）"):
                st.dataframe(
                    pd.DataFrame(roster).head(10),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("名簿を読み込んでください。")

    # ── メインエリア ─────────────────────────────
    article = st.text_area(
        "記事テキストを貼り付けてください",
        height=260,
        placeholder="ここに記事本文を貼り付け…",
    )

    ready = bool(article.strip() and roster and api_key)
    run = st.button(
        "✅ チェック実行",
        type="primary",
        disabled=not ready,
        help="名簿・記事・APIキーをすべて入力してから実行してください。",
    )

    if run:
        st.session_state.pop("result", None)
        with st.spinner("AIが照合中です…（数秒〜十数秒かかります）"):
            try:
                st.session_state["result"] = check_with_claude(
                    article.strip(), roster, api_key
                )
                st.session_state["article_saved"] = article.strip()
            except anthropic.AuthenticationError:
                st.error("APIキーが無効です。正しいキーを入力してください。")
            except json.JSONDecodeError as exc:
                st.error(f"AIの応答を解析できませんでした: {exc}")
            except Exception as exc:
                st.error(f"エラーが発生しました: {exc}")

    # ── 結果表示 ──────────────────────────────────
    if "result" in st.session_state:
        result = st.session_state["result"]
        saved_article = st.session_state.get("article_saved", "")
        entities = result.get("entities", [])
        summary = result.get("summary", {})

        # scope_out を除外してサマリー集計
        active = [e for e in entities if e.get("status") != "scope_out"]

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("チェック数", summary.get("total", len(active)))
        c2.metric("✅ 一致", summary.get("ok", 0))
        c3.metric("❌ 誤字疑い", summary.get("mismatch", 0))
        c4.metric(
            "⚠️ 要確認",
            summary.get("not_found", 0) + summary.get("warning", 0),
        )

        # ハイライト表示
        st.subheader("記事（チェック結果）")
        st.caption("マーカーにマウスを重ねると詳細が表示されます。")
        st.html(build_html(saved_article, active))

        # 不一致リスト
        issues = [e for e in active if e.get("status") != "ok"]
        st.divider()
        if issues:
            st.subheader(f"⚠️ 不一致・要確認リスト（{len(issues)} 件）")
            for ent in issues:
                color = COLORS.get(ent.get("status", ""), "#eee")
                label = LABELS.get(ent.get("status", ""), "?")
                c1, c2, c3, c4 = st.columns([1.2, 2, 2, 4])
                c1.markdown(
                    f'<span style="background:{color};padding:2px 10px;'
                    f'border-radius:4px;font-size:0.85em">{label}</span>',
                    unsafe_allow_html=True,
                )
                c2.markdown(f"**{ent.get('text', '')}**")
                c3.markdown(f"→ {ent.get('suggestion') or '—'}")
                c4.markdown(ent.get("issue") or "")
        else:
            st.success("✅ 問題は見つかりませんでした。")


if __name__ == "__main__":
    main()
