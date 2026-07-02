"""
スポーツ記事 固有名詞チェッカー
名簿データと照合して記事内の固有名詞の誤りを検出します。
"""

import json
import html
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
# 名簿パーサー（Claude API / Tool Use）
# ════════════════════════════════════════════════

ROSTER_TOOL = {
    "name": "save_roster",
    "description": "名簿テキストから抽出した全メンバーを保存する",
    "input_schema": {
        "type": "object",
        "properties": {
            "members": {
                "type": "array",
                "description": "全メンバーのリスト",
                "items": {
                    "type": "object",
                    "description": (
                        "1名分のデータ。チーム名と氏名は必須。"
                        "学年・役職・ポジション・背番号など名簿に存在する情報は"
                        "日本語フィールド名で追加する。"
                    ),
                    "properties": {
                        "チーム名": {"type": "string"},
                        "氏名": {"type": "string"},
                    },
                    "required": ["チーム名", "氏名"],
                    "additionalProperties": True,
                },
            }
        },
        "required": ["members"],
    },
}

ROSTER_PARSE_USER_PREFIX = """\
以下の名簿テキストから全員の情報を抽出し、save_roster ツールを呼び出してください。

抽出ルール：
- 「チーム名」と「氏名」は必須フィールドです
- テキストに存在するその他の情報（学年、ポジション、役職、背番号、地域名など）も全て含めてください
- フィールド名は日本語で、内容に合わせて自然な名称にしてください
- チーム名が記載されていない行は文脈から推測してください
- 氏名は姓名を結合した完全な表記にしてください

名簿テキスト：
"""


def parse_roster_with_claude(text: str, api_key: str) -> list:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        tools=[ROSTER_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": ROSTER_PARSE_USER_PREFIX + text}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "save_roster":
            return block.input.get("members", [])
    return []


def parse_roster_csv(source) -> list:
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
# Claude API による照合（Tool Use）
# ════════════════════════════════════════════════

CHECK_TOOL = {
    "name": "save_check_result",
    "description": "記事の固有名詞照合結果を保存する",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text":       {"type": "string", "description": "記事内の表記（完全一致する文字列）"},
                        "type":       {"type": "string", "enum": ["人名", "チーム名", "地域名", "その他"]},
                        "status":     {"type": "string", "enum": ["ok", "mismatch", "not_found", "warning", "scope_out"]},
                        "suggestion": {"type": "string"},
                        "issue":      {"type": "string"},
                    },
                    "required": ["text", "type", "status"],
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "total":     {"type": "integer"},
                    "ok":        {"type": "integer"},
                    "mismatch":  {"type": "integer"},
                    "not_found": {"type": "integer"},
                    "warning":   {"type": "integer"},
                },
                "required": ["total", "ok", "mismatch", "not_found", "warning"],
            },
        },
        "required": ["entities", "summary"],
    },
}

CHECK_USER_ROSTER_HEADER = """\
以下の名簿と記事を照合し、save_check_result ツールで結果を報告してください。

## 名簿データ（JSON）
名簿のフィールドは入力によって異なります。「チーム名」と「氏名」は共通フィールドです。
それ以外（学年・役職・ポジション・背番号・地域など）は名簿に含まれる場合のみ存在します。
"""

CHECK_USER_ARTICLE_HEADER = """

## 記事テキスト
"""

CHECK_USER_SUFFIX = """

## 照合タスク
記事テキスト内に登場する固有名詞を**全て**抽出し、名簿と照合してください。
- 人名（選手名・監督名・コーチ名・その他スタッフ名）
- チーム名
- 名簿に含まれるその他の固有名詞（地域名・所属機関名など）
- 名簿に属性情報（学年・ポジション・背番号など）がある場合、記事内で言及されていれば整合性も確認

## 注意事項
- 名簿に登録されていない大会来賓・連盟役員・記者名などは対象外（status="scope_out"）
- text フィールドは記事内に**そのまま存在する文字列**にしてください（ハイライト検索に使用します）
- 同じ固有名詞が複数回出現する場合、最初の1回だけを報告してください

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
    user_msg = (
        CHECK_USER_ROSTER_HEADER
        + roster_json
        + CHECK_USER_ARTICLE_HEADER
        + article
        + CHECK_USER_SUFFIX
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        tools=[CHECK_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "save_check_result":
            return block.input
    return {"entities": [], "summary": {"total": 0, "ok": 0, "mismatch": 0, "not_found": 0, "warning": 0}}


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
    merged = []
    for ann in annotations:
        if merged and ann["start"] < merged[-1]["end"]:
            continue
        merged.append(ann)

    def escape(text: str) -> str:
        return html.escape(text).replace("\n", "<br>")

    parts = [
        '<div style="font-size:0.95rem;line-height:1.9;font-family:sans-serif;'
        'background:#fafafa;color:#1a1a1a;padding:20px;border-radius:8px;border:1px solid #ddd">'
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


# ════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════


def main():
    st.title("📰 記事固有名詞チェッカー")
    st.caption("名簿データと記事を照合し、人名・チーム名などの誤りを検出します。")

    # ── サイドバー（APIキー＋名簿）─────────────────
    with st.sidebar:
        # API キー
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            try:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
            except Exception:
                api_key = ""
        if not api_key:
            api_key = st.text_input(
                "🔑 Anthropic APIキー",
                type="password",
                key="api_key_input",
                help=".envファイルにANTHROPIC_API_KEYを設定すると毎回入力不要です。",
            )

        st.divider()
        st.header("📋 名簿の読み込み")

        roster_mode = st.radio(
            "入力方式", ["CSVファイル", "テキスト貼り付け"],
            horizontal=True,
            key="roster_mode",
        )

        # モード切替時に前回のテキスト解析結果をクリア
        if st.session_state.get("_roster_mode") != roster_mode:
            st.session_state.pop("roster_parsed", None)
            st.session_state["_roster_mode"] = roster_mode

        roster = []

        if roster_mode == "CSVファイル":
            with st.expander("CSVテンプレートを確認・ダウンロード"):
                st.code(CSV_TEMPLATE, language="text")
                st.download_button(
                    "⬇️ CSVテンプレートをダウンロード",
                    data=CSV_TEMPLATE.encode("utf-8-sig"),
                    file_name="名簿テンプレート.csv",
                    mime="text/csv",
                )
            uploaded_files = st.file_uploader(
                "名簿CSVをアップロード（複数可）",
                type=["csv"],
                accept_multiple_files=True,
                help="複数チームのCSVをまとめて選択できます。文字コードはUTF-8またはShift-JIS（BOM付き推奨）",
            )
            if uploaded_files:
                for f in uploaded_files:
                    roster.extend(parse_roster_csv(f))

        else:  # テキスト貼り付け
            roster_text = st.text_area(
                "名簿テキストを貼り付け",
                height=240,
                placeholder=(
                    "どんな形式でも対応します。例：\n\n"
                    "佐賀北高校\n山田 太郎 3年 監督\n鈴木 一郎 2年 ショート\n\n"
                    "チーム名：有田工業\n氏名：田中花子 / 役職：主将 / 学年：3年"
                ),
            )

            if not api_key:
                st.caption("⚠️ テキスト解析にはAPIキーが必要です。")

            parse_clicked = st.button(
                "🔍 AIで解析",
                disabled=not (roster_text.strip() and api_key),
                use_container_width=True,
                key="parse_roster_btn",
            )
            if parse_clicked:
                with st.spinner("名簿を解析中…"):
                    try:
                        parsed = parse_roster_with_claude(roster_text.strip(), api_key)
                        st.session_state["roster_parsed"] = parsed
                    except anthropic.AuthenticationError:
                        st.error("APIキーが無効です。")
                    except json.JSONDecodeError as exc:
                        st.error(f"解析結果を読み取れませんでした: {exc}")
                    except Exception as exc:
                        st.error(f"解析エラー: {exc}")

            roster = st.session_state.get("roster_parsed", [])

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

        st.subheader("記事（チェック結果）")
        st.caption("マーカーにマウスを重ねると詳細が表示されます。")
        st.html(build_html(saved_article, active))

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
