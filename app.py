"""
スポーツ記事 固有名詞チェッカー
名簿データと照合して記事内の固有名詞の誤りを検出します。
"""

import base64
import json
import html
import os
from datetime import datetime, timedelta, timezone
from io import StringIO

import streamlit as st
import anthropic
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader

JST = timezone(timedelta(hours=9))

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
        model="claude-sonnet-4-6",
        max_tokens=16000,
        tools=[ROSTER_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": ROSTER_PARSE_USER_PREFIX + text}],
    )
    if response.stop_reason == "max_tokens":
        st.warning("⚠️ 名簿が長すぎて途中で打ち切られました。テキストを分割して再解析してください。")
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


def extract_text_from_pdf(source) -> str:
    try:
        reader = PdfReader(source)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception as exc:
        st.error(f"PDFの読み込みエラー: {exc}")
        return ""


def build_reference_docs(uploaded_files) -> list:
    """アップロードされた参考資料PDFをClaudeに渡すdocumentブロックの元データに変換する"""
    docs = []
    for f in uploaded_files or []:
        try:
            data = f.read()
        except Exception as exc:
            st.error(f"「{f.name}」の読み込みエラー: {exc}")
            continue
        docs.append({
            "title": f.name,
            "data": base64.standard_b64encode(data).decode("utf-8"),
        })
    return docs


def reference_docs_to_blocks(reference_docs: list) -> list:
    return [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": doc["data"],
            },
            "title": doc["title"],
        }
        for doc in reference_docs
    ]


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

CHECK_USER_REFERENCE_NOTE = """

## 添付の参考資料
上記の名簿に加えて、参考資料としてPDFを添付しています。名簿だけでは判断できない固有名詞（大会名・施設名・地域名など）や
属性情報は、この参考資料の記載を優先的な根拠として照合に使用してください。
"""


def check_with_claude(article: str, roster: list, api_key: str, reference_docs: list = None) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    roster_json = json.dumps(roster, ensure_ascii=False, indent=2)
    user_text = (
        CHECK_USER_ROSTER_HEADER
        + roster_json
        + CHECK_USER_ARTICLE_HEADER
        + article
        + CHECK_USER_SUFFIX
    )
    if reference_docs:
        user_text += CHECK_USER_REFERENCE_NOTE
        content = reference_docs_to_blocks(reference_docs) + [{"type": "text", "text": user_text}]
    else:
        content = user_text
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        tools=[CHECK_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "save_check_result":
            return block.input
    return {"entities": [], "summary": {"total": 0, "ok": 0, "mismatch": 0, "not_found": 0, "warning": 0}}


# ════════════════════════════════════════════════
# AI校閲（Claude API / web_search ツール）
# ════════════════════════════════════════════════

KOUETSU_SYSTEM_PROMPT = """\
## 1. 役割と目的

- 「記事校正・校閲の高度専門家」として、ニュース記事のプロフェッショナルな日本語テキストを徹底的に分析し、その品質を極限まで高める役割を担う。
- 語彙の正確性、文法の厳密さ、論理的整合性、事実の正確性の4点を軸に、読み手に誤解を与えない洗練された文章へと昇華させることを最終目標とする。


## 2. 出力フォーマット

### 2.1 全体構成

1. 冒頭で必ず「指摘事項の簡単なまとめ」を述べ、修正の全体像を箇条書きでコンパクトに提示する。
2. 続いて、2.3節の観点に沿った詳細な分析を行う。

### 2.2 指摘・提案の書き方

1. 明確な誤用や改善案:【提案】として、該当箇所の引用と修正案を対比させて提示する。
2. 執筆者の意図を測りかねる箇所:【要確認】として、具体的に何を確認すべきか記述する。
   - 例: かぎ括弧で囲まれた言葉などは、取材メモや資料に基づき記者が意図的に選択した表現である可能性を考慮し、配慮した表現で指摘する。

### 2.3 分析観点

#### 2.3.1 語彙・語義
文脈において最も適切な言葉が選ばれているかを評価する。
- 例:「参画」「参加」「従事」の使い分け

#### 2.3.2 助詞・接続表現
主語・述語の呼応、格助詞、係り受けの曖昧さ、前後の文脈との論理的整合性、および読点(、)の配置の妥当性を確認する。

#### 2.3.3 事実確認
重点チェック項目は以下の通り。

**(1) 人名・固有名詞の照合(最重要)**
- 記事中の人名・団体名・会社名・地名・施設名・催事名が、インターネット検索の結果の表記と完全に一致しているか
- 同音・類似の名前の取り違えがないか
  - 例:「真央」と「麻央」
- 敬称・肩書が正確か
- 読み仮名(ふりがな)が正確か

**(2) 数値・日付・時刻・単位の整合性**
- 日付・曜日・時刻(午前/午後)が本文の他の記述やインターネット検索の結果と矛盾しないか
- 単位(円/万円、トン/キロ、ミリ/センチ等)や時刻の午前/午後の記載が常識の範囲内か
- 前後関係・順序が逆転していないか
- 割合・金額・点数などの計算結果が正しいか(単純な数値だけでなく、そこから導かれる「率」や「差」も再計算して検算する)
- 電話番号とファクス番号など、似た性質の番号の取り違えがないか

**(3) 時制・時系列の整合性(日本標準時基準)**
- 校閲開始時にまず日本標準時(JST)における現在の日時を把握したうえで、記事中の時制表現を検証する
- 相対的な時間表現が、現在日時と矛盾していないか
  - 例:「昨日」「来週」「今年度」「まもなく」「〜する予定」等について、未来の出来事が過去形で断定されていないか、既に経過しているはずの日付が「今後」として記述されていないか
- 記事中に複数の日時・期日が登場する場合、それらの前後関係が本文の論理展開(発生→報道、予定→実施等)と整合しているか
- 「開催予定」「開催された」など、時制を伴う動詞の使い分けが、現在日時を基準として適切か

**(4) 主体・関係性の確認**
- 「誰が」「どの組織が」その行為・発言・決定の主体かを本文全体で一貫して確認する
  - 例: 県 vs 市、A署 vs B署、国の機関 vs 自治体
- 罪名・条例名・制度名など、法的・制度的な名称が正確か(正式名である必要はないが略称は正確か)
- 「AがBに委託」「AがBを主催」等の関係性が逆になっていないか

**(5) 取材内容の言い換え・要約の妥当性**
- 記者が取材メモを要約・言い換えた表現が、事実の程度や有無を変えてしまっていないか
  - 例:「暴言」の事実を「暴言や暴力」と拡大していないか、「無償化」等の断定表現に検証の余地がないか
- 断定的な表現(「〜のみ」「〜していない」等)は、根拠が原稿内に明示されているか確認する

**(6) 表記・同音異義語**
- 同音異義語の誤変換がないか
  - 例:「広報」と「公報」
- 用語の使われ方に、インターネット検索で見つけた過去記事との食い違いや、原義に基づく不自然さがないか
  - 例:「のぼり」と「吹き流し」


## 3. 例外処理・判断保留の原則

- システム上の読み取りエラーの可能性: 客観的な文法規則を最優先しつつ、文字コードなどシステム上の読み取りエラーの可能性がある箇所は、修正提案を避けてユーザーへの注意喚起に留める。
- 時制・時系列の判断保留: 時制・時系列の確認にあたっては、必ず現在のJSTを最初に確認したうえで判断根拠とする。断定が難しい場合(例: 記事の初出時期や配信予定日が不明な場合)は【要確認】として扱う。


## 4. トーン制御

- 通常時: 冷静、分析的、かつ誠実なプロフェッショナルとして、「です・ます」調で回答する。常に根拠に基づいた論理的な説明を提供する。
- 特例(原稿が完璧で指摘事項が一つもない場合のみ): 2文以内で「普段の冷静さを失った過剰な褒め言葉」を述べ、速やかに理性的な専門家のトーンに戻る。
"""


def build_kouetsu_user_message(article: str, has_reference_docs: bool = False) -> str:
    now_jst = datetime.now(JST).strftime("%Y年%m月%d日(%a) %H:%M JST")
    text = (
        f"現在の日本標準時(JST): {now_jst}\n"
        "上記の現在日時を基準に、時制・時系列の整合性を判断してください。\n\n"
        "## 記事原稿\n"
        f"{article}"
    )
    if has_reference_docs:
        text += (
            "\n\n## 添付の参考資料\n"
            "記事に関連する参考資料（発表資料・過去記事等）のPDFを添付しています。"
            "事実確認（人名・数値・日付・主体関係など）では、この参考資料を"
            "インターネット検索より優先的な一次情報として扱ってください。"
        )
    return text


def run_kouetsu(article: str, api_key: str, reference_docs: list = None) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    request_kwargs = dict(
        model="claude-opus-5",
        max_tokens=12000,
        system=KOUETSU_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
        output_config={"effort": "high"},
    )
    user_text = build_kouetsu_user_message(article, has_reference_docs=bool(reference_docs))
    if reference_docs:
        user_content = reference_docs_to_blocks(reference_docs) + [{"type": "text", "text": user_text}]
    else:
        user_content = user_text
    messages = [{"role": "user", "content": user_content}]
    response = client.messages.create(messages=messages, **request_kwargs)

    # サーバー側ツール(web_search)の反復上限に達した場合は1回だけ再送して継続させる
    if response.stop_reason == "pause_turn":
        messages.append({"role": "assistant", "content": response.content})
        response = client.messages.create(messages=messages, **request_kwargs)

    if response.stop_reason == "refusal":
        return {"refusal": True, "text": ""}

    text = "".join(block.text for block in response.content if block.type == "text")
    return {"refusal": False, "text": text}


# ════════════════════════════════════════════════
# ハイライト HTML 生成
# ════════════════════════════════════════════════

COLORS = {
    "ok":        "#a8e6a3",
    "mismatch":  "#ff9999",
    "not_found": "#ffcc88",
    "warning":   "#ffff88",
}
LABELS = {
    "ok":        "一致",
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
        if not tip and ent["status"] == "ok":
            tip = "名簿と一致"
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

        st.divider()
        st.header("📎 参考資料（任意）")
        st.caption("発表資料・過去記事などのPDFを添付すると、固有名詞チェック・AI校閲の両方で事実確認の根拠として使用します。")
        reference_files = st.file_uploader(
            "参考資料PDFをアップロード（複数可）",
            type=["pdf"],
            accept_multiple_files=True,
            key="reference_pdf_uploader",
        )
        reference_docs = build_reference_docs(reference_files)
        if reference_docs:
            st.success(f"✅ 参考資料 {len(reference_docs)} 件添付済み")

    # ── メインエリア：記事入力 ───────────────────────
    st.subheader("📄 記事の入力")
    article_mode = st.radio(
        "入力方式", ["テキスト貼り付け", "PDFファイル"],
        horizontal=True,
        key="article_mode",
    )

    if st.session_state.get("_article_mode") != article_mode:
        st.session_state.pop("article_pdf_text", None)
        st.session_state["_article_mode"] = article_mode

    if article_mode == "テキスト貼り付け":
        article = st.text_area(
            "記事テキストを貼り付けてください",
            height=260,
            placeholder="ここに記事本文を貼り付け…",
            key="article_text_input",
        )
    else:
        uploaded_pdf = st.file_uploader("記事PDFをアップロード", type=["pdf"])
        if uploaded_pdf is not None and st.session_state.get("_article_pdf_name") != uploaded_pdf.name:
            with st.spinner("PDFからテキストを抽出中…"):
                st.session_state["article_pdf_text"] = extract_text_from_pdf(uploaded_pdf)
            st.session_state["_article_pdf_name"] = uploaded_pdf.name

        article = st.text_area(
            "抽出したテキスト（必要に応じて編集してください）",
            height=260,
            key="article_pdf_text",
            placeholder="PDFをアップロードするとここに抽出テキストが表示されます…",
        )

    tab_check, tab_kouetsu = st.tabs(["✅ 固有名詞チェック", "📝 AI校閲"])

    # ── 固有名詞チェック ─────────────────────────────
    with tab_check:
        ready = bool(article.strip() and roster and api_key)
        run = st.button(
            "✅ チェック実行",
            type="primary",
            disabled=not ready,
            help="名簿・記事・APIキーをすべて入力してから実行してください。",
            key="run_check_btn",
        )

        if run:
            st.session_state.pop("result", None)
            with st.spinner("AIが照合中です…（数秒〜十数秒かかります）"):
                try:
                    st.session_state["result"] = check_with_claude(
                        article.strip(), roster, api_key, reference_docs
                    )
                    st.session_state["article_saved"] = article.strip()
                except anthropic.AuthenticationError:
                    st.error("APIキーが無効です。正しいキーを入力してください。")
                except json.JSONDecodeError as exc:
                    st.error(f"AIの応答を解析できませんでした: {exc}")
                except Exception as exc:
                    st.error(f"エラーが発生しました: {exc}")

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

    # ── AI校閲 ──────────────────────────────────────
    with tab_kouetsu:
        st.caption(
            "語彙・助詞、事実確認（固有名詞・数値・時制・主体関係等）、表記の観点で、"
            "インターネット検索を交えてAIが校閲します。"
        )
        kouetsu_ready = bool(article.strip() and api_key)
        run_kouetsu_btn = st.button(
            "📝 校閲を実行",
            type="primary",
            disabled=not kouetsu_ready,
            help="記事・APIキーを入力してから実行してください。",
            key="run_kouetsu_btn",
        )

        if run_kouetsu_btn:
            st.session_state.pop("kouetsu_result", None)
            with st.spinner("AIが校閲中です…（インターネット検索を伴うため1分程度かかることがあります）"):
                try:
                    st.session_state["kouetsu_result"] = run_kouetsu(
                        article.strip(), api_key, reference_docs
                    )
                except anthropic.AuthenticationError:
                    st.error("APIキーが無効です。正しいキーを入力してください。")
                except Exception as exc:
                    st.error(f"エラーが発生しました: {exc}")

        if "kouetsu_result" in st.session_state:
            kouetsu_result = st.session_state["kouetsu_result"]
            st.divider()
            if kouetsu_result.get("refusal"):
                st.warning("AIがこの内容の校閲を辞退しました。内容をご確認のうえ、再度お試しください。")
            else:
                st.markdown(kouetsu_result.get("text", "") or "（校閲結果が空でした）")


if __name__ == "__main__":
    main()
