---
title: "Claude Codeの「Advisorストラテジー」を、同じ失敗を2回したら自動発動させるhookを作った"
emoji: "🩺"
type: "tech"
topics: ["claudecode", "anthropic", "hooks", "llm", "ai"]
published: false
---

## Advisorストラテジーとは

Claude CodeやAnthropicのAPIには `advisor` という仕組みがあります。実行を担うモデル（executor、例: Sonnet）が、判断が難しい局面だけ、より強いモデル（advisor、例: Opus）に会話全体の履歴を渡して意見を求める、というパターンです。

これは筆者の思いつきではなく、Anthropicが [The Advisor Strategy](https://claude.com/blog/the-advisor-strategy) として公式に発表している設計です。ブログの主張はシンプルです。

> Frontier-level reasoning applies only when the executor needs it, and the rest of the run stays at executor-level cost.

タスクの大部分は最上位モデルの推論力を必要としない。必要になるのは一部の判断ポイントだけ。だから常時フラグシップモデルを走らせるのではなく、安いモデルに実行させ、要所だけ強いモデルにエスカレーションする方が、コストと精度の両方で得をする、という主張です。実際に公開されているベンチマークでも、

- SWE-bench: Sonnet + Opusアドバイザーで**+2.7ポイント、コストは11.9%減**
- BrowseComp: Haiku + Opusアドバイザーで、**Sonnet単体よりコスト85%減、性能は倍増**

という数値が出ています。実装面では「1回のリクエスト内でハンドオフが完結し、追加のラウンドトリップやコンテキスト管理は不要」ともされています。

筆者はこの戦略を知る前から「デフォルトモデルをsonnetにしておいて、要所だけadvisor経由でopusに切り替えられればトークン効率が良いはずだ」という直感を持っていましたが、これは独自の思いつきではなく、Anthropicが数値付きで実証済みの設計そのものでした。

---

## 解決されていない問題:「いつ」呼ぶか

ブログの説明でカバーされていないのが、**advisorを呼ぶタイミングの設計**です。Claude Code上での `advisor` ツールの説明には次のような指針が書かれています。

> Call advisor BEFORE substantive work -- before writing, before committing to an interpretation, before building on an assumption.
> Also call advisor: When you believe the task is complete. [...] When stuck -- errors recurring, approach not converging, results that don't fit.

つまり「詰まったら呼べ」という指針はモデル自身の自己判断に委ねられています。筆者は普段Claude Codeを使っていて、これがうまく機能している感覚がありませんでした。同じコマンドを微妙に変えて2回、3回と失敗し続けても、モデルが自発的にadvisorを呼ばないことがある。

実際、このhookを作る前は、この「いつ呼ぶか」の判断を筆者自身が担っていました。作業の様子を見ていて「これは同じところで詰まっているな」と感じたら、その都度「Advisorに確認して」と手動で指示を出す、というのが実際の運用でした。つまり**トリガーの役割を人間が肩代わりしていた**わけです。これは機能はしますが、当然ながら筆者がその場で画面を見ていないと発火しません。セッションを離席中に詰まっていても誰も気づかず、Claudeはただ同じ失敗を繰り返し続けます。

この記事で作るhookは、要するに**それまで筆者が担っていた「観察してタイミングを判断する」役割を、コード側に委譲する**という話です。「いつ呼ぶか」を自己判断ではなく確実な仕組みにできないか、というのが本記事の出発点です。

---

## なぜCLAUDE.mdの指示だけでは不十分か

最初に思いつくのは、CLAUDE.mdに「同じ箇所で2回失敗したらadvisorを呼べ」と書いておくことです。実際、筆者の `~/.claude/CLAUDE.md` には次の一文があります。

```markdown
## Escalate to advisor after repeated failures

When a coding or deployment task fails at the same spot twice in a row
(same command/script, same underlying error), stop retrying the same
approach and call the `advisor` tool before continuing. Give it the full
context of what has been tried and why it failed.
```

しかしこれは**お願い**であって**強制**ではありません。CLAUDE.mdやメモリは、モデルがそれを読んで自発的に従うことを期待する仕組みで、「何回失敗したか」を正確に数え続けるような状態管理はモデルの自己観察に依存してしまい、信頼性が低くなります。

確実に「2回失敗したら必ず発火する」を実現するには、Claude Codeの **hooks** 機能でコード側に強制させる必要があります。

---

## hooksの基礎

hooksは、Claude Codeのライフサイクル上の特定のイベントで、任意のシェルコマンド（またはLLM評価、サブエージェント呼び出し）を自動実行する仕組みです。設定は `settings.json` に書きます。

```json
{
  "hooks": {
    "EVENT_NAME": [
      {
        "matcher": "ToolName",
        "hooks": [
          { "type": "command", "command": "your-command-here" }
        ]
      }
    ]
  }
}
```

イベントはJSONをstdinで受け取り、処理結果をJSONでstdoutに返すと、その内容に応じてClaude Codeが「続行」「ブロック」「文脈を注入」などを行います。主なイベントは以下の通りです（`settings.json` のJSON Schemaで確認できる正式なイベント名の一部）。

| イベント | 発火タイミング |
| --- | --- |
| `PreToolUse` | ツール実行前（ブロック可能） |
| `PostToolUse` | ツールが成功した後 |
| `PostToolUseFailure` | ツールが失敗した後 |
| `Stop` | Claudeが応答を終える時 |
| `SessionStart` / `SessionEnd` | セッション開始/終了時 |
| `UserPromptSubmit` | ユーザーがプロンプト送信時 |
| `PreCompact` / `PostCompact` | コンテキスト圧縮の前後 |

今回使うのは `PostToolUseFailure` です。

---

## 実機検証: PostToolUseFailureは本当にBashの失敗で発火するのか

ここでいったん足を止めます。Web検索で得られる情報だけでは、`PostToolUseFailure` が具体的にどんな条件で発火するのかが曖昧でした。「ツール呼び出し自体がクラッシュした場合」と「シェルコマンドが非ゼロで終了した場合」は別物である可能性があり、後者（`pytest` や `npm test` が失敗する、というよくあるケース）を捉えられないと今回の目的には使えません。

推測で進めず、実際に手元のインストール（Claude Code v2.1.222）で検証しました。まず一時的なデバッグ用hookを仕込み、実際のstdin JSONをダンプします。

```json
{
  "hooks": {
    "PostToolUse": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "~/.claude/hooks/debug_dump.sh" }] }],
    "PostToolUseFailure": [{ "matcher": "Bash", "hooks": [{ "type": "command", "command": "~/.claude/hooks/debug_dump.sh" }] }]
  }
}
```

`debug_dump.sh` はstdinをそのまま一時ファイルに保存するだけの1行スクリプトです。この状態で `exit 1` するBashコマンドを実行させると、以下のJSONが捕捉されました（実際の生ログ）。

```json
{
  "hook_event_name": "PostToolUseFailure",
  "tool_name": "Bash",
  "tool_input": { "command": "jq . ~/.claude/settings.json >/dev/null && echo \"valid\"\n...\nbash -c 'exit 1'" },
  "error": "Exit code 1\nvalid",
  "is_interrupt": false,
  "duration_ms": 30
}
```

`tool_response` は無く（成功時のみ付く）、代わりに `error` フィールドに `"Exit code N\n<出力>"` という形式でエラー内容が入っていました。これで確定です。**Bashコマンドが非ゼロで終了すると、確かに `PostToolUseFailure` が発火し、`error` フィールドに終了コードと出力が入る**ということが実機で確認できました。

---

## 実装（初版）

これを踏まえて、失敗を検知して2回目でadvisor呼び出しを促すhookを書きました。

```bash
#!/bin/bash
input=$(cat)
session_id=$(echo "$input" | jq -r '.session_id // "nosession"')
command=$(echo "$input" | jq -r '.tool_input.command // ""')
error=$(echo "$input" | jq -r '.error // ""')

spot=$(echo "$command" | tr -s ' \t\n' ' ' | awk '{print $1}')
problem=$(echo "$error" | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-150)

sig=$(printf '%s|%s' "$spot" "$problem" | shasum -a 256 | cut -c1-16)

state_dir="$HOME/.claude/hook_state/$session_id"
mkdir -p "$state_dir"
state_file="$state_dir/failures.json"
[ -f "$state_file" ] || echo '{}' > "$state_file"

count=$(jq -r --arg k "$sig" '.[$k] // 0' "$state_file")
count=$((count + 1))

if [ "$count" -ge 2 ]; then
  jq --arg k "$sig" 'del(.[$k])' "$state_file" > "$state_file.tmp" && mv "$state_file.tmp" "$state_file"
  jq -n --arg spot "$spot" \
    '{hookSpecificOutput: {hookEventName: "PostToolUseFailure", additionalContext: ("You have failed running \($spot) at this same spot twice in a row. Call the advisor tool now.")}}'
else
  jq --arg k "$sig" --argjson c "$count" '.[$k] = $c' "$state_file" > "$state_file.tmp" && mv "$state_file.tmp" "$state_file"
  exit 0
fi
```

設計のポイントは3つです。

1. **状態はセッション単位で `~/.claude/hook_state/<session_id>/` に保存**。プロジェクトのリポジトリ配下には置かない（`git status` を汚さないため）
2. **完全一致のコマンド文字列ではなく「同じ問題」でシグネチャ化**。Claudeはリトライごとにコマンドを微妙に変えるので、コマンド全文の一致を条件にすると発火しない
3. **2回目で発火したらカウントをリセット（latch）**。3回目以降も同じ失敗が続く度にadvisorへスパムしないようにする

同一コマンドを2回連続で失敗させてこのhookをパイプ経由でテストしたところ、1回目は無反応、2回目で以下のJSONが出力され、意図通り動作しているように見えました。

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUseFailure",
    "additionalContext": "You have failed running pytest at this same spot twice in a row..."
  }
}
```

これでいけると思っていました。

---

## 見つかったバグとその顛末

実際のセッションで実機テストをしていたところ、おかしなことが起きました。「同じ失敗」を2回連続で起こしたつもりなのに、状態ファイルを見ると異なる2つのハッシュキーが記録されていたのです。

```json
{
  "021509254ae7e3a2": 1,
  "cc0d4de7acf2fe1f": 1
}
```

最初はこれを「テスト方法のミス」だと判断しました。実際、2回目のテストでは `cat ~/.claude/hook_state/.../failures.json` と `python3 -c "import xxx"` を1つのBashコマンドとして連結して実行してしまっており、`spot`（コマンドの先頭1語）が `cat` になっていたことが原因でした。単独のコマンドとして分離して再実行すると、期待通り2回目で発火しました。

ここで一度「解決した」と思ったのですが、レビューを求めたところ、これは**テスト方法のミスであると同時に、設計そのものの欠陥でもある**という指摘を受けました。`spot` はBashに渡された文字列全体の先頭1語を取っているだけなので、Claudeが複数コマンドを `&&` や改行で連結して1回のBash呼び出しにまとめた場合、そのつど「先頭のコマンド」が変わり、同じ根本原因の失敗が別々のシグネチャとして記録されてしまいます。これはテストでだけ起きた偶然ではなく、**通常運用でClaudeが複数コマンドを連結する度に発生する構造的な問題**でした。

これを実機で再現してみます。前置コマンドだけ変えて、根本原因は同じ失敗を2回起こします。

```bash
# 1回目
cd /tmp && python3 -c "import nonexistent_module_xyz_test"

# 2回目（前置コマンドだけ違う）
ls /tmp >/dev/null && python3 -c "import nonexistent_module_xyz_test"
```

結果、状態ファイルには案の定、別々の2つのシグネチャが記録され、advisorへの誘導は発火しませんでした。

```json
{
  "b912226e6701927f": 1,
  "10b3bec3beeea635": 1
}
```

修正は単純です。**`spot`（コマンドの先頭語）をハッシュから外し、エラー出力の末尾行だけでシグネチャを作る**ようにしました。

```diff
- spot=$(echo "$command" | tr -s ' \t\n' ' ' | awk '{print $1}')
  problem=$(echo "$error" | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-150)

- sig=$(printf '%s|%s' "$spot" "$problem" | shasum -a 256 | cut -c1-16)
+ sig=$(printf '%s' "$problem" | shasum -a 256 | cut -c1-16)
```

同じ前置コマンド違いのテストを再実行すると、今度は意図通り2回目で発火しました。

```text
PostToolUseFailure:Bash hook additional context: You have hit this same
underlying failure twice in a row across Bash calls. Do not retry the
same approach again. Call the advisor tool now before continuing...
```

この一連の流れ（バグに気づく→誤診断する→指摘されて構造的な問題だと分かる→再現させる→直す→再検証する）は、このhookをそのまま真似て使おうとする人が確実に踏む部分だと思うので、あえて経過をそのまま書きました。

---

## 設計の裏付け: advisor_rankという内部制約

余談ですが、この調査の過程でインストール済みバイナリの文字列を `strings` コマンドで調べていたところ、次のような内部エラーメッセージ文字列を見つけました。

```text
'{}' has no advisor rank in the model catalog. Switch to a public model
alias (opus, sonnet, fable)...
(advisor must be at least as capable as the base model)
```

Claude Code内部にはモデルごとの「advisorランク」というものがあり、**advisorに指定するモデルは、実行を担うベースモデル以上のランクでなければならない**という制約がコード上に明文化されていました。つまり「安いモデルを実行役にして、高いモデルをadvisorにする」という構成は運用上のTipsではなく、実装レベルで前提とされている設計だということです。sonnet（実行）+ opus（advisor）という構成が「正しい使い方」であることの裏付けになりました。

:::message
ここで挙げているのはドキュメント化されていない内部文字列であり、バージョンによって変わる可能性があります。挙動の裏付けとしては読めますが、これに依存した実装は避けたほうが無難です。
:::

---

## 既知の限界と前提条件

正直に書いておきます。

### シグネチャ衝突のリスク

エラー出力の末尾行だけでシグネチャ化するようにしたため、根本原因が全く異なる2つの失敗が、たまたま同じ末尾行（出力が空の `Exit code 1` や、汎用的な `command not found` など）になった場合、誤って「同じ失敗」として扱われる可能性があります。

「前置コマンド依存で見逃す（false negative）」問題を解消する代わりに、「無関係な失敗を同一視する（false positive）」リスクを受け入れる、というトレードオフを選んだ形です。今回のユースケースでは「見逃す」方が実害が大きいと判断してこちらを選びましたが、この特性は使う前に知っておくべきだと思います。

### advisorツール自体がexperimental

`strings` 調査では `CLAUDE_CODE_ENABLE_EXPERIMENTAL_ADVISOR_TOOL` という環境変数も見つかりました。advisorツールは現時点ですべての環境にデフォルトで有効というわけではなく、有効化のための前提条件がある可能性があります。

:::message alert
このhookを試す場合は、まず `advisor` ツールが自分の環境で呼び出せるかどうかを先に確認してください。
:::

---

## まとめ

- このhookを作る前は、「同じところで詰まっている」と気づいて「Advisorに確認して」と指示するのは筆者自身の役目だった。今回作ったのは、その観察とタイミング判断をコード側に委譲する仕組み
- 「sonnetを実行役にして要所だけopusにエスカレーションする」構成は、Anthropicが数値付きで実証している公式戦略（the Advisor Strategy）
- しかし「いつエスカレーションするか」はモデルの自己判断に委ねられており、これを確実にしたい場合はCLAUDE.mdの指示だけでは不十分で、hooksによる強制が必要
- `PostToolUseFailure` はBashコマンドの非ゼロ終了で確実に発火することを実機で確認した
- 失敗の「同一性」をコマンド文字列でシグネチャ化すると、Claudeがコマンドを連結するたびに見逃しが発生する構造的な欠陥があり、エラー出力の末尾行のみでキーイングする方が頑健
- その代わりに衝突（false positive）のリスクを負っていることは自覚して使う

コード全体は以下の通りです。

```bash
#!/bin/bash
# PostToolUseFailure hook (matcher: Bash). Tracks repeated failures at the
# "same spot" (same error signature) and nudges Claude to consult the
# advisor tool after the 2nd occurrence in a session.

input=$(cat)
session_id=$(echo "$input" | jq -r '.session_id // "nosession"')
error=$(echo "$input" | jq -r '.error // ""')

problem=$(echo "$error" | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-150)

# Key on the error tail only, not the command's first token: Claude routinely
# varies the preamble between retries (cd vs ls, different chained commands),
# which would otherwise fragment one real failure into multiple signatures
# that never reach the threshold.
sig=$(printf '%s' "$problem" | shasum -a 256 | cut -c1-16)

state_dir="$HOME/.claude/hook_state/$session_id"
mkdir -p "$state_dir"
state_file="$state_dir/failures.json"
[ -f "$state_file" ] || echo '{}' > "$state_file"

count=$(jq -r --arg k "$sig" '.[$k] // 0' "$state_file")
count=$((count + 1))

if [ "$count" -ge 2 ]; then
  jq --arg k "$sig" 'del(.[$k])' "$state_file" > "$state_file.tmp" && mv "$state_file.tmp" "$state_file"
  jq -n \
    '{hookSpecificOutput: {hookEventName: "PostToolUseFailure", additionalContext: "You have hit this same underlying failure twice in a row across Bash calls. Do not retry the same approach again. Call the advisor tool now before continuing, and give it the full context of what has been tried."}}'
else
  jq --arg k "$sig" --argjson c "$count" '.[$k] = $c' "$state_file" > "$state_file.tmp" && mv "$state_file.tmp" "$state_file"
  exit 0
fi
```

`settings.json` 側の設定は次の通りです。

```json
{
  "hooks": {
    "PostToolUseFailure": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/advisor_nudge.sh", "timeout": 10 }
        ]
      }
    ]
  }
}
```
