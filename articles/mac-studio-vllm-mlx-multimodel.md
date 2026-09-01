---
title: "vllm-mlx に移行しても速くならない ― Mac Studio で踏んだ2つの既定値"
emoji: "🍎"
type: "tech"
topics: ["mlx", "vllm", "ollama", "llm", "applesilicon"]
published: false
---

Mac Studio 上で動かしているローカル LLM の推論エンジンを、Ollama から [vllm-mlx](https://github.com/waybarrios/vllm-mlx) に移した。その過程で、**設定を何も変えないまま「移行したのに速くならない」状態で立ち上がる**箇所が2つあった。どちらもエラーを出さないので、気づくのが遅れる。同じ構成を組む人向けに、そこだけ書いておく。

:::message
レイテンシの実測はまだ取れていない。本記事は「移行時にここで詰まる」という設定上の話に限定する。効果の検証は別途。
:::

## 前提

- **Mac Studio 上でのサービング。** 扱うデータの都合でローカル推論が要件
- **マルチアクセス。** 社内の複数ユーザーが同時に叩く
- **ツール呼び出し前提。** チャットから社内システムを操作するので function calling が動かないと成立しない
- **ツールスキーマが膨大。** リクエストごとの入力トークンの大半が、毎回ほぼ同一のスキーマ定義で占められる
- **マルチモデルロード。** 用途ごとにモデルを使い分けるため複数を常駐させたい

4つ目が効いている。会話本体が数十トークンでも、その前に数千トークンのスキーマが毎回付く。**毎リクエストの prefill で、ほぼ同じ長大なプレフィックスを再計算している**わけで、ここをキャッシュで効かせたい。それが vLLM 系への移行を検討した出発点だった。

:::message
念のため補足すると、Ollama にも `OLLAMA_MAX_LOADED_MODELS`(複数モデルの常駐)と `OLLAMA_NUM_PARALLEL`(並列リクエスト)の設定はある。「Ollama にはできない」から移ったわけではない。
:::

## vllm-metal ではなく vllm-mlx を選んだ

先に [vllm-metal](https://github.com/vllm-project/vllm-metal) を検証した。`vllm-project` org 配下でドキュメントも docs.vllm.ai にあるが、リポジトリ自身の説明は "Community maintained" である。2点で要件に合わず見送った。

- **マルチモデルの同時ロードができない。** vLLM 本体同様1プロセス1モデルが基本で、複数常駐させるならプロセスを並べて前段にルータを置くことになる
- **ツール呼び出しがスコープ外に見える。** README にも公開ドキュメントにも tool calling / tool parser の記載が見当たらない

単一モデルでツール呼び出しが不要なら、むしろこちらの方が素直だと思う。今回の要件と噛み合わなかっただけである。

vllm-mlx は非公式プロジェクト(個人メンテナのリポジトリで、vLLM 本体からの派生でもない)だが、要件との噛み合いが良かった。`--models-config` に YAML を渡すと複数モデルを常駐させられる。

```bash
vllm-mlx serve --models-config /etc/vllm-mlx/models.yaml --host 0.0.0.0 --port 8000
```

```yaml
manager:
  memory_budget_gb: 100
  idle_unload_seconds: 300

models:
  - name: fast
    path: /path/to/model
    preload: true
    continuous_batching: true
    estimated_memory_gb: 4
```

振り分けは OpenAI 互換の `model` フィールドで行う。クライアントが `model: "fast"` と指定するだけでよく、**前段にルータを別コンポーネントとして立てる必要がない。** マルチモデルを扱うとこの層が増えて運用対象が増えがちなので、1プロセスに畳めたのは大きかった。

ツールパーサーも `--tool-call-parser` で選べる。`auto` / `mistral` / `qwen` / `llama` / `hermes` / `deepseek` / `kimi` / `granite` / `nemotron` / `xlam` / `functionary` / `glm47` が用意されている。

## 既定値の罠 1: `memory_budget_gb` は重みしか数えない

vLLM 本体には `gpu_memory_utilization` があり、既定は **0.9** である。これはクラスタ化した複数 GPU 環境で、**1つの GPU の中で vLLM が単独で動く**前提の最適化だ。GPU の9割をモデル重みと KV キャッシュのために先に確保する。だから素の vLLM で複数モデルを同居させるならこの値を下げる、というのが定石になる。

**vllm-mlx はこの流儀を採っていない。** メモリ制御が二段構えになっている。

| 設定 | 対象 | 既定 |
|---|---|---|
| `manager.memory_budget_gb` / `--memory-budget-gb` | **モデル重みの常駐量のみ** | YAML 指定 |
| `--cache-memory-percent` | キャッシュに充てる RAM の割合 | `0.20` |
| `--cache-memory-mb` | キャッシュ上限の絶対値 | Auto |

罠は `memory_budget_gb` の定義にある。**これはモデル重みだけを数えており、KV キャッシュとアクティベーションを含まない。** 128GB 機で「100GB まで使ってよい」と設定しても、その 100GB は重みの話で、KV キャッシュはその上に積まれる。ドキュメントが 128GB 機で 80〜100GB を目安としているのは、残りを OS・アクティベーション・KV キャッシュで食う前提だからだ。

`gpu_memory_utilization` の感覚で「9割まで使える」と重みを詰め込むと、同時アクセスが乗った瞬間に破綻する。移行時に頭を切り替える必要がある箇所だった。

## 既定値の罠 2: 効かせたい機能が、軒並みデフォルト無効

これが本題である。vllm-mlx では**移行の目的そのものだった機能が、既定でオフになっている。**

| フラグ | 既定 |
|---|---|
| `--continuous-batching` | `False` |
| `--use-paged-cache` | `False` |
| `--prefix-trie-cache` | `False` |
| `--enable-auto-tool-choice` | `False` |

何も指定せずに起動すると、**Ollama から移行した意味がほぼ無い状態で立ち上がる。** しかもエラーは出ない。普通に応答が返ってくるので、「移行したのに速くならない」という形でしか気づけない。

関連するパラメータも併せて見ておきたい。

- `--paged-cache-block-size` ― 1ブロックあたりのトークン数(既定 `64`)
- `--max-cache-blocks` ― 最大ブロック数(既定 `1000`)
- `--prefix-trie-cache-size` ― prompt trie のエントリ数上限(既定 **`32`**)
- `--prefix-trie-cache-memory-mb` ― trie のメモリ上限(既定なし)

特に `--prefix-trie-cache-size` の既定 `32` は、常駐ユーザー数と会話本数によっては簡単に溢れる。ツールスキーマの prefill を効かせるのが目的なら、実際の同時セッション数を見て決めるべき値だ。

:::message
用語として整理しておくと、**TTFT(初回トークンまでの時間)を直接縮めるのは prefix caching** であって、paged KV cache そのものではない。paged 側は KV の断片化を防いでメモリ効率を上げ、その結果として同時実行数を稼ぐ役割になる。「PagedAttention でレイテンシが改善する」と一括りにすると、どのフラグが何に効いているのか見えなくなる。
:::

## ツール呼び出しは、載せる前に軽く投げて確かめる

パーサーが用意されているモデルでも、モデルのバージョンや量子化の組み合わせによってはパースを外すことがある。ツール呼び出し前提のシステムでは、これは静かに壊れる。モデルは自然言語としてはもっともらしい応答を返し、ただ `tool_calls` が空になる、という形で失敗する。

そこで、本番のスキーマを載せる前に**引数1つの最小ツールを1本だけ定義して呼ばせるスモークテスト**を通すことにした。数秒で終わる。「パーサーが対応表に載っている」ことと「この環境で実際にパースできる」ことの間には差があるので、確認しておく価値がある。

## まとめ

- **`gpu_memory_utilization` の感覚を持ち込まない。** vllm-mlx は「重みの常駐予算」と「キャッシュ割合」の二段構えで、`memory_budget_gb` は重みしか数えない
- **効かせたい機能はデフォルトで無効。** `--continuous-batching` / `--use-paged-cache` / `--prefix-trie-cache` はいずれも既定 `False`。エラーが出ないまま「移行したのに速くならない」になる
- ツールパーサーは、対応表を信じずに一度投げて確かめる

冒頭に書いた通り、**レイテンシの実測はまだ無い。** 本記事は「移行時にここで詰まる」という設定上の話であって、効果を数字で裏づけたものではない。実測はフラグ全オフ / 全オン / Ollama の比較として別途取る。

## 参考

- [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) ― MLX ネイティブの推論サーバ(非公式)
  - [Model Registry](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/model-registry.md) ― マルチモデル構成の YAML スキーマ
  - [CLI Reference](https://github.com/waybarrios/vllm-mlx/blob/main/docs/reference/cli.md) ― 本記事で挙げたフラグと既定値の出典
- [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) ― Apple Silicon 向け vLLM ハードウェアプラグイン(コミュニティ保守)
- [Ollama FAQ](https://docs.ollama.com/faq) ― `OLLAMA_MAX_LOADED_MODELS` / `OLLAMA_NUM_PARALLEL`
