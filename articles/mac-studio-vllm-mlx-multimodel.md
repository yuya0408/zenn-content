---
title: "vllm-mlx に移行しても速くならない ― Mac Studio で踏んだ2つの既定値"
emoji: "🍎"
type: "tech"
topics: ["mlx", "vllm", "ollama", "llm", "applesilicon"]
published: false
---

Mac Studio 上で動かしているローカル LLM の推論エンジンを、Ollama から [vllm-mlx](https://github.com/waybarrios/vllm-mlx) に移した。その過程で、**設定を何も変えないまま「移行したのに速くならない」状態で立ち上がる**箇所が2つあった。どちらもエラーを出さないので、気づくのが遅れる。同じ構成を組む人向けに、そこだけ書いておく。

:::message
構成が成立することとプレフィックスキャッシュが効くことは実測で確認した。ただし Ollama との比較や、同時アクセス下での待ち時間の分布はまだ取れていない。本記事は「移行時にここで詰まる」という話が主で、性能比較の記事ではない。
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

## 実測: 2モデルを同時に載せて確かめた

上記を踏まえて、実運用候補の2モデルを単一サーバーに同時 preload して確認した。`--models-config` で登録し、ポート 18012 で起動している。

- [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
- [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit)

確認できたのは以下である。

- **両モデルとも BatchedEngine(paged-cache + continuous-batching)でロードされる。** レジストリ経由でも、前節のフラグを入れれば意図した経路に乗る
- **単一エンドポイントで `model` フィールドによる切り替えが正常に動く。** 前段のルータは不要のまま
- **tool calling が両モデルで動く**
- **プレフィックスキャッシュが効いている。** Qwen 側で **1.17s → 0.63s**。同一プレフィックスの再送で約 46% 短縮した
- **A/B/A/B と交互にリクエストしても安定する。** 一方のモデルへの切り替えが他方の状態を壊さない
- **`enable_thinking: false` による思考出力の抑制が効く**

狙っていた構成そのものは、ここで成立が確認できた。

## 設計上の制約: `tool_call_parser` はプロセス全体で1つ

ただし、この検証で**マルチモデル構成に効いてくる制約**が1つ見つかった。

`--tool-call-parser` は**プロセス全体でグローバルな単一の値**であり、`--models-config` の YAML でモデルごとに上書きできない。ソースを追うと `vllm_mlx/server.py` にモジュールレベルの変数として置かれている。

```python
_tool_call_parser: str | None = None   # パーサー名: auto, mistral, qwen, llama, hermes...
_tool_parser_instance = None           # 実体化されたパーサー
```

`_get_or_init_tool_parser()` が遅延初期化で1個だけ実体化し、以後は全リクエストでその1個を使い回す(リクエスト間で `reset()` を呼ぶ)。モデルごとの状態ではない。

これが問題になるのは、**異なるモデルファミリを同居させたとき**である。Qwen 系と Gemma 系では本来必要なパーサーが違うのに、プロセスに1つしか持てない。どちらかに固定すれば、もう一方のツール呼び出しが壊れる。

したがって、**マルチモデル構成では実質的にフォーマット非依存の `--tool-call-parser auto` が必須になる。** 実測では auto で両モデルとも正常に動作した。auto はツール呼び出しの出力フォーマットを順に試して判別する方式で、ドキュメント上も Gemma 4 形式(`<|tool_call>call:name...<tool_call|>`)や Mistral 形式(`[TOOL_CALLS]`)などが判別対象に挙がっている。

:::message alert
公開ドキュメントの tool calling ガイドは `--tool-call-parser` を単一モデル起動時の CLI フラグとして説明しており、**複数モデルを異なるファミリで同居させた場合にパーサーがどう扱われるかには触れていない。** 「モデルごとに適切なパーサーを指定すればよい」と考えて構成を組むと、ここで詰まる。
:::

## ツール呼び出しは、載せる前に軽く投げて確かめる

パーサーが用意されているモデルでも、モデルのバージョンや量子化の組み合わせによってはパースを外すことがある。ツール呼び出し前提のシステムでは、これは静かに壊れる。モデルは自然言語としてはもっともらしい応答を返し、ただ `tool_calls` が空になる、という形で失敗する。

そこで、本番のスキーマを載せる前に**引数1つの最小ツールを1本だけ定義して呼ばせるスモークテスト**を通すことにした。数秒で終わる。「パーサーが対応表に載っている」ことと「この環境で実際にパースできる」ことの間には差があるので、確認しておく価値がある。

前節の通りパーサーはプロセスに1つしか持てないので、**マルチモデル構成では登録した全モデルに対してこれを通す**必要がある。1モデルで通ったから他も通る、とは言えない。

## まとめ

- **`gpu_memory_utilization` の感覚を持ち込まない。** vllm-mlx は「重みの常駐予算」と「キャッシュ割合」の二段構えで、`memory_budget_gb` は重みしか数えない
- **効かせたい機能はデフォルトで無効。** `--continuous-batching` / `--use-paged-cache` / `--prefix-trie-cache` はいずれも既定 `False`。エラーが出ないまま「移行したのに速くならない」になる
- **`tool_call_parser` はプロセスに1つしか持てない。** モデルごとの上書きができないため、異なるファミリを同居させるなら `auto` が実質必須になる
- ツールパーサーは、対応表を信じずに登録した全モデルに投げて確かめる

2モデル同時ロードの構成が成立すること、プレフィックスキャッシュが効くこと(Qwen で 1.17s → 0.63s)までは実測で確認できた。**ただし Ollama との比較と、同時アクセス下での待ち時間の分布はまだ取れていない。** 移行の目的だったレイテンシ改善そのものを裏づけるには、フラグ全オフ / 全オン / Ollama の3点比較が要る。そこは別途測る。

## 参考

- [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) ― MLX ネイティブの推論サーバ(非公式)
  - [Model Registry](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/model-registry.md) ― マルチモデル構成の YAML スキーマ
  - [CLI Reference](https://github.com/waybarrios/vllm-mlx/blob/main/docs/reference/cli.md) ― 本記事で挙げたフラグと既定値の出典
  - [Tool Calling](https://github.com/waybarrios/vllm-mlx/blob/main/docs/guides/tool-calling.md) ― `--tool-call-parser` と `auto` の判別対象フォーマット
  - [`vllm_mlx/server.py`](https://github.com/waybarrios/vllm-mlx/blob/main/vllm_mlx/server.py) ― `_tool_call_parser` / `_tool_parser_instance` がモジュールレベルに置かれている箇所
- [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) ― Apple Silicon 向け vLLM ハードウェアプラグイン(コミュニティ保守)
- [Ollama FAQ](https://docs.ollama.com/faq) ― `OLLAMA_MAX_LOADED_MODELS` / `OLLAMA_NUM_PARALLEL`
