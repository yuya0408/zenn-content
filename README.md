# zenn-content

Zenn の記事を管理するリポジトリ。Zenn の GitHub 連携先としてこのリポジトリを指定する。

## 構成

```
.
├── articles/   # 記事（Markdown）。ファイル名（拡張子を除く）が記事の slug になる
├── images/     # 記事内で /images/... として参照する画像
└── package.json
```

- 記事はテーマを問わずすべて `articles/` 直下に置く（サブディレクトリは Zenn が認識しない）。
- SECOM 関連の分析コードは別リポジトリ [secom-defect-prediction](https://github.com/yuya0408/secom-defect-prediction) にある。

## ローカルでのプレビュー

```bash
npm install
npx zenn preview   # http://localhost:8000
```

## 新しい記事を作る

```bash
npx zenn new:article
```

`published: true` にして push すると、連携済みの Zenn アカウントに反映される。
