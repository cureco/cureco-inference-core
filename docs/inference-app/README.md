# 簡易推論アプリ

Windows x64用C# WPFアプリです。examples/csharp/inference-appにVisual Studio solutionを含むソースがあります。
SDK組み込みサンプルとして改変できます。

ONNXとPNG/JPEG/BMPを開き、推論を実行します。検出枠・マスクは画像上、詳細は結果欄へ表示します。
JSONと表示PNGを保存できます。JSONにはint32のセグメンテーションmaskも含みます。
API稼働中はモデル変更できません。補助入力が必要なモデルはSDKを使ってください。
画像は20 MiB / 16メガピクセルまでです。

## 起動設定

モデル・ポートを選び「アプリ起動時にAPIを開始」を必要に応じて選択し、設定を保存します。
次回起動時に保存モデルでAPIを開始します。GUI最小化も選択できます。
GUIが不要な場合:

```text
CurecoInference.exe --serve
```

保存済みのモデル・ポート・トークンで直ちにAPIを開始します。
設定不正やモデル読み込み失敗時は終了コード1で終了し、設定フォルダへstartup-error.logを出力します。
Windowsサービスではありません。OSのタスクスケジューラ等から、設定を保存したWindowsユーザーとして起動できます。
自動起動登録はアプリ・インストーラから勝手に行いません。

設定は利用者のLocalApplicationData配下Cureco/InferenceAppへ保存します。
検証用や複数インスタンスの設定分離にはCURECO_INFERENCE_SETTINGS_DIRで別の設定ディレクトリを指定できます。
トークンはDPAPI CurrentUserで保護し、別PCや別ユーザーにコピーしても使えません。
モデルは設定に含まず、移動・削除すると起動時読み込みが失敗します。[API仕様](api.md)

CI成果物は署名前の検証用です。配布は署名・実機確認後に行います。
自己完結型ビルドはPython/.NETの別途導入を不要にしますが、DLLを含むフォルダ全体が必要です。
