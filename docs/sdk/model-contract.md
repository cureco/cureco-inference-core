# モデル契約と拡張

現在の契約はschema_version=1、C ABIはv1です。
ONNXメタデータcureco_contractにJSONで保存します。
グラフと契約のシグネチャが不一致の場合は拒否し、不明な出力を推測で解釈しません。

| タスク | グラフ出力 | 結果 |
| --- | --- | --- |
| classification | output: float32 [B,C] logits | ラベル・信頼度 |
| detection | boxes: float32、labels: int64、scores: float32 | 元画像座標xyxy・ラベル・信頼度 |
| segmentation | output: float32 [B,C,H,W] logits | 元画像解像度のint32クラスIDマスク |

画像入力はimg_tensor、float32 NCHW、3チャネル、固定のモデル入力高さ・幅です。
1回の呼び出しは画像1枚です。分類の補助入力にはparam_tensorを使いSDKからfloat配列を渡します。
簡易アプリは補助入力が必要なモデルを拒否します。

前処理はSDKでRGB、bicubic/antialias stretch resize、1/255スケール、
指定時に3チャネルmean/std正規化を行います。
検出はモデル入力画像ピクセルxyxy、0始まりクラスID、confidenceを受け取ります。
NMS=inside_graphなら再適用せず、not_appliedならSDK内NMSを使用します。
セグメンテーションはクラス軸argmax後に元画像へ最近傍リサイズします。
mask_policyを指定する場合background_modeはimplicit/explicit、ignore_indexは255です。

具体例はtests/integration/test_sdk.pyのwrite_modelで生成できます。
入力名・dtype・shape、output_semantics、preprocessing、idx_to_classなどを確認できます。
既存形式の受け付けは限定的です。新しいモデルにはv1契約を明示してください。

## アルゴリズムの追加

既存契約へ正規化できる新しいネットワークは、言語バインディングやアプリの変更を不要にします。
意味が異なる出力は、識別可能なタスク・契約バージョンを定義し、sdk/nativeに検証・前後処理、
合成モデルの横断テストを追加します。ABI変更は別途バージョン管理し既存ABIを破壊しません。
不明な契約は明示的な未対応エラーにします。汎用プラグイン機構の先行公開はしません。
