# ローカル推論API

http://127.0.0.1:8765（ポート変更可能）のIPv4 loopbackのみで待ち受けます。
HTTPSではなく外部公開用ではありません。すべてのリクエストにAuthorization: Bearer <token>が必要です。
GUIでトークンをコピーし、設定を保存して再起動後も同じトークンを使用します。
コード・URL・ログへトークンを固定値で残さないでください。

- GET /health: {"status":"ready"}
- POST /infer: PNG/JPEG/BMPのバイナリを送信します（multipartやBase64 JSONではありません）。
- Content-Type: image/png、image/jpeg、image/bmp
- 応答: {"result": <SDKのJSON>, "mask": <int32配列またはnull>}
- maskは行優先です。幅・高さはresultの値を使います。

認証不備は401、許可しないHost/Originは403、処理中は429、未対応Content-Typeは415、
サイズ超過は413、読めない画像等は400です。API推論は同時1件で無制限の待ち行列を作りません。
CORSやブラウザからの直接利用は提供しません。

Pythonクライアント例（SDK不要）:

```python
import os
from pathlib import Path
from urllib.request import Request, urlopen

request = Request("http://127.0.0.1:8765/infer",
    data=Path("image.png").read_bytes(),
    headers={"Authorization": "Bearer " + os.environ["CURECO_API_TOKEN"],
             "Content-Type": "image/png"})
with urlopen(request, timeout=60) as response:
    print(response.read().decode("utf-8"))
```

画像・モデルはローカルで処理されます。同一アカウントのプロセスに対する分離機構ではありません。
