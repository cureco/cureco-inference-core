"""異常検知エンジン（PatchCore-lite）— 良品学習のみで欠陥を検出する。

正常パッチ集合を WideResNet50 中間特徴（layer2+layer3）に埋め込み、
greedy k-center coreset で圧縮したメモリバンクを構築する。判定はバンクへの
最近傍距離（PatchCore [Roth+ 2022] の簡易実装）。

出自: filament-inspect（線材欠陥検査PoC）の detect/stage2.py からの寄贈。
実測レポートは filament-inspect docs/REPORT.md・DESIGN_MODES.md を参照。

依存はオプション: `pip install cureco-inference-core[anomaly]`（torch / torchvision）。
教師あり分類（CurecoInference・ONNX）とはモデル形式が異なるため独立モジュールとする。
バンク＝モデルファイル（.pt）であり、**学習（fit）はエッジ内で数分・アノテーション不要**。

基本的な使い方:
    ```python
    from cureco_inference_core.anomaly import PatchCoreLite

    judge = PatchCoreLite()                # weights_path= でバックボーン重みを固定可
    judge.fit(normal_crops)                # (N,96,96) or (N,96,96,3) uint8
    judge.save("bank.pt")

    judge = PatchCoreLite().load("bank.pt")
    scores, maps = judge.score_crops(crops)
    ```
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch / torchvision が見つかりません。異常検知モジュールは "
        "`pip install cureco-inference-core[anomaly]` でインストールしてください"
    ) from e

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# 推論バッチの丸め先。候補数はチャンクごとに変わるが、形状が毎回変わると cuDNN の
# autotune が走り直して遅くなるため、数種類に固定する。
# cuDNN autotune（下の cudnn.benchmark）は「初めて見る入力形状」ごとに約5秒の総当たり探索を
# 行うため、バッチ形状は1種類に固定する。複数バケット（旧: 8/16/32/64/128）だと各バケットの
# 初回で5秒止まり、エリアモード（フレームごとに候補数が変わる）ではデモ中に何度も固まって
# 見えた。128固定なら探索は起動時ウォームアップの1回だけ。小バッチの切り上げコストは
# 128件ぶんの推論 ≈ 40ms/回で、静止製品のエリアにも、バッファで吸収するラインにも実害がない。
_BATCH_BUCKETS = (128,)


class PatchCoreLite:
    def __init__(
        self,
        device: str = "cuda",
        crop: int = 96,
        input_size: int = 160,
        coreset_size: int = 15000,
        pre_subsample: int = 300_000,
        proj_dim: int = 128,
        seed: int = 0,
        weights_path: Path | str | None = None,
    ):
        # weights_path: WideResNet50 の state_dict（.pt）。オフライン環境や
        # PyInstaller 同梱で torchvision の実行時ダウンロードを避けたい場合に指定する。
        self.weights_path = Path(weights_path) if weights_path else None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.crop = crop
        self.input_size = input_size
        self.coreset_size = coreset_size
        self.pre_subsample = pre_subsample
        self.proj_dim = proj_dim
        self.seed = seed
        self.half = self.device.type == "cuda"         # fp16 推論（Blackwell TensorCore 活用）
        self.bank: torch.Tensor | None = None          # (K, D)
        self._bank_sq: torch.Tensor | None = None      # |b|^2 キャッシュ（距離計算用）
        self._bank_t: torch.Tensor | None = None       # bank^T（推論dtype）キャッシュ
        self.normal_scores: np.ndarray | None = None   # 正常検証パッチのスコア分布（閾値校正用）
        self.path: Path | None = None                  # save/load したバンクの場所（absorb 後の保存用）
        self.color: bool = False                       # カラーバンクか（objectプロファイル）
        self.dense_scores: np.ndarray | None = None    # 全面スキャンの画像レベル校正分布
        self._backbone = None
        self._feats: dict[str, torch.Tensor] = {}

    # ---------- backbone ----------

    def _ensure_backbone(self) -> None:
        if self._backbone is not None:
            return
        # weights_path 指定時はローカル state_dict（オフライン現場・同梱配布向け）。
        # 未指定なら torchvision の学習済み重み（初回のみダウンロード）。
        if self.weights_path is not None:
            m = wide_resnet50_2(weights=None)
            m.load_state_dict(torch.load(self.weights_path, map_location="cpu"))
        else:
            m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        # 特徴は layer2/layer3 からしか取らないので layer4 以降は捨てる。
        # 出力される特徴は不変（＝既存メモリバンクと互換）で、推論コストだけ落ちる。
        m.layer4 = torch.nn.Identity()
        m.fc = torch.nn.Identity()
        m.eval().to(self.device)
        for p in m.parameters():
            p.requires_grad_(False)
        m.layer2.register_forward_hook(lambda _m, _i, o: self._feats.__setitem__("l2", o))
        m.layer3.register_forward_hook(lambda _m, _i, o: self._feats.__setitem__("l3", o))
        if self.device.type == "cuda":
            # NHWC + cuDNN autotune。TensorCore が本来の性能で回り、本ワークロードで実測2.3倍。
            # autotune は入力形状ごとに走るため、バッチは _BATCH_BUCKETS（128の1種）に固定する。
            m = m.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        self._backbone = m
        if self.device.type == "cuda":
            # autotune の約5秒をここで1回だけ払う（唯一のバッチ形状をダミーで踏んでおく）。
            # これを怠ると、最初に候補が出たフレーム/チャンクの処理が5秒止まって見える。
            self._embed(np.zeros((1, self.crop, self.crop), dtype=np.uint8))

    def _pad_to_bucket(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """バッチを固定サイズへ切り上げパディングし、(パディング後, 元の件数) を返す。"""
        k = len(x)
        for b in _BATCH_BUCKETS:
            if k <= b:
                if b > k:
                    x = torch.cat([x, x[-1:].expand(b - k, -1, -1, -1)])
                break
        return x, k

    @torch.inference_mode()
    def _embed(self, crops: np.ndarray, batch: int = 128, to_cpu: bool = False) -> torch.Tensor:
        """(N, H, W) uint8 グレー または (N, H, W, 3) uint8 BGRカラー → パッチ特徴 (N, P, D)。

        カラー入力は「小物体＋色系欠陥」対応のエリアモード用（バックボーンは元々RGBの
        ImageNetモデル。グレーは3chに複製していただけなので、実カラーを入れると色ムラ・
        変色が特徴の逸脱として現れる）。バンクはグレー/カラーで別物になる（混在不可）。
        to_cpu=True で埋め込みをCPUに退避（大量パッチの fit 時のVRAM対策）。
        """
        self._ensure_backbone()
        outs = []
        cuda = self.device.type == "cuda"
        mean = _IMAGENET_MEAN.to(self.device)
        std = _IMAGENET_STD.to(self.device)
        color = crops.ndim == 4
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i : i + batch]).to(self.device).float().div_(255.0)
            if color:
                x = x.permute(0, 3, 1, 2)[:, [2, 1, 0]]   # BGR(HWC) → RGB(CHW)
            else:
                x = x.unsqueeze(1)
            # 輝度不変化: パッチ平均を0.5に揃える（照明ドリフト・画像間の明るさ差への耐性）。
            # カラーはチャンネル別に揃える＝色かぶり除去。局所的な色の異常は残る
            x = x - x.mean(dim=(2, 3), keepdim=True) + 0.5
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
            if not color:
                x = x.repeat(1, 3, 1, 1)
            x = (x - mean) / std
            k = len(x)
            if cuda:
                x, k = self._pad_to_bucket(x)
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                self._backbone(x)
            f2, f3 = self._feats["l2"].float(), self._feats["l3"].float()
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([f2, f3], dim=1)                     # (B, 1536, gh, gw)
            f = F.avg_pool2d(f, 3, stride=1, padding=1)        # 局所平滑（PatchCore 準拠）
            f = f[:k]                                          # パディング分を捨てる
            b, d, gh, gw = f.shape
            emb = f.permute(0, 2, 3, 1).reshape(b, gh * gw, d)
            outs.append(emb.cpu() if to_cpu else emb)
        self._grid = (gh, gw)
        return torch.cat(outs)

    # ---------- fit ----------

    def fit(self, normal_crops: np.ndarray, val_ratio: float = 0.1) -> None:
        self.color = normal_crops.ndim == 4   # カラーバンクかを学習データで確定
        g = torch.Generator().manual_seed(self.seed)
        n_val = max(8, int(len(normal_crops) * val_ratio))
        perm = torch.randperm(len(normal_crops), generator=g).numpy()
        val, train = normal_crops[perm[:n_val]], normal_crops[perm[n_val:]]

        feats = self._embed(train, to_cpu=True).reshape(-1, self._embed_dim())  # (M, D) on CPU
        if len(feats) > self.pre_subsample:
            idx = torch.randperm(len(feats), generator=g)[: self.pre_subsample]
            feats = feats[idx]
        feats = feats.to(self.device)
        self.bank = self._greedy_coreset(feats, min(self.coreset_size, len(feats)))
        self._bank_sq = self._bank_t = None
        self.normal_scores = self.score_crops(val)[0]

    def _embed_dim(self) -> int:
        return 512 + 1024  # layer2 + layer3

    def _greedy_coreset(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """johnson-lindenstrauss 射影空間での greedy k-center 選択。"""
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        proj = torch.randn(x.shape[1], self.proj_dim, generator=g).to(x.device)
        z = x @ proj / (self.proj_dim ** 0.5)
        n = len(z)
        selected = torch.empty(k, dtype=torch.long, device=x.device)
        selected[0] = torch.randint(n, (1,), generator=g).item()
        dmin = torch.cdist(z, z[selected[0]].unsqueeze(0)).squeeze(1)
        for i in range(1, k):
            selected[i] = torch.argmax(dmin)
            d = torch.cdist(z, z[selected[i]].unsqueeze(0)).squeeze(1)
            dmin = torch.minimum(dmin, d)
        return x[selected].contiguous()

    # ---------- score ----------

    def _nn_dist(self, flat: torch.Tensor) -> torch.Tensor:
        """クエリ (M, D) → メモリバンクへの最近傍距離 (M,)。"""
        if self.bank is None:
            raise RuntimeError("fit() または load() でメモリバンクを構築してください")
        if self._bank_t is None:
            self._bank_sq = (self.bank.float() ** 2).sum(dim=1)  # (K,)
            self._bank_t = (self.bank.half() if self.half else self.bank.float()).T.contiguous()
        mins = []
        step = 8192
        for i in range(0, len(flat), step):
            q = flat[i : i + step]
            # |q-b|^2 = |q|^2 + |b|^2 - 2 q・b。重い内積のみ fp16 TensorCore で計算
            qb = ((q.half() if self.half else q) @ self._bank_t).float()   # (s, K)
            d2 = (q.float() ** 2).sum(dim=1, keepdim=True) + self._bank_sq - 2 * qb
            mins.append(d2.clamp_min_(0).min(dim=1).values.sqrt())
        return torch.cat(mins)

    def _cells(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """クロップ内座標のボックス → 重なる特徴セル範囲 (gy1, gy2, gx1, gx2)。±1セルの余裕込み。"""
        gh, gw = self._grid
        s = self.crop
        x1, y1, x2, y2 = box
        gx1 = max(0, int(x1 * gw / s) - 1)
        gx2 = min(gw, int(np.ceil(x2 * gw / s)) + 1)
        gy1 = max(0, int(y1 * gh / s) - 1)
        gy2 = min(gh, int(np.ceil(y2 * gh / s)) + 1)
        return gy1, max(gy2, gy1 + 1), gx1, max(gx2, gx1 + 1)

    @torch.inference_mode()
    def score_regions(
        self,
        crops: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        crop_index: list[int] | None = None,
        batch: int = 128,
    ) -> np.ndarray:
        """ボックス（クロップ内座標）に重なる特徴セルだけを見た異常スコア (len(boxes),)。

        全セル（例 20×20=400）の最近傍距離を出す score_crops と違い、実際に読む数セルしか
        バンク照合しないため候補が多いときに大幅に速い。読む範囲は score_crops + 領域max と同一。
        crop_index[i] は boxes[i] が属する crops のインデックス（省略時は i 対 i）。
        """
        feats = self._embed(crops, batch=batch)                # (N, P, D)
        _, p_cnt, d = feats.shape
        owner_of = crop_index if crop_index is not None else range(len(boxes))
        gw = self._grid[1]
        idx: list[int] = []
        owner: list[int] = []
        for i, (box, ci) in enumerate(zip(boxes, owner_of)):
            gy1, gy2, gx1, gx2 = self._cells(box)
            for gy in range(gy1, gy2):
                base = ci * p_cnt + gy * gw
                idx.extend(range(base + gx1, base + gx2))
                owner.extend([i] * (gx2 - gx1))
        sel = torch.as_tensor(idx, dtype=torch.long, device=feats.device)
        dmin = self._nn_dist(feats.reshape(-1, d)[sel])
        own = torch.as_tensor(owner, dtype=torch.long, device=feats.device)
        out = torch.zeros(len(boxes), device=feats.device)
        out.scatter_reduce_(0, own, dmin, reduce="amax", include_self=False)
        return out.cpu().numpy()

    @torch.inference_mode()
    def score_crops(self, crops: np.ndarray, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """各パッチの異常スコアと異常マップを返す。

        戻り値: (scores (N,), maps (N, gh, gw))。score = バンク最近傍距離の最大値。
        """
        feats = self._embed(crops, batch=batch)                # (N, P, D)
        n, p_cnt, d = feats.shape
        dmin = self._nn_dist(feats.reshape(-1, d))
        gh, gw = self._grid
        maps = dmin.reshape(n, gh, gw)
        return maps.amax(dim=(1, 2)).cpu().numpy(), maps.cpu().numpy()

    def threshold(self, quantile: float = 0.995, headroom: float = 1.05) -> float:
        """正常検証パッチのスコア分布から判定閾値を決める。"""
        if self.normal_scores is None:
            raise RuntimeError("normal_scores がありません（fit 済みモデルを使用してください）")
        return float(np.quantile(self.normal_scores, quantile) * headroom)

    # ---------- feedback ----------

    @torch.inference_mode()
    def absorb(self, crops: np.ndarray) -> int:
        """「これは正常」と判定されたパッチをメモリバンクへ追記する（オペレーター・フィードバック）。

        メモリバンク型異常検知の強み: **再学習なしで即時反映**できる。誤検出の位置のパッチを
        バンクに足せば、以後その見え方は「正常の記憶」に含まれ、異常スコアが下がる。
        校正分布（normal_scores）は変えない = 閾値は動かさない（動作点はオペレーターの
        感度スライダーが担う。フィードバックのたびに閾値が揺れると挙動が説明不能になる）。

        戻り値: 追加した特徴行数。バンクは追記で肥大するため、保存側（save）で
        絶対上限は設けない（数百件のフィードバック ≈ 数万行で、検索コストは線形・軽微）。
        """
        feats = self._embed(crops).reshape(-1, self._embed_dim())
        self.bank = torch.cat([self.bank, feats.to(self.bank.dtype)])
        self._bank_sq = self._bank_t = None   # 距離キャッシュを無効化
        return int(feats.shape[0])

    # ---------- 全面スキャン（objectプロファイル） ----------

    def tile_image(self, img: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """画像を crop×crop タイルに分割（重なりなし・端は内側に寄せて全域カバー）。

        戻り値: (タイル配列, 各タイルの左上座標)。グレー(H,W)/カラー(H,W,3)どちらも可。
        """
        c = self.crop
        h, w = img.shape[:2]
        ys = list(range(0, max(h - c, 0) + 1, c))
        xs = list(range(0, max(w - c, 0) + 1, c))
        if ys[-1] != max(h - c, 0):
            ys.append(max(h - c, 0))
        if xs[-1] != max(w - c, 0):
            xs.append(max(w - c, 0))
        origins = [(x, y) for y in ys for x in xs]
        tiles = np.stack([img[y : y + c, x : x + c] for x, y in origins])
        return tiles, origins

    def calibrate_dense(self, normal_images: list[np.ndarray]) -> None:
        """全面スキャンの画像レベル校正。正常画像の「タイル最大スコア」分布を保存する。

        タイル単体の校正分布（normal_scores）では全面スキャンの閾値は決められない
        （1枚に約180タイルあるため、タイル閾値そのままでは正常でも毎枚どこかが超える）。
        """
        scores = []
        for img in normal_images:
            tiles, _ = self.tile_image(img)
            s, _ = self.score_crops(tiles)
            scores.append(float(s.max()))
        self.dense_scores = np.array(scores, dtype=np.float64)

    def dense_threshold(self, headroom: float = 1.02) -> float:
        """全面スキャンの画像レベル閾値（校正した正常最大スコアの上限 × 余裕）。"""
        if self.dense_scores is None:
            raise RuntimeError("calibrate_dense 済みのバンクが必要です（objectプロファイルで再学習）")
        return float(np.max(self.dense_scores)) * headroom

    # ---------- io ----------

    def save(self, path: Path) -> None:
        torch.save(
            {
                "bank": self.bank.cpu(),
                "normal_scores": self.normal_scores,
                "crop": self.crop,
                "input_size": self.input_size,
                # 検査プロファイルのメタ情報（取り違え検出・全面スキャン校正）
                "color": self.color,
                "dense_scores": self.dense_scores,
            },
            path,
        )
        self.path = Path(path)

    def load(self, path: Path) -> "PatchCoreLite":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.bank = ckpt["bank"].to(self.device)
        self._bank_sq = self._bank_t = None
        self.normal_scores = ckpt["normal_scores"]
        self.crop = ckpt["crop"]
        self.input_size = ckpt["input_size"]
        self.color = bool(ckpt.get("color", False))          # 旧バンクはグレー
        self.dense_scores = ckpt.get("dense_scores", None)   # 旧バンクは全面スキャン未校正
        self.path = Path(path)  # フィードバック（absorb）後の保存先として記憶
        return self
