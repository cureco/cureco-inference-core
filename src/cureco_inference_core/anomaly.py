"""異常検知エンジン（PatchCore-lite）— 良品学習のみで欠陥を検出する。

正常パッチ集合を WideResNet50 中間特徴（layer2+layer3）に埋め込み、
greedy k-center coreset で圧縮したメモリバンクを構築する。判定はバンクへの
最近傍距離（PatchCore [Roth+ 2022] の簡易実装）。

出自: cureco-edge（外観検査エッジ）の detect/stage2.py。**実装はこのモジュールが唯一の実体**で、
エッジ側は同梱バックボーン重みを既定にした薄い派生クラスを置くだけにしている
（同じコードを2箇所に持たない）。実測レポートは cureco-edge docs/REPORT.md・DESIGN_MODES.md を参照。

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

# 全面スキャンの既定パラメータ（V4 実測で決定。cureco-edge docs/DESIGN_MODES.md V4節）。
# 新規に学習するバンクへ適用する。学習済みバンクは保存値を使う（校正分布と対のため）
DENSE_STRIDE_DEFAULT = 64   # 96pxタイルを64px刻み（1.5倍重なり）
DENSE_TOPK_DEFAULT = 16     # 画像スコア = セル上位16平均（最大値より分離度が高い）


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
        # バッチの丸め先（インスタンス単位で上書き可能）。既定は候補判定用の128固定。
        # 全画像を1枚単位で埋め込む用途（構造モデル等）は小さいバケットに差し替える
        self.batch_buckets: tuple[int, ...] = _BATCH_BUCKETS
        self.half = self.device.type == "cuda"         # fp16 推論（Blackwell TensorCore 活用）
        self.bank: torch.Tensor | None = None          # (K, D)
        self._bank_sq: torch.Tensor | None = None      # |b|^2 キャッシュ（距離計算用）
        self._bank_t: torch.Tensor | None = None       # bank^T（推論dtype）キャッシュ
        self.defect_bank: torch.Tensor | None = None   # NG見本バンク（V5。見逃し登録の特徴）
        self._dbank_sq: torch.Tensor | None = None
        self._dbank_t: torch.Tensor | None = None
        # 各行の由来タグ（0=学習, >0=訂正ID）。訂正のアンドゥ（forget）で行単位に取り消す
        self.bank_tags: torch.Tensor | None = None
        self.defect_tags: torch.Tensor | None = None
        self.normal_scores: np.ndarray | None = None   # 正常検証パッチのスコア分布（閾値校正用）
        self.path: Path | None = None                  # save/load したバンクの場所（absorb 後の保存用）
        self.color: bool = False                       # カラーバンクか（objectプロファイル）
        self.dense_scores: np.ndarray | None = None    # 全面スキャンの画像レベル校正分布
        # 全面スキャンのパラメータ（V4）。校正と判定で必ず一致させるためバンクに保存する。
        # stride: タイル刻み（None=crop・重なりなし）。topk: 画像スコア=セル上位k平均（1=最大値）
        self.dense_stride: int | None = None
        self.dense_topk: int = 1
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
        for b in self.batch_buckets:
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

    def _bank_mats(self, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        """バンクの距離計算キャッシュ (|b|^2, bank^T) を返す。which: normal / defect。"""
        if which == "normal":
            if self.bank is None:
                raise RuntimeError("fit() または load() でメモリバンクを構築してください")
            if self._bank_t is None:
                self._bank_sq = (self.bank.float() ** 2).sum(dim=1)  # (K,)
                self._bank_t = (
                    self.bank.half() if self.half else self.bank.float()
                ).T.contiguous()
            return self._bank_sq, self._bank_t
        assert self.defect_bank is not None
        if self._dbank_t is None:
            self._dbank_sq = (self.defect_bank.float() ** 2).sum(dim=1)
            self._dbank_t = (
                self.defect_bank.half() if self.half else self.defect_bank.float()
            ).T.contiguous()
        return self._dbank_sq, self._dbank_t

    def _nn_dist(self, flat: torch.Tensor, which: str = "normal") -> torch.Tensor:
        """クエリ (M, D) → バンクへの最近傍距離 (M,)。"""
        b_sq, b_t = self._bank_mats(which)
        mins = []
        step = 8192
        for i in range(0, len(flat), step):
            q = flat[i : i + step]
            # |q-b|^2 = |q|^2 + |b|^2 - 2 q・b。重い内積のみ fp16 TensorCore で計算
            qb = ((q.half() if self.half else q) @ b_t).float()   # (s, K)
            d2 = (q.float() ** 2).sum(dim=1, keepdim=True) + b_sq - 2 * qb
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
        ng_floor: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """ボックス（クロップ内座標）に重なる特徴セルだけを見た異常スコア (len(boxes),)。

        全セル（例 20×20=400）の最近傍距離を出す score_crops と違い、実際に読む数セルしか
        バンク照合しないため候補が多いときに大幅に速い。読む範囲は score_crops + 領域max と同一。
        crop_index[i] は boxes[i] が属する crops のインデックス（省略時は i 対 i）。

        ng_floor を渡すと NG見本バンク照合（V5）も行い、(scores, ng_hit) を返す。
        ng_hit[i] は「ボックス内のどこかのセルが、正常バンクより登録済みNG見本に近く、
        かつ正常距離が ng_floor 以上（=それなりに異常）」。閾値未満の候補を確定に
        押し上げる判断材料で、NG見本が未登録なら全 False。
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
        flat = feats.reshape(-1, d)[sel]
        dmin = self._nn_dist(flat)
        own = torch.as_tensor(owner, dtype=torch.long, device=feats.device)
        out = torch.zeros(len(boxes), device=feats.device)
        out.scatter_reduce_(0, own, dmin, reduce="amax", include_self=False)
        if ng_floor is None:
            return out.cpu().numpy()
        ng_hit = np.zeros(len(boxes), dtype=bool)
        if self.defect_bank is not None and len(flat):
            dng = self._nn_dist(flat, which="defect")
            cell_hit = ((dng <= dmin) & (dmin >= ng_floor)).float()
            hits = torch.zeros(len(boxes), device=feats.device)
            hits.scatter_reduce_(0, own, cell_hit, reduce="amax", include_self=False)
            ng_hit = hits.cpu().numpy() > 0
        return out.cpu().numpy(), ng_hit

    @torch.inference_mode()
    def score_crops(
        self, crops: np.ndarray, batch: int = 128, with_ng: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """各パッチの異常スコアと異常マップを返す。

        戻り値: (scores (N,), maps (N, gh, gw))。score = バンク最近傍距離の最大値。
        with_ng=True では (scores, maps, ng_maps) を返す（ng_maps は NG見本バンクへの
        距離マップ。見本が未登録なら None。全面スキャンの押し上げ判定・V5×V4 用）。
        """
        feats = self._embed(crops, batch=batch)                # (N, P, D)
        n, p_cnt, d = feats.shape
        flat = feats.reshape(-1, d)
        dmin = self._nn_dist(flat)
        gh, gw = self._grid
        maps = dmin.reshape(n, gh, gw)
        if not with_ng:
            return maps.amax(dim=(1, 2)).cpu().numpy(), maps.cpu().numpy()
        ng_maps = (
            self._nn_dist(flat, which="defect").reshape(n, gh, gw).cpu().numpy()
            if self.defect_bank is not None else None
        )
        return maps.amax(dim=(1, 2)).cpu().numpy(), maps.cpu().numpy(), ng_maps

    def threshold(self, quantile: float = 0.995, headroom: float = 1.05) -> float:
        """正常検証パッチのスコア分布から判定閾値を決める。"""
        if self.normal_scores is None:
            raise RuntimeError("normal_scores がありません（fit 済みモデルを使用してください）")
        return float(np.quantile(self.normal_scores, quantile) * headroom)

    # ---------- feedback ----------

    @torch.inference_mode()
    def _ensure_tags(self) -> None:
        """由来タグを bank / defect_bank と同じ長さに揃える（未設定や旧バンクは学習=0扱い）。"""
        if self.bank is not None and (
            self.bank_tags is None or len(self.bank_tags) != len(self.bank)
        ):
            self.bank_tags = torch.zeros(len(self.bank), dtype=torch.long, device=self.bank.device)
        if self.defect_bank is not None and (
            self.defect_tags is None or len(self.defect_tags) != len(self.defect_bank)
        ):
            self.defect_tags = torch.zeros(
                len(self.defect_bank), dtype=torch.long, device=self.defect_bank.device)

    def absorb(self, crops: np.ndarray, tag: int = 0) -> int:
        """「これは正常」と判定されたパッチをメモリバンクへ追記する（オペレーター・フィードバック）。

        メモリバンク型異常検知の強み: **再学習なしで即時反映**できる。誤検出の位置のパッチを
        バンクに足せば、以後その見え方は「正常の記憶」に含まれ、異常スコアが下がる。
        校正分布（normal_scores）は変えない = 閾値は動かさない（動作点はオペレーターの
        感度スライダーが担う。フィードバックのたびに閾値が揺れると挙動が説明不能になる）。

        tag: この行の由来（訂正ID）。>0 を渡すと forget(tag) で行単位に取り消せる。
        戻り値: 追加した特徴行数。バンクは追記で肥大するため、保存側（save）で
        絶対上限は設けない（数百件のフィードバック ≈ 数万行で、検索コストは線形・軽微）。
        """
        self._ensure_tags()
        feats = self._embed(crops).reshape(-1, self._embed_dim())
        self.bank = torch.cat([self.bank, feats.to(self.bank.dtype)])
        self.bank_tags = torch.cat([
            self.bank_tags,
            torch.full((feats.shape[0],), int(tag), dtype=torch.long, device=self.bank.device)])
        self._bank_sq = self._bank_t = None   # 距離キャッシュを無効化
        return int(feats.shape[0])

    def forget(self, tag: int) -> tuple[int, int]:
        """由来タグ tag の行を bank / defect_bank から取り除く（訂正のアンドゥ）。

        追記の逆操作。中間の行を消してもタグで選ぶので順序に依存しない。
        戻り値: (取り消した正常行数, 取り消したNG見本行数)。
        """
        self._ensure_tags()
        n_good = n_ng = 0
        if self.bank is not None and self.bank_tags is not None:
            keep = self.bank_tags != int(tag)
            n_good = int((~keep).sum())
            if n_good:
                self.bank = self.bank[keep]
                self.bank_tags = self.bank_tags[keep]
                self._bank_sq = self._bank_t = None
        if self.defect_bank is not None and self.defect_tags is not None:
            keep = self.defect_tags != int(tag)
            n_ng = int((~keep).sum())
            if n_ng:
                self.defect_bank = self.defect_bank[keep] if keep.any() else None
                self.defect_tags = self.defect_tags[keep] if keep.any() else None
                self._dbank_sq = self._dbank_t = None
        return n_good, n_ng

    #: NG見本として登録してよい下限（正常バンクへの距離 ÷ 校正しきい値）。
    #: **相対的な選別（パッチ内のピーク比）だけでは足りない。** 背景など「正常そのもの」を
    #: 切り出すと、絶対値は低いのにパッチ内では上位になり、正常な見え方が
    #: 「欠陥の見本」として登録されてしまう。以後その見え方が**全画像で**押し上げられ、
    #: 1回の誤クリックで検査が使い物にならなくなる（実測: 誤登録1件で確定欠陥429件）。
    NG_REGISTER_FLOOR = 0.7

    #: 指した所が**自分の周囲から浮いている**ことを求める比（パッチ内の 最大セル / 中央値）。
    #:
    #: 絶対床（NG_REGISTER_FLOOR）だけでは歯止めにならない。実測（VisA capsules・
    #: 良品画像の無作為な位置100点／正解マスク中心25点）:
    #:
    #: | 基準 | 良品を受理 | 欠陥を受理 |
    #: |---|---|---|
    #: | 絶対床のみ（従来） | **87.5%** | 100% |
    #: | 画像内で上位0.5%以内 | 52.0% | 100% |
    #: | **局所コントラスト 1.3以上** | **28.0%** | 80.0% |
    #:
    #: 絶対床が効かないのは原理的な話で、**しきい値未満を拾うのがNG登録の用途**だから、
    #: 床をしきい値付近まで上げると機能自体が死ぬ。画像内パーセンタイルも効かない
    #: （96pxパッチの「最大」セルを画像全体と比べるので、どこを指しても上位に来る）。
    #:
    #: 局所コントラストなら分離する（AUROC 0.856）。本物の欠陥は周囲の正常テクスチャから
    #: 浮いているが、無作為に指した正常な場所は周囲と似ているため。
    #: 見逃しの2割を取りこぼすが、**拒否してもラベルは残して次の学習に回す**設計なので
    #: 失われるのは即時反映だけ。正常を欠陥として覚える害（実測: 誤登録1件で確定欠陥429件）
    #: のほうが重い。
    NG_REGISTER_CONTRAST = 1.3

    @torch.inference_mode()
    def register_defect(self, crops: np.ndarray, tag: int = 0) -> int:
        """「これは欠陥（見逃し）」のパッチをNG見本バンクへ登録する（absorb の対称形・V5）。

        以後、閾値未満の候補でも「正常バンクより登録済みNG見本に近い」セルがあれば
        確定に押し上げられる（score_regions の ng_floor 照合）。少数ショットの欠陥記憶。

        クロップ全セルを登録すると欠陥の周囲の正常テクスチャまで「欠陥の見本」になり
        誤押し上げの種になるため、クロップ内で異常スコアが立っているセルだけを選ぶ
        （ピークの50%以上・最大16セル）。

        正常な場所を誤って指したときに「正常の見え方」を欠陥として覚えないよう、
        2つの歯止めを課す:
          ・絶対的な下限（NG_REGISTER_FLOOR）
          ・**指した所が周囲から浮いていること**（NG_REGISTER_CONTRAST）。実測では
            こちらが効く（絶対床だけでは良品の無作為な位置を87.5%受理していた）
        **戻り値0は「登録しなかった」**を意味する（呼び出し側は現場にそう伝えること。
        黙って何も起きないと、運用者は登録できたと思い込む）。
        """
        feats = self._embed(crops)                             # (N, P, D)
        n, p_cnt, d = feats.shape
        flat = feats.reshape(-1, d)
        dmin = self._nn_dist(flat).reshape(n, p_cnt)           # 正常バンクへの距離
        floor = 0.0
        if self.normal_scores is not None:                # 校正済みなら絶対的な床を課す
            floor = float(self.threshold()) * self.NG_REGISTER_FLOOR
        rows = []
        for i in range(n):
            di = dmin[i]
            # 周囲から浮いていなければ、正常な場所を指したとみなして登録しない
            med = float(di.median())
            if float(di.max()) < med * self.NG_REGISTER_CONTRAST:
                continue
            keep = torch.nonzero((di >= di.max() * 0.5) & (di >= floor)).squeeze(1)
            if len(keep) > 16:
                keep = keep[torch.argsort(di[keep], descending=True)[:16]]
            if len(keep):
                rows.append(feats[i][keep])
        if not rows:                                      # 正常にしか見えない → 登録しない
            return 0
        add = torch.cat(rows)
        self._ensure_tags()
        self.defect_bank = (
            add if self.defect_bank is None
            else torch.cat([self.defect_bank, add.to(self.defect_bank.dtype)])
        )
        new_tags = torch.full((add.shape[0],), int(tag), dtype=torch.long, device=add.device)
        self.defect_tags = (
            new_tags if self.defect_tags is None
            else torch.cat([self.defect_tags, new_tags]))
        self._dbank_sq = self._dbank_t = None
        return int(add.shape[0])

    # ---------- 全面スキャン（objectプロファイル） ----------

    def tile_image(
        self, img: np.ndarray, stride: int | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """画像を crop×crop タイルに分割（端は内側に寄せて全域カバー）。

        stride を crop 未満にすると重なりタイルになる（V4: タイル境界をまたぐ欠陥にも
        「中央に収まる窓」ができ、分離度が上がる）。省略時は self.dense_stride → crop。
        戻り値: (タイル配列, 各タイルの左上座標)。グレー(H,W)/カラー(H,W,3)どちらも可。
        """
        c = self.crop
        s = stride or self.dense_stride or c
        h, w = img.shape[:2]
        ys = list(range(0, max(h - c, 0) + 1, s))
        xs = list(range(0, max(w - c, 0) + 1, s))
        if ys[-1] != max(h - c, 0):
            ys.append(max(h - c, 0))
        if xs[-1] != max(w - c, 0):
            xs.append(max(w - c, 0))
        origins = [(x, y) for y in ys for x in xs]
        tiles = np.stack([img[y : y + c, x : x + c] for x, y in origins])
        return tiles, origins

    def dense_image_score(self, maps: np.ndarray) -> float:
        """全タイルの異常マップ → 画像レベルスコア（セル上位 dense_topk の平均）。

        最大値（topk=1）は単一セルのノイズに敏感で、正常側の裾が重くなる。
        上位k平均は「本物の欠陥は隣接セルまとめて立つ」性質を使った分離度改善（V4）。
        """
        flat = maps.reshape(-1)
        k = min(max(self.dense_topk, 1), flat.size)
        if k == 1:
            return float(flat.max())
        top = np.partition(flat, flat.size - k)[flat.size - k:]
        return float(top.mean())

    def calibrate_dense(self, normal_images: list[np.ndarray]) -> None:
        """全面スキャンの画像レベル校正。正常画像の画像レベルスコア分布を保存する。

        タイル単体の校正分布（normal_scores）では全面スキャンの閾値は決められない
        （1枚に約180タイルあるため、タイル閾値そのままでは正常でも毎枚どこかが超える）。
        スコアの定義（dense_stride / dense_topk）は判定時と同一（バンクに保存される）。
        """
        scores = []
        for img in normal_images:
            tiles, _ = self.tile_image(img)
            _, maps = self.score_crops(tiles)
            scores.append(self.dense_image_score(maps))
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
                # NG見本バンク（V5・見逃し登録）。未登録なら None
                "defect_bank": None if self.defect_bank is None else self.defect_bank.cpu(),
                # 全面スキャンのスコア定義（V4）。校正分布と対で保存する
                "dense_stride": self.dense_stride,
                "dense_topk": self.dense_topk,
                # 由来タグ（訂正アンドゥ用）。訂正のたびに保存されるので版をまたいで残る
                "bank_tags": None if self.bank_tags is None else self.bank_tags.cpu(),
                "defect_tags": None if self.defect_tags is None else self.defect_tags.cpu(),
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
        db = ckpt.get("defect_bank", None)                   # 旧バンクはNG見本なし
        self.defect_bank = None if db is None else db.to(self.device)
        self._dbank_sq = self._dbank_t = None
        self.dense_stride = ckpt.get("dense_stride", None)   # 旧バンクは重なりなし
        self.dense_topk = int(ckpt.get("dense_topk", 1))     # 旧バンクは最大値
        bt = ckpt.get("bank_tags", None)
        self.bank_tags = None if bt is None else bt.to(self.device)
        dt = ckpt.get("defect_tags", None)
        self.defect_tags = None if dt is None else dt.to(self.device)
        self._ensure_tags()   # タグの無い（学習のみの）バンクは全行=0（学習由来）に揃える
        self.path = Path(path)  # フィードバック（absorb）後の保存先として記憶
        return self
