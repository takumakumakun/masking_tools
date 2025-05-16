# Interactive Mask Creator

Segment Anything Model (SAM) を使用した，マスク画像作成ツールです．

## 機能

- 画像上でクリックするだけで，物体のマスクを自動生成
- 直感的な操作で物体と背景を指定可能
- マスクの取り消し・やり直し機能
- キーボードショートカット対応

## 必要条件

- Python 3.8以上
- PyTorch 2.0.0以上
- OpenCV 4.8.0以上
- Segment Anything Model (SAM)
- tkinter (Pythonの標準ライブラリ)

## インストール方法

1. リポジトリをクローン：
```bash
git clone [repository_url]
cd masking_tools
```

2. 必要なパッケージをインストール：
```bash
pip install -r requirements.txt
```

3. SAMモデルのダウンロード：
- [SAMモデル](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)をダウンロード
- `model`ディレクトリを作成し，ダウンロードしたモデルファイルを配置：
```bash
mkdir model
mv sam_vit_h_4b8939.pth model/
```

## 使い方

1. アプリケーションの起動：
```bash
python interactive_mask.py
```

2. 操作方法：
- **左クリック**：物体の領域を示す点（緑）を追加
- **右クリック/Control+クリック**：背景の領域を示す点（赤）を追加
- **Zキー**：直前の操作を取り消し
- **Sキー**：マスク画像を保存
- **Oキー**：新しい画像を開く
- **Cキー**：現在の操作をクリア
- **Qキー**：プログラム終了

3. マスクの保存：
- マスクはPNG形式で保存されます
- 元の画像と同じ解像度で保存されます
- デフォルトで`output`ディレクトリに保存されます

## 注意事項
- 取り消し可能な操作は最大20回までです

## ライセンス

このプロジェクトはApache 2.0ライセンスの下で公開されています．

### Segment Anything Model (SAM) について

このプロジェクトは[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)を使用しています．
SAMはMeta AI Researchによって開発され，Apache 2.0ライセンスの下で提供されています．

### 引用

SAMを使用する場合は，以下の論文を引用してください：

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```