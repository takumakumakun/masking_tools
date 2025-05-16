import cv2
import numpy as np
import os
import warnings
import torch
from segment_anything import sam_model_registry, SamPredictor
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from typing import List, Tuple

# PyTorchの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)

# MacでのNSOpenPanelの警告を抑制
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

class InteractiveMaskCreator:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Interactive Mask Creator")
        
        # メインフレーム
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 画像表示用のキャンバス
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=600)
        self.canvas.pack()
        
        # 操作パネル
        self.control_frame = ttk.LabelFrame(self.main_frame, text="操作パネル", padding="5")
        self.control_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ボタン
        ttk.Button(self.control_frame, text="画像を開く(O)", command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="保存(S)", command=self.save_mask).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="取り消し(Z)", command=self.undo).pack(fill=tk.X, pady=2)
        ttk.Button(self.control_frame, text="クリア(C)", command=self.clear).pack(fill=tk.X, pady=2)
        
        # 操作方法の説明
        self.help_frame = ttk.LabelFrame(self.main_frame, text="操作方法", padding="5")
        self.help_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        help_text = """
左クリック：物体の領域を示す点（緑）
右クリック/Control+クリック：背景の領域を示す点（赤）
Zキー：直前の操作を取り消し
Sキー：マスク画像を保存
Oキー：新しい画像を開く
Cキー：現在の操作をクリア
Qキー：プログラム終了
        """
        ttk.Label(self.help_frame, text=help_text, justify=tk.LEFT).pack()
        
        # 変数の初期化
        self.image_path = None
        self.current_image = None
        self.current_image_rgb = None
        self.mask = None
        self.points: List[Tuple[int, int]] = []
        self.point_labels: List[int] = []
        self.undo_stack: List[Tuple[List[Tuple[int, int]], List[int], str]] = []
        self.photo = None
        self.max_undo_steps = 20  # 取り消し可能な最大回数
        self.original_image = None  # 元の画像を保持
        self.original_image_rgb = None  # 元のRGB画像を保持
        self.scale = 1.0  # 画像のスケール係数
        self.predictor_original = None  # 元の画像サイズ用の予測器
        self.canvas_scale = 1.0  # キャンバス上の画像スケール
        
        # SAMモデルの初期化
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = sam_model_registry["vit_h"](checkpoint="./model/sam_vit_h_4b8939.pth")
            self.model = self.model.to('cpu')
            self.predictor = SamPredictor(self.model)
        
        # イベントの設定
        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Button-2>", self.right_click)
        self.canvas.bind("<Button-3>", self.right_click)
        self.canvas.bind("<Control-Button-1>", self.right_click)
        self.root.bind("<Key>", self.key_press)
        
        # 初期画像の読み込み
        self.open_image()
        
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.image_path = file_path
            self.load_image()
            
    def load_image(self):
        try:
            # 元の画像を読み込み
            self.original_image = cv2.imread(self.image_path)
            self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # 表示用にリサイズ
            pil_image = Image.fromarray(self.original_image_rgb)
            max_size = 800
            width, height = pil_image.size
            if width > max_size or height > max_size:
                ratio = max_size / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.scale = ratio
            else:
                self.scale = 1.0
            
            # リサイズされた画像をNumPy配列に変換
            self.current_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            self.current_image_rgb = np.array(pil_image)
            
            # 両方のサイズでSAMモデルを初期化
            self.predictor.set_image(self.current_image_rgb)  # 表示用
            self.predictor_original = SamPredictor(self.model)
            self.predictor_original.set_image(self.original_image_rgb)  # マスク生成用
            
            # 状態のリセット
            self.points = []
            self.point_labels = []
            self.undo_stack = []
            self.mask = None
            
            # 表示の更新
            self.update_display()
            
        except Exception as e:
            print(f"画像の読み込みに失敗しました: {e}")
            return
            
    def update_display(self):
        if self.current_image is None:
            return
            
        try:
            # キャンバスのサイズを画像に合わせる
            canvas_width = self.current_image.shape[1]
            canvas_height = self.current_image.shape[0]
            self.canvas.config(width=canvas_width, height=canvas_height)
            
            # 表示用の画像を効率的に作成
            if self.mask is not None:
                # マスクを表示用サイズにリサイズ
                display_mask = cv2.resize(self.mask, 
                                        (canvas_width, canvas_height),
                                        interpolation=cv2.INTER_NEAREST)
                
                display_image = self.current_image.copy()
                mask_overlay = np.zeros_like(display_image)
                mask_overlay[display_mask > 0] = [0, 255, 0]  # 緑色のオーバーレイ
                cv2.addWeighted(display_image, 0.7, mask_overlay, 0.3, 0, display_image)
            else:
                display_image = self.current_image.copy()
            
            # 点の描画
            if self.points:
                for i, (x, y) in enumerate(self.points):
                    # 元の画像座標を表示用座標に変換
                    display_x = int(x * self.scale)
                    display_y = int(y * self.scale)
                    
                    # 点の色を設定（物体：緑，背景：赤）
                    color = (0, 255, 0) if self.point_labels[i] == 1 else (0, 0, 255)
                    cv2.circle(display_image, (display_x, display_y), 5, color, -1)
            
            # PILイメージに変換
            image = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image)
            
            # キャンバスの更新
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
        except Exception as e:
            print(f"表示の更新に失敗しました: {e}")
            
    def left_click(self, event):
        # キャンバス座標を元の画像座標に変換
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        
        # 現在の状態を保存
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)
        self.undo_stack.append((self.points.copy(), self.point_labels.copy(), "add_point"))
        
        # 新しい点を追加（元の画像座標）
        self.points.append((x, y))
        self.point_labels.append(1)
        self.update_mask()
        
    def right_click(self, event):
        # キャンバス座標を元の画像座標に変換
        x = int(event.x / self.scale)
        y = int(event.y / self.scale)
        
        # 現在の状態を保存
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)
        self.undo_stack.append((self.points.copy(), self.point_labels.copy(), "add_point"))
        
        # 新しい点を追加（元の画像座標）
        self.points.append((x, y))
        self.point_labels.append(0)
        self.update_mask()
        
    def update_mask(self):
        if not self.points:
            self.mask = None
            self.update_display()
            return
            
        try:
            # 元の画像サイズでマスクを予測
            masks, _, _ = self.predictor_original.predict(
                point_coords=np.array(self.points),
                point_labels=np.array(self.point_labels),
                multimask_output=False
            )
            self.mask = masks[0] * 255
            self.update_display()
            
        except Exception as e:
            print(f"マスクの更新に失敗しました: {e}")
        
    def undo(self):
        if not self.undo_stack:
            print("取り消す操作がありません")
            return
            
        # 最後の操作を取得
        last_points, last_labels, operation_type = self.undo_stack.pop()
        
        if operation_type == "add_point":
            # 最後に追加した点を削除
            self.points = last_points
            self.point_labels = last_labels
            print(f"最後の点を削除しました．残りの取り消し可能な操作: {len(self.undo_stack)}回")
        elif operation_type == "clear":
            # クリア前の状態を復元
            self.points = last_points
            self.point_labels = last_labels
            print(f"クリア操作を取り消しました．残りの取り消し可能な操作: {len(self.undo_stack)}回")
            
        self.update_mask()
        
    def clear(self):
        if self.points:  # 点が存在する場合のみ保存
            # クリア前の状態を保存（最大取り消し回数を制限）
            if len(self.undo_stack) >= self.max_undo_steps:
                self.undo_stack.pop(0)  # 最も古い操作を削除
            self.undo_stack.append((self.points.copy(), self.point_labels.copy(), "clear"))
            
            # 状態をクリア
            self.points = []
            self.point_labels = []
            self.mask = None
            self.update_display()
            print(f"すべての点をクリアしました．残りの取り消し可能な操作: {len(self.undo_stack)}回")
        
    def save_mask(self):
        if self.mask is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            initialdir="./output",
            initialfile=f"mask_{os.path.splitext(os.path.basename(self.image_path))[0]}.png"
        )
        
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # 元の画像サイズのマスクを保存
            cv2.imwrite(file_path, self.mask)
            print(f"マスクを保存しました: {file_path}")
            print(f"マスクサイズ: {self.mask.shape}")
            
    def key_press(self, event):
        if event.char.lower() == 'q':
            self.root.quit()
        elif event.char.lower() == 's':
            self.save_mask()
        elif event.char.lower() == 'z':
            self.undo()
        elif event.char.lower() == 'c':
            self.clear()
        elif event.char.lower() == 'o':
            self.open_image()

def main():
    root = tk.Tk()
    app = InteractiveMaskCreator(root)
    root.mainloop()

if __name__ == "__main__":
    main() 