# pip install shap html2image  # 画像化まで行う場合
import numpy as np
import shap

# ===== データ =====
tokens = ["Md", "Lw", "Ie"]
values = np.array([0.02716848347336054, 0.043611083179712296, -0.004582919621546017], dtype=float)

# ★ ここがポイント： base_values は 1次元配列(長さ1)にする！
base = np.array([0.5021830797195435], dtype=float)

# ===== Explanation を正しく構築 =====
ex = shap.Explanation(
    values=values,  # (n_features,)
    base_values=base,  # (1,)  ← 0次元にしない
    data=np.array(tokens, dtype=object),  # (n_features,)
    feature_names=list(tokens),
)

# ===== NotebookでHTML表示（動作確認） =====
# shap.plots.text(ex)  # そのままNotebookに描画
html_obj = shap.plots.text(ex, display=False)  # HTMLオブジェクトとして取得
