import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_importance(directory, bin_size="1Mb"):
    if bin_size == "1Mb":
        bin_size_int = 1000000
    else:
        bin_size_int = int(bin_size)

    # データの読み込み
    df = pd.read_csv(os.path.join(directory, "feature_importance.csv"))

    # 位置情報をbin_sizeで割って整数に丸める
    df['Position_Binned'] = (df['Position'] // bin_size_int).astype(int)

    # 同じ区間内の重要度を平均して集約
    grouped = df.groupby(['Chromosome', 'Position_Binned']).mean().reset_index()

    # 染色体番号0を除外
    grouped = grouped[grouped['Chromosome'] != 0]

    # 染色体番号に対応した色分け
    base_colors = plt.cm.Pastel1(np.linspace(0, 1, 20))
    additional_colors = np.array([[0.5, 0.5, 0.5, 1], [0.6, 0.2, 0.6, 1], [0.1, 0.6, 0.6, 1], [0.9, 0.6, 0.1, 1]])
    colors = np.vstack((base_colors, additional_colors))

    # 散布図の作成
    plt.figure(figsize=(10, 6))
    for offset in np.linspace(-0.3, 0.3, 11):  # 0.1 * 5 = 0.5
        plt.scatter(grouped['Chromosome'] + offset, grouped['Position_Binned'], c=grouped['Importance'], s=grouped['Importance']*1000, cmap='viridis')
    # plt.scatter(grouped['Chromosome'], grouped['Position_Binned'], c=grouped['Importance'], s=grouped['Importance']*1000, cmap='viridis')
    plt.colorbar(label='Average Importance')
    plt.xlabel('Chromosome')
    plt.ylabel(f'Position (binned by {bin_size})')
    plt.title(f'Feature Importance Visualization (Binned by {bin_size})')

    # 23, 24番染色体のラベルをX, Yに変更
    labels = [str(i) for i in range(1, 23)] + ['X', 'Y']
    plt.xticks(list(range(1, 25)), labels)

    # x軸のラベルの背景色を染色体番号に対応する色に変更、文字色を黒に変更
    ax = plt.gca()
    for i, tick in enumerate(ax.get_xticks()):
        ax.get_xticklabels()[i].set_color('black')  # 文字色を黒に変更
        ax.get_xticklabels()[i].set_bbox(dict(facecolor=colors[int(tick)-1], edgecolor='none', pad=2))

    plt.savefig(os.path.join(directory, f'feature_importance_visualization_binned_{bin_size}.png'), dpi=300, bbox_inches='tight')

def visualize_top_n(directory, top_n=50, bin_size="1Mb"):
    # データの読み込み
    df = pd.read_csv(os.path.join(directory, "feature_importance.csv"))

    # トップNの変異を選択
    top_variants = df.sort_values(by='Importance', ascending=False).head(top_n)
    # ラベルを作成
    variant_labels = ["Chr." + str(chrom) + ":" + str(pos) + ":" + str(token) for chrom, pos, token in zip(top_variants['Chromosome'], top_variants['Position'], top_variants['Token'])]

    # 位置情報をbin_sizeで割って整数に丸める
    df['Position_Binned'] = (df['Position'] // int(bin_size.replace("Mb", "000000"))).astype(int)
    # 同じ区間内の重要度を平均して集約
    grouped = df.groupby(['Chromosome', 'Position_Binned']).mean().reset_index()
    # トップNの領域を選択
    top_regions = grouped.sort_values(by='Importance', ascending=False).head(top_n)
    # ラベルを作成
    region_labels = ["Chr." + str(chrom) + ":" + str(pos*int(bin_size.replace("Mb", "000000"))) + "-" + str((pos+1)*int(bin_size.replace("Mb", "000000"))) for chrom, pos in zip(top_regions['Chromosome'], top_regions['Position_Binned'])]

     # 染色体番号に対応した色分け
    base_colors = plt.cm.tab20c(np.linspace(0, 1, 20))
    additional_colors = np.array([[0.5, 0.5, 0.5, 1], [0.6, 0.2, 0.6, 1], [0.1, 0.6, 0.6, 1], [0.9, 0.6, 0.1, 1]])
    colors = np.vstack((base_colors, additional_colors))

    # 表の作成
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.axis('tight')
    table_data = [['Top Variants', 'Top Regions']]
    cell_colors = [['lightgray', 'lightgray']]
    for v_label, r_label in zip(variant_labels, region_labels):
        table_data.append([v_label, r_label])
        chrom_num_v = int(v_label.split(":")[0].replace("Chr.", ""))
        chrom_num_r = int(r_label.split(":")[0].replace("Chr.", ""))
        cell_colors.append([colors[chrom_num_v-1], colors[chrom_num_r-1]])
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(fontweight='bold')

    # 画像として保存
    plt.savefig(os.path.join(directory, f'top_{top_n}_table.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    directory = sys.argv[1]
    # bin_size = sys.argv[2]
    visualize_importance(directory, "1Mb")
    visualize_top_n(directory, 50, "1Mb")