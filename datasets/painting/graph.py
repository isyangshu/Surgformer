import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_graph_modified(attention, ax):
    N = len(attention)  # 获取维度 N
    # 创建一个有向图
    G = nx.Graph()

    # 添加节点
    for i in range(N):
        G.add_node(i+1)

    # 添加边和权重
    for i in range(N):
        for j in range(N):
            if i != j and attention[i][j] > 0:  # 仅当权重大于0时添加边
                G.add_edge(i+1, j+1, weight=attention[i][j])

    # 获取边的权重列表
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]  # 修正以获取实际添加的边的权重
    edge_colors = [plt.cm.Blues(weight) for weight in edge_weights]
    # 绘制图形
    pos = nx.circular_layout(G)  # 设置节点的位置布局
    nx.draw_networkx(G, pos, arrows=None, with_labels=True, node_color='lightblue', node_size=400, font_size=10, font_weight='bold', edge_color=edge_colors, width=edge_weights, edge_cmap=plt.cm.Blues, ax=ax)
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
    sm.set_array(edge_weights)
    # plt.colorbar(sm)

    ax.axis('off')  # 关闭坐标轴

def visualize_attention_graph_modified_(attention, ax):
    N = len(attention)  # 获取维度 N
    # 创建一个有向图
    G = nx.Graph()

    # 添加节点
    for i in range(N):
        G.add_node(i+1)

    # 添加边和权重
    for i in range(N):
        for j in range(N):
            if i != j and attention[i][j] > 0:  # 仅当权重大于0时添加边
                G.add_edge(i+1, j+1, weight=attention[i][j])

    # 获取边的权重列表
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]  # 修正以获取实际添加的边的权重
    edge_colors = [plt.cm.Blues(weight) for weight in edge_weights]
    # 绘制图形
    pos = nx.circular_layout(G)  # 设置节点的位置布局
    nx.draw_networkx(G, pos, arrows=None, with_labels=True, node_color='lightblue', node_size=400, font_size=10, font_weight='bold', edge_color=edge_colors, width=edge_weights, edge_cmap=plt.cm.Blues, ax=ax)
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
    sm.set_array(edge_weights)
    plt.colorbar(sm)

    ax.axis('off')  # 关闭坐标轴

# 设置绘图环境
fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw={'width_ratios': [1, 1, 1]})
plt.subplots_adjust(wspace=0.5)

# 生成一个 16*16 的示例注意力矩阵
attention = np.random.rand(16, 16)
# 示例使用
visualize_attention_graph_modified(attention, axs[0])

# 生成一个 8*8 的示例注意力矩阵，并填充到 16*16
attention = np.random.rand(8, 8)
extended_matrix = np.full((16, 16), 0.001)  # 使用非零值填充整个矩阵
extended_matrix[8:, 8:] = attention  # 将8x8矩阵放在左上角
# 示例使用
visualize_attention_graph_modified(extended_matrix, axs[1])

# 生成一个 4*4 的示例注意力矩阵，并填充到 16*16
attention = np.random.rand(4, 4)
extended_matrix = np.full((16, 16), 0.001)  # 使用非零值填充整个矩阵
extended_matrix[12:, 12:] = attention  # 将4x4矩阵放在左上角
# 示例使用
visualize_attention_graph_modified_(extended_matrix, axs[2])

plt.tight_layout()  # 调整子图的布局
plt.show()

fig.savefig('/Users/yangshu/Downloads/工作记录/phase recognition/temporal.eps',dpi=600,format='eps')