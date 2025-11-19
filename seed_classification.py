"""
种子数据集分类程序
使用PCA进行降维，然后使用多种分类算法进行分类
同时对比分析直接分类与PCA降维后分类的性能差异
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SeedClassifier:
    """种子分类器类"""
    
    def __init__(self, filepath):
        """
        初始化分类器
        参数:
            filepath: 数据集文件路径，这里我用了相对路径，把数据集和seed_classification.py放在同一个文件夹下就行
        """
        self.filepath = os.path.join(os.path.dirname(__file__), 'seeds.tsv')
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_pca_train = None
        self.X_pca_test = None
        self.pca = None
        self.scaler = None
        self.label_encoder = None
        
        # 存储直接分类和PCA降维后分类的结果
        self.direct_results = None
        self.pca_results = None
        
    def load_data(self):
        """加载数据集"""
        print("=" * 80)
        print(" " * 28 + "步骤1: 加载数据")
        print("=" * 80)
        
        # 读取TSV文件
        self.data = pd.read_csv(self.filepath, sep='\t', header=None)
        
        # 设置列名
        columns = ['面积', '周长', '紧凑度', '籽粒长度', '籽粒宽度', '不对称系数', '籽粒腹沟长度', '品种']
        self.data.columns = columns
        
        print(f"数据集形状: {self.data.shape}")
        
        print(f"\n前5行数据:")
        print("-" * 80)
        # 使用格式化输出确保对齐
        header = f"{'面积':>8s}  {'周长':>8s}  {'紧凑度':>8s}  {'籽粒长度':>8s}  {'籽粒宽度':>8s}  {'不对称系数':>10s}  {'籽粒腹沟长度':>12s}  {'品种':>10s}"
        print(header)
        print("-" * 80)
        for idx in range(5):
            row = self.data.iloc[idx]
            print(f"{row['面积']:8.2f}  {row['周长']:8.2f}  {row['紧凑度']:8.4f}  "
                  f"{row['籽粒长度']:8.3f}  {row['籽粒宽度']:8.3f}  {row['不对称系数']:10.4f}  "
                  f"{row['籽粒腹沟长度']:12.3f}  {row['品种']:>10s}")
        print("-" * 80)
        
        print(f"\n数据统计信息:")
        print("=" * 80)
        # 使用格式化输出统计信息
        stats = self.data.describe()
        stats_header = f"{'统计量':>10s}  {'面积':>8s}  {'周长':>8s}  {'紧凑度':>8s}  {'籽粒长度':>8s}  {'籽粒宽度':>8s}  {'不对称系数':>10s}  {'籽粒腹沟长度':>12s}"
        print(stats_header)
        print("=" * 80)
        
        stat_names = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        stat_labels = ['样本数', '均值', '标准差', '最小值', '25%分位', '中位数', '75%分位', '最大值']
        
        # 修复索引错误：使用数字列
        numeric_columns = self.data.columns[:-1]  # 排除最后一列（品种）
        
        for stat_name, stat_label in zip(stat_names, stat_labels):
            values = stats.loc[stat_name, numeric_columns]
            if stat_name == 'count':
                print(f"{stat_label:>10s}  {values.iloc[0]:8.0f}  {values.iloc[1]:8.0f}  {values.iloc[2]:8.0f}  "
                      f"{values.iloc[3]:8.0f}  {values.iloc[4]:8.0f}  {values.iloc[5]:10.0f}  "
                      f"{values.iloc[6]:12.0f}")
            else:
                print(f"{stat_label:>10s}  {values.iloc[0]:8.2f}  {values.iloc[1]:8.2f}  {values.iloc[2]:8.4f}  "
                      f"{values.iloc[3]:8.3f}  {values.iloc[4]:8.3f}  {values.iloc[5]:10.4f}  "
                      f"{values.iloc[6]:12.3f}")
        print("=" * 80)
        
        print(f"\n类别分布:")
        class_dist = self.data['品种'].value_counts()
        print("-" * 40)
        for class_name, count in class_dist.items():
            percentage = count/len(self.data)*100
            bar = '█' * int(percentage/2)
            print(f"{class_name:12s}: {count:3d} 样本 ({percentage:5.1f}%) {bar}")
        print("-" * 40)
        
        # 分离特征和标签
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        
        # 对标签进行编码
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        
        print(f"\n特征矩阵形状: {self.X.shape}")
        print(f"标签数组形状: {self.y.shape}")
        print(f"类别编码映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        数据预处理：划分训练集和测试集，标准化
        
        参数:
            test_size: 测试集比例
            random_state: 随机种子
        """
        print("\n" + "=" * 80)
        print(" " * 26 + "步骤2: 数据预处理")
        print("=" * 80)
        
        # 划分训练集和测试集，使用分层采样保持类别比例
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"训练集大小: {self.X_train.shape[0]:3d} ({(1-test_size)*100:.0f}%)")
        print(f"测试集大小: {self.X_test.shape[0]:3d} ({test_size*100:.0f}%)")
        
        print(f"\n训练集类别分布:")
        print("-" * 40)
        unique, counts = np.unique(self.y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  类别 {self.label_encoder.classes_[u]:12s}: {c:3d} 样本")
        print("-" * 40)
        
        # 标准化处理（重要：PCA对数据尺度敏感，必须标准化）
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ 数据标准化完成")
        print("  - 均值归一化为0")
        print("  - 标准差归一化为1")
        
    def analyze_pca_components(self):
        """
        详细分析PCA主成分
        解释什么是主成分，以及如何选择合适的主成分数量
        """
        print("\n" + "=" * 80)
        print(" " * 25 + "步骤3: PCA主成分分析")
        print("=" * 80)
        
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 25 + "什么是主成分？" + " " * 38 + "│")
        print("└" + "─" * 78 + "┘")
        print("\n【基本概念】")
        print("  主成分是原始特征的线性组合，具有以下特点：")
        print("  1. 第一主成分 (PC1): 捕获数据中最大方差的方向")
        print("  2. 第二主成分 (PC2): 捕获剩余方差中最大的方向（与PC1正交）")
        print("  3. 第三主成分 (PC3): 捕获剩余方差中次大的方向（与PC1、PC2正交）")
        print("  4. 以此类推...")
        
        print("\n【通俗理解】")
        print("  ▸ 原始数据有7个特征（面积、周长、紧凑度、籽粒长度等）")
        print("  ▸ 这些特征之间可能存在相关性（例如面积大，周长通常也大）")
        print("  ▸ PCA找到新的坐标系统，消除特征间的相关性")
        print("  ▸ 第一根轴（PC1）指向数据变化最大的方向，包含最多信息")
        print("  ▸ 第二根轴（PC2）指向次大的方向，且与第一根轴垂直")
        print("  ▸ 这样可以用更少的特征保留大部分原始信息")
        
        print("\n【数学原理】")
        print("  ▸ PCA通过特征值分解协方差矩阵实现降维")
        print("  ▸ 特征向量表示主成分的方向")
        print("  ▸ 特征值表示该方向上的方差大小")
        print("  ▸ 方差越大，该主成分包含的信息越多")
        
        # 执行完整的PCA，查看所有主成分
        pca_full = PCA()
        pca_full.fit(self.X_train_scaled)
        
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 23 + "各主成分方差贡献率" + " " * 36 + "│")
        print("└" + "─" * 78 + "┘")
        print("\n" + "=" * 80)
        print(f"{'主成分':^10s} │ {'方差解释率':^14s} │ {'累积解释率':^14s} │ {'说明':^30s}")
        print("=" * 80)
        
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        for i, (var, cum_var) in enumerate(zip(pca_full.explained_variance_ratio_, cumsum)):
            status = ""
            if cum_var >= 0.99:
                status = "✓ 达到99%阈值"
            elif cum_var >= 0.95:
                status = "✓ 达到95%阈值"
            elif cum_var >= 0.90:
                status = "✓ 达到90%阈值"
            elif cum_var >= 0.85:
                status = "○ 达到85%阈值"
            
            print(f"  PC{i+1:<7d} │   {var:6.2%}      │   {cum_var:6.2%}      │ {status:30s}")
        
        print("=" * 80)
        
        # 绘制详细的方差解释率图
        self.plot_variance_ratio_detailed(pca_full)
        
        # 推荐主成分数量
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 23 + "主成分数量选择建议" + " " * 36 + "│")
        print("└" + "─" * 78 + "┘")
        print("\n" + "-" * 80)
        print(f"{'目标方差保留率':^20s} │ {'需要主成分数量':^20s} │ {'说明':^30s}")
        print("-" * 80)
        
        recommendations = [
            (0.85, "较少的主成分，快速计算"),
            (0.90, "平衡信息与效率"),
            (0.95, "保留大部分信息（推荐）"),
            (0.99, "几乎保留全部信息")
        ]
        
        for threshold, desc in recommendations:
            n_comp = np.argmax(cumsum >= threshold) + 1
            print(f"   保留 {threshold:.0%} 方差   │     需要 {n_comp} 个主成分     │ {desc:30s}")
        
        print("-" * 80)
        
        print("\n【选择建议】")
        print("  ▸ 可视化需求: 选择2-3个主成分（可以绘制2D/3D图）")
        print("  ▸ 一般应用: 选择保留90%-95%方差的主成分数量")
        print("  ▸ 高精度需求: 选择保留95%-99%方差的主成分数量")
        print("  ▸ 本数据集建议: 5-6个主成分（保留99%以上方差）")
        
        return pca_full
        
    def apply_pca(self, n_components=None):
        """
        应用PCA降维
        
        参数:
            n_components: 保留的主成分数量
                         None = 自动选择（保留99%方差以获得更好准确率）
                         int = 指定数量
                         float = 保留的方差比例
        """
        # 如果未指定，自动选择保留99%方差的主成分数量（提高准确率）
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(self.X_train_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            # 修改为保留99%方差，而不是95%
            n_components = np.argmax(cumsum >= 0.99) + 1
            print(f"\n⚙ 自动选择主成分数量: {n_components} (保留99%方差)")
        
        # 应用PCA降维
        self.pca = PCA(n_components=n_components)
        self.X_pca_train = self.pca.fit_transform(self.X_train_scaled)
        self.X_pca_test = self.pca.transform(self.X_test_scaled)
        
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 28 + "PCA降维结果" + " " * 39 + "│")
        print("└" + "─" * 78 + "┘")
        print(f"\n  ✓ PCA降维完成")
        print(f"    • 原始特征数量: 7")
        print(f"    • 降维后特征数: {self.X_pca_train.shape[1]}")
        print(f"    • 保留的总方差: {sum(self.pca.explained_variance_ratio_):.2%}")
        print(f"    • 信息损失程度: {(1-sum(self.pca.explained_variance_ratio_)):.2%}")
        
        # 显示主成分的组成
        self.show_pca_components()
        
        # 如果降到2维或3维，进行可视化
        if self.X_pca_train.shape[1] == 2:
            self.plot_pca_2d()
        elif self.X_pca_train.shape[1] == 3:
            self.plot_pca_3d()
            
    def show_pca_components(self):
        """显示主成分的具体组成（各原始特征的权重）"""
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 25 + "主成分组成分析" + " " * 38 + "│")
        print("└" + "─" * 78 + "┘")
        
        print("\n【说明】")
        print("  每个主成分都是原始7个特征的加权组合")
        print("  权重的绝对值越大，该特征对该主成分的贡献越大")
        print("  正权重表示正相关，负权重表示负相关")
        
        feature_names = ['面积', '周长', '紧凑度', '籽粒长度', '籽粒宽度', '不对称系数', '籽粒腹沟长度']
        components = self.pca.components_
        
        for i in range(min(3, len(components))):  # 显示前3个主成分
            print(f"\n{'─' * 80}")
            print(f"第{i+1}主成分 (解释方差: {self.pca.explained_variance_ratio_[i]:.2%})")
            print(f"{'─' * 80}")
            print(f"{'排名':^6s} │ {'特征名称':^15s} │ {'权重':^12s} │ {'贡献说明':^30s}")
            print(f"{'─' * 80}")
            
            # 获取权重并排序
            weights = list(zip(feature_names, components[i]))
            weights.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for rank, (feature, weight) in enumerate(weights, 1):
                sign = '正相关 +' if weight >= 0 else '负相关 -'
                contribution = ""
                if abs(weight) > 0.5:
                    contribution = "★★★ 主要贡献特征"
                elif abs(weight) > 0.3:
                    contribution = "★★☆ 重要贡献特征"
                elif abs(weight) > 0.1:
                    contribution = "★☆☆ 次要贡献特征"
                else:
                    contribution = "☆☆☆ 微弱贡献"
                
                print(f"  {rank:2d}   │ {feature:12s}   │ {sign} {abs(weight):.4f} │ {contribution:30s}")
        
        print(f"{'─' * 80}")
        
    def plot_variance_ratio_detailed(self, pca):
        """绘制详细的方差解释率图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: 各主成分方差解释率（柱状图）
        ax1 = axes[0, 0]
        bars = ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                       pca.explained_variance_ratio_,
                       color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('主成分编号', fontsize=13, fontweight='bold')
        ax1.set_ylabel('方差解释率', fontsize=13, fontweight='bold')
        ax1.set_title('各主成分的方差解释率', fontsize=15, fontweight='bold', pad=15)
        ax1.set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim([0, max(pca.explained_variance_ratio_) * 1.2])
        
        # 在柱子上添加数值
        for i, (bar, val) in enumerate(zip(bars, pca.explained_variance_ratio_)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 子图2: 累积方差解释率曲线
        ax2 = axes[0, 1]
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'o-', 
                linewidth=2.5, markersize=10, color='darkgreen', markerfacecolor='lightgreen',
                markeredgecolor='darkgreen', markeredgewidth=2)
        ax2.axhline(y=0.99, color='red', linestyle='--', linewidth=2, label='99%阈值', alpha=0.7)
        ax2.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='95%阈值', alpha=0.7)
        ax2.axhline(y=0.90, color='blue', linestyle='--', linewidth=2, label='90%阈值', alpha=0.7)
        ax2.fill_between(range(1, len(cumsum) + 1), cumsum, alpha=0.2, color='green')
        ax2.set_xlabel('主成分数量', fontsize=13, fontweight='bold')
        ax2.set_ylabel('累积方差解释率', fontsize=13, fontweight='bold')
        ax2.set_title('累积方差解释率曲线', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='lower right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim([0, 1.05])
        
        # 在曲线上标注关键点
        for i, val in enumerate(cumsum, 1):
            if val >= 0.99 and (i == 1 or cumsum[i-2] < 0.99):
                ax2.annotate(f'{i}个PC\n达到{val:.1%}', 
                           xy=(i, val), xytext=(i+0.5, val-0.08),
                           fontsize=9, ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        # 子图3: 前4个主成分的特征权重热力图
        ax3 = axes[1, 0]
        feature_names = ['面积', '周长', '紧凑度', '籽粒\n长度', '籽粒\n宽度', '不对称\n系数', '籽粒腹\n沟长度']
        n_show = min(4, len(pca.components_))
        heatmap_data = pca.components_[:n_show]
        
        im = ax3.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(feature_names)))
        ax3.set_xticklabels(feature_names, fontsize=10)
        ax3.set_yticks(range(n_show))
        ax3.set_yticklabels([f'PC{i+1}' for i in range(n_show)], fontsize=11)
        ax3.set_title('主成分的特征权重热力图', fontsize=15, fontweight='bold', pad=15)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('权重系数', fontsize=11, fontweight='bold')
        
        # 在热力图上添加数值
        for i in range(n_show):
            for j in range(len(feature_names)):
                color = 'white' if abs(heatmap_data[i, j]) > 0.5 else 'black'
                text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                              ha="center", va="center", color=color, fontsize=9, fontweight='bold')
        
        # 子图4: 特征相关性矩阵
        ax4 = axes[1, 1]
        corr_matrix = np.corrcoef(self.X_train_scaled.T)
        im2 = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, fontsize=10, rotation=45, ha='right')
        ax4.set_yticks(range(len(feature_names)))
        ax4.set_yticklabels(feature_names, fontsize=10)
        ax4.set_title('原始特征相关性矩阵\n(PCA前)', fontsize=15, fontweight='bold', pad=15)
        
        cbar2 = plt.colorbar(im2, ax=ax4)
        cbar2.set_label('相关系数', fontsize=11, fontweight='bold')
        
        # 在相关性矩阵上添加数值
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color=color, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('pca_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ 详细PCA分析图已保存为 'pca_detailed_analysis.png'")
        
    def plot_pca_2d(self):
        """绘制PCA降维后的2D散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        classes = self.label_encoder.classes_
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        markers = ['o', 's', '^']
        
        # 左图：训练集
        ax1 = axes[0]
        for i, class_name in enumerate(classes):
            mask = self.y_train == i
            ax1.scatter(self.X_pca_train[mask, 0], 
                       self.X_pca_train[mask, 1],
                       c=colors[i], 
                       label=f'{class_name} (训练集)',
                       alpha=0.7,
                       edgecolors='black',
                       s=120,
                       marker=markers[i],
                       linewidths=1.5)
        
        ax1.set_xlabel(f'第一主成分 PC1\n(解释方差: {self.pca.explained_variance_ratio_[0]:.2%})', 
                      fontsize=13, fontweight='bold')
        ax1.set_ylabel(f'第二主成分 PC2\n(解释方差: {self.pca.explained_variance_ratio_[1]:.2%})', 
                      fontsize=13, fontweight='bold')
        ax1.set_title('训练集在PCA空间的分布', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
        ax1.axvline(x=0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
        
        # 右图：测试集
        ax2 = axes[1]
        for i, class_name in enumerate(classes):
            mask = self.y_test == i
            ax2.scatter(self.X_pca_test[mask, 0], 
                       self.X_pca_test[mask, 1],
                       c=colors[i], 
                       label=f'{class_name} (测试集)',
                       alpha=0.7,
                       edgecolors='black',
                       s=120,
                       marker=markers[i],
                       linewidths=1.5)
        
        ax2.set_xlabel(f'第一主成分 PC1\n(解释方差: {self.pca.explained_variance_ratio_[0]:.2%})', 
                      fontsize=13, fontweight='bold')
        ax2.set_ylabel(f'第二主成分 PC2\n(解释方差: {self.pca.explained_variance_ratio_[1]:.2%})', 
                      fontsize=13, fontweight='bold')
        ax2.set_title('测试集在PCA空间的分布', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
        ax2.axvline(x=0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('pca_2d_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ 2D散点图已保存为 'pca_2d_scatter.png'")
        
    def plot_pca_3d(self):
        """绘制PCA降维后的3D散点图"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 11))
        ax = fig.add_subplot(111, projection='3d')
        
        classes = self.label_encoder.classes_
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        markers = ['o', 's', '^']
        
        for i, class_name in enumerate(classes):
            mask = self.y_train == i
            ax.scatter(self.X_pca_train[mask, 0], 
                      self.X_pca_train[mask, 1],
                      self.X_pca_train[mask, 2],
                      c=colors[i], 
                      label=class_name,
                      alpha=0.7,
                      edgecolors='black',
                      s=100,
                      marker=markers[i],
                      linewidths=1.5)
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%})', 
                     fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%})', 
                     fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel(f'PC3 ({self.pca.explained_variance_ratio_[2]:.1%})', 
                     fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('PCA 3D空间数据分布', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig('pca_3d_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ 3D散点图已保存为 'pca_3d_scatter.png'")
    
    def train_direct_classifiers(self, use_cv=True):
        """
        训练直接分类器（不经过PCA降维）
        
        参数:
            use_cv: 是否使用交叉验证
        """
        print("\n" + "=" * 80)
        print(" " * 22 + "步骤4A: 训练直接分类器（不降维）")
        print("=" * 80)
        
        print("\n【说明】")
        print("  使用原始的7个特征直接进行分类，不经过PCA降维")
        print("  这将作为基准，用于对比PCA降维后的分类效果")
        
        # 定义多个分类器
        classifiers = {
            'Logistic回归': LogisticRegression(max_iter=2000, random_state=42, C=10),
            '决策树': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5),
            'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            '朴素贝叶斯': GaussianNB(),
            'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
            '随机森林': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
            '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print("\n开始训练各分类器...")
        print("-" * 80)
        
        for name, clf in classifiers.items():
            print(f"\n正在训练: {name}")
            
            # 记录训练时间
            start_time = time.time()
            
            # 交叉验证（在训练集上）
            if use_cv:
                cv_scores = cross_val_score(clf, self.X_train_scaled, self.y_train, 
                                           cv=5, scoring='accuracy')
            
            # 训练模型
            clf.fit(self.X_train_scaled, self.y_train)
            
            # 预测
            y_pred_train = clf.predict(self.X_train_scaled)
            y_pred_test = clf.predict(self.X_test_scaled)
            
            # 记录训练时间
            train_time = time.time() - start_time
            
            # 计算各项指标
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test, average='weighted')
            recall = recall_score(self.y_test, y_pred_test, average='weighted')
            f1 = f1_score(self.y_test, y_pred_test, average='weighted')
            
            print(f"  测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            
            # 保存结果
            results[name] = {
                'classifier': clf,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores if use_cv else None,
                'train_time': train_time,
                'y_pred_test': y_pred_test
            }
        
        self.direct_results = results
        print("\n✓ 直接分类器训练完成")
        return results
    
    def train_pca_classifiers(self, use_cv=True):
        """
        训练PCA降维后的分类器
        
        参数:
            use_cv: 是否使用交叉验证
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "步骤4B: 训练PCA降维后的分类器")
        print("=" * 80)
        
        print("\n【说明】")
        print(f"  使用PCA降维后的{self.X_pca_train.shape[1]}个主成分进行分类")
        print(f"  保留了原始数据{sum(self.pca.explained_variance_ratio_):.2%}的方差")
        
        # 定义多个分类器
        classifiers = {
            'Logistic回归': LogisticRegression(max_iter=2000, random_state=42, C=10),
            '决策树': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5),
            'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
            '朴素贝叶斯': GaussianNB(),
            'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
            '随机森林': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
            '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print("\n开始训练各分类器...")
        print("-" * 80)
        
        for name, clf in classifiers.items():
            print(f"\n正在训练: {name}")
            
            # 记录训练时间
            start_time = time.time()
            
            # 交叉验证（在训练集上）
            if use_cv:
                cv_scores = cross_val_score(clf, self.X_pca_train, self.y_train, 
                                           cv=5, scoring='accuracy')
            
            # 训练模型
            clf.fit(self.X_pca_train, self.y_train)
            
            # 预测
            y_pred_train = clf.predict(self.X_pca_train)
            y_pred_test = clf.predict(self.X_pca_test)
            
            # 记录训练时间
            train_time = time.time() - start_time
            
            # 计算各项指标
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test, average='weighted')
            recall = recall_score(self.y_test, y_pred_test, average='weighted')
            f1 = f1_score(self.y_test, y_pred_test, average='weighted')
            
            print(f"  测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            
            # 保存结果
            results[name] = {
                'classifier': clf,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores if use_cv else None,
                'train_time': train_time,
                'y_pred_test': y_pred_test
            }
            
            # 绘制混淆矩阵
            self.plot_confusion_matrix(self.y_test, y_pred_test, name + " (PCA)")
        
        self.pca_results = results
        
        # 绘制准确率对比
        self.plot_accuracy_comparison(results, "PCA降维后")
        
        # 绘制交叉验证对比
        if use_cv:
            self.plot_cv_comparison(results)
        
        print("\n✓ PCA降维后分类器训练完成")
        return results
    
    def compare_methods(self):
        """对比直接分类和PCA降维后分类的结果"""
        print("\n" + "=" * 80)
        print(" " * 18 + "步骤5: 对比分析 - 直接分类 vs PCA降维分类")
        print("=" * 80)
        
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 24 + "对比分析说明" + " " * 39 + "│")
        print("└" + "─" * 78 + "┘")
        
        print("\n【直接分类】")
        print("  • 使用全部7个原始特征进行分类")
        print("  • 优点: 保留所有原始信息，无信息损失")
        print("  • 缺点: 特征间可能存在相关性，维度较高")
        
        print("\n【PCA降维后分类】")
        print(f"  • 使用{self.X_pca_train.shape[1]}个主成分进行分类")
        print(f"  • 保留了{sum(self.pca.explained_variance_ratio_):.2%}的原始方差")
        print("  • 优点: 降低维度，消除相关性，加快计算速度")
        print(f"  • 缺点: 信息损失{1-sum(self.pca.explained_variance_ratio_):.2%}，可解释性降低")
        
        # 创建对比表格
        comparison_data = []
        
        for clf_name in self.direct_results.keys():
            direct = self.direct_results[clf_name]
            pca = self.pca_results[clf_name]
            
            comparison_data.append({
                '分类器': clf_name,
                '直接分类_测试准确率': direct['test_accuracy'],
                'PCA降维_测试准确率': pca['test_accuracy'],
                '准确率差异': pca['test_accuracy'] - direct['test_accuracy'],
                '直接分类_F1分数': direct['f1_score'],
                'PCA降维_F1分数': pca['f1_score'],
                'F1差异': pca['f1_score'] - direct['f1_score'],
                '直接分类_训练时间': direct['train_time'],
                'PCA降维_训练时间': pca['train_time'],
                '时间差异': pca['train_time'] - direct['train_time'],
                '直接分类_交叉验证': direct['cv_scores'].mean() if direct['cv_scores'] is not None else 0,
                'PCA降维_交叉验证': pca['cv_scores'].mean() if pca['cv_scores'] is not None else 0
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # 打印详细对比表格
        print("\n" + "=" * 120)
        print(f"{'分类器':^15s} │ {'直接分类':^50s} │ {'PCA降维分类':^50s}")
        print("=" * 120)
        print(f"{'':^15s} │ {'测试准确率':^12s} {'F1分数':^12s} {'交叉验证':^12s} {'训练时间(s)':^12s} │ "
              f"{'测试准确率':^12s} {'F1分数':^12s} {'交叉验证':^12s} {'训练时间(s)':^12s}")
        print("=" * 120)
        
        for _, row in df_comparison.iterrows():
            print(f"{row['分类器']:^15s} │ "
                  f"{row['直接分类_测试准确率']:^12.4f} {row['直接分类_F1分数']:^12.4f} "
                  f"{row['直接分类_交叉验证']:^12.4f} {row['直接分类_训练时间']:^12.4f} │ "
                  f"{row['PCA降维_测试准确率']:^12.4f} {row['PCA降维_F1分数']:^12.4f} "
                  f"{row['PCA降维_交叉验证']:^12.4f} {row['PCA降维_训练时间']:^12.4f}")
        
        print("=" * 120)
        
        # 打印差异分析表格
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 26 + "性能差异分析" + " " * 39 + "│")
        print("└" + "─" * 78 + "┘")
        
        print("\n" + "=" * 90)
        print(f"{'分类器':^15s} │ {'准确率变化':^20s} │ {'F1分数变化':^20s} │ {'训练时间变化':^20s}")
        print("=" * 90)
        
        for _, row in df_comparison.iterrows():
            acc_change = row['准确率差异']
            f1_change = row['F1差异']
            time_change = row['时间差异']
            
            acc_symbol = "↑" if acc_change > 0 else ("↓" if acc_change < 0 else "→")
            f1_symbol = "↑" if f1_change > 0 else ("↓" if f1_change < 0 else "→")
            time_symbol = "↓" if time_change < 0 else ("↑" if time_change > 0 else "→")
            
            print(f"{row['分类器']:^15s} │ "
                  f"{acc_symbol} {acc_change:>7.4f} ({acc_change*100:>6.2f}%) │ "
                  f"{f1_symbol} {f1_change:>7.4f} ({f1_change*100:>6.2f}%) │ "
                  f"{time_symbol} {time_change:>7.4f}s ({time_change/row['直接分类_训练时间']*100:>6.1f}%)")
        
        print("=" * 90)
        
        # 统计总体表现
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 26 + "总体统计分析" + " " * 39 + "│")
        print("└" + "─" * 78 + "┘")
        
        avg_direct_acc = df_comparison['直接分类_测试准确率'].mean()
        avg_pca_acc = df_comparison['PCA降维_测试准确率'].mean()
        avg_direct_f1 = df_comparison['直接分类_F1分数'].mean()
        avg_pca_f1 = df_comparison['PCA降维_F1分数'].mean()
        avg_direct_time = df_comparison['直接分类_训练时间'].mean()
        avg_pca_time = df_comparison['PCA降维_训练时间'].mean()
        
        print(f"\n平均测试准确率:")
        print(f"  直接分类:     {avg_direct_acc:.4f} ({avg_direct_acc*100:.2f}%)")
        print(f"  PCA降维分类:  {avg_pca_acc:.4f} ({avg_pca_acc*100:.2f}%)")
        print(f"  差异:         {avg_pca_acc - avg_direct_acc:+.4f} ({(avg_pca_acc - avg_direct_acc)*100:+.2f}%)")
        
        print(f"\n平均F1分数:")
        print(f"  直接分类:     {avg_direct_f1:.4f}")
        print(f"  PCA降维分类:  {avg_pca_f1:.4f}")
        print(f"  差异:         {avg_pca_f1 - avg_direct_f1:+.4f}")
        
        print(f"\n平均训练时间:")
        print(f"  直接分类:     {avg_direct_time:.4f}秒")
        print(f"  PCA降维分类:  {avg_pca_time:.4f}秒")
        print(f"  差异:         {avg_pca_time - avg_direct_time:+.4f}秒 ({(avg_pca_time - avg_direct_time)/avg_direct_time*100:+.1f}%)")
        
        # 绘制对比图表
        self.plot_method_comparison(df_comparison)
        
        # 结论
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 30 + "结论" + " " * 43 + "│")
        print("└" + "─" * 78 + "┘")
        
        print("\n【准确率方面】")
        if avg_pca_acc > avg_direct_acc:
            print(f"  ✓ PCA降维后分类平均准确率提高了{(avg_pca_acc - avg_direct_acc)*100:.2f}%")
            print("  原因: PCA消除了特征间的相关性，降低了噪声影响")
        elif avg_pca_acc < avg_direct_acc:
            print(f"  ✗ PCA降维后分类平均准确率降低了{(avg_direct_acc - avg_pca_acc)*100:.2f}%")
            print(f"  原因: 信息损失({1-sum(self.pca.explained_variance_ratio_):.2%})影响了分类性能")
            print(f"  说明: 当前使用{self.X_pca_train.shape[1]}个主成分，可尝试增加主成分数量")
        else:
            print("  → 两种方法准确率相当")
        
        print("\n【训练时间方面】")
        if avg_pca_time < avg_direct_time:
            print(f"  ✓ PCA降维后训练时间减少了{(avg_direct_time - avg_pca_time)/avg_direct_time*100:.1f}%")
            print("  原因: 特征数从7个降到了" + str(self.X_pca_train.shape[1]) + "个，计算量减少")
        else:
            print(f"  注意: PCA降维后训练时间反而增加了{(avg_pca_time - avg_direct_time)/avg_direct_time*100:.1f}%")
            print("  原因: 小数据集上降维带来的优势不明显")
        
        print("\n【综合建议】")
        if avg_pca_acc >= avg_direct_acc * 0.99:  # 准确率损失小于1%
            print("  ✓ 推荐使用PCA降维:")
            print("    - 准确率损失极小或有提升")
            print("    - 降低了模型复杂度")
            print("    - 便于数据可视化")
            print("    - 消除了特征间的相关性")
        else:
            print("  建议根据实际需求选择:")
            print("    - 追求最高准确率: 使用直接分类（准确率更高）")
            print("    - 需要降维可视化: 使用PCA降维分类（可绘制2D/3D图）")
            print("    - 大规模数据: PCA降维可显著提升效率")
            print(f"    - 提升PCA性能: 增加主成分数量（当前{self.X_pca_train.shape[1]}个，建议5-6个）")
        
        return df_comparison
    
    def plot_method_comparison(self, df_comparison):
        """绘制方法对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        classifiers = df_comparison['分类器'].values
        x = np.arange(len(classifiers))
        width = 0.35
        
        # 子图1: 测试准确率对比
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x - width/2, df_comparison['直接分类_测试准确率'], width, 
                       label='直接分类', alpha=0.8, color='steelblue', edgecolor='black')
        bars2 = ax1.bar(x + width/2, df_comparison['PCA降维_测试准确率'], width, 
                       label='PCA降维分类', alpha=0.8, color='coral', edgecolor='black')
        
        ax1.set_xlabel('分类器', fontsize=12, fontweight='bold')
        ax1.set_ylabel('测试准确率', fontsize=12, fontweight='bold')
        ax1.set_title('测试准确率对比', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(classifiers, rotation=20, ha='right', fontsize=10)
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim([0.85, 1.0])
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 子图2: F1分数对比
        ax2 = axes[0, 1]
        bars3 = ax2.bar(x - width/2, df_comparison['直接分类_F1分数'], width, 
                       label='直接分类', alpha=0.8, color='steelblue', edgecolor='black')
        bars4 = ax2.bar(x + width/2, df_comparison['PCA降维_F1分数'], width, 
                       label='PCA降维分类', alpha=0.8, color='coral', edgecolor='black')
        
        ax2.set_xlabel('分类器', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1分数', fontsize=12, fontweight='bold')
        ax2.set_title('F1分数对比', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(classifiers, rotation=20, ha='right', fontsize=10)
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim([0.85, 1.0])
        
        # 子图3: 训练时间对比
        ax3 = axes[1, 0]
        bars5 = ax3.bar(x - width/2, df_comparison['直接分类_训练时间'], width, 
                       label='直接分类', alpha=0.8, color='steelblue', edgecolor='black')
        bars6 = ax3.bar(x + width/2, df_comparison['PCA降维_训练时间'], width, 
                       label='PCA降维分类', alpha=0.8, color='coral', edgecolor='black')
        
        ax3.set_xlabel('分类器', fontsize=12, fontweight='bold')
        ax3.set_ylabel('训练时间 (秒)', fontsize=12, fontweight='bold')
        ax3.set_title('训练时间对比', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels(classifiers, rotation=20, ha='right', fontsize=10)
        ax3.legend(fontsize=11)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 子图4: 准确率差异热力图
        ax4 = axes[1, 1]
        
        metrics = ['测试准确率', 'F1分数', '交叉验证', '训练时间']
        diff_data = np.array([
            df_comparison['准确率差异'].values,
            df_comparison['F1差异'].values,
            (df_comparison['PCA降维_交叉验证'] - df_comparison['直接分类_交叉验证']).values,
            df_comparison['时间差异'].values
        ])
        
        im = ax4.imshow(diff_data, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
        ax4.set_xticks(x)
        ax4.set_xticklabels(classifiers, rotation=45, ha='right', fontsize=10)
        ax4.set_yticks(range(len(metrics)))
        ax4.set_yticklabels(metrics, fontsize=11)
        ax4.set_title('PCA降维分类 vs 直接分类 (差异热力图)\n绿色表示PCA更优，红色表示直接分类更优', 
                     fontsize=14, fontweight='bold', pad=15)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('差异值', fontsize=10)
        
        # 在热力图上添加数值
        for i in range(len(metrics)):
            for j in range(len(classifiers)):
                if i < 3:  # 准确率、F1、交叉验证
                    text = ax4.text(j, i, f'{diff_data[i, j]:.4f}',
                                  ha="center", va="center", color="black", fontsize=9)
                else:  # 训练时间
                    text = ax4.text(j, i, f'{diff_data[i, j]:.3f}s',
                                  ha="center", va="center", color="black", fontsize=9)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✓ 方法对比图已保存为 'method_comparison.png'")
    
    def plot_confusion_matrix(self, y_true, y_pred, classifier_name):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：数量
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0], cbar_kws={'label': '样本数量'},
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        axes[0].set_title(f'{classifier_name}\n混淆矩阵 (数量)', fontweight='bold', fontsize=14, pad=15)
        axes[0].set_ylabel('真实标签', fontweight='bold', fontsize=12)
        axes[0].set_xlabel('预测标签', fontweight='bold', fontsize=12)
        
        # 右图：百分比
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[1], cbar_kws={'label': '百分比 (%)'},
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        axes[1].set_title(f'{classifier_name}\n混淆矩阵 (百分比)', fontweight='bold', fontsize=14, pad=15)
        axes[1].set_ylabel('真实标签', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('预测标签', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        filename = f'confusion_matrix_{classifier_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_accuracy_comparison(self, results, method_name):
        """绘制准确率对比图"""
        classifiers = list(results.keys())
        train_accuracies = [results[name]['train_accuracy'] for name in classifiers]
        test_accuracies = [results[name]['test_accuracy'] for name in classifiers]
        
        # 按测试集准确率排序
        sorted_indices = np.argsort(test_accuracies)[::-1]
        classifiers = [classifiers[i] for i in sorted_indices]
        train_accuracies = [train_accuracies[i] for i in sorted_indices]
        test_accuracies = [test_accuracies[i] for i in sorted_indices]
        
        x = np.arange(len(classifiers))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars1 = ax.bar(x - width/2, train_accuracies, width, 
                      label='训练集准确率', alpha=0.85, color='steelblue', 
                      edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, test_accuracies, width, 
                      label='测试集准确率', alpha=0.85, color='coral', 
                      edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('分类器', fontsize=13, fontweight='bold')
        ax.set_ylabel('准确率', fontsize=13, fontweight='bold')
        ax.set_title(f'{method_name}分类器性能对比', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=20, ha='right', fontsize=11)
        ax.legend(fontsize=12, loc='lower left')
        ax.set_ylim([0.82, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        
        # 在柱状图上添加数值
        for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
            ax.text(i - width/2, train_acc + 0.005, f'{train_acc:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(i + width/2, test_acc + 0.005, f'{test_acc:.3f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')
        
        plt.tight_layout()
        filename = f'accuracy_comparison_{method_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cv_comparison(self, results):
        """绘制交叉验证对比箱线图"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        cv_data = []
        labels = []
        
        for name, result in results.items():
            if result['cv_scores'] is not None:
                cv_data.append(result['cv_scores'])
                labels.append(name)
        
        bp = ax.boxplot(cv_data, labels=labels, patch_artist=True,
                       notch=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # 美化箱线图
        colors = plt.cm.Set3(np.linspace(0, 1, len(cv_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(1.5)
        
        # 设置其他元素样式
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1.5, linestyle='--')
        for cap in bp['caps']:
            cap.set(linewidth=1.5)
        for median in bp['medians']:
            median.set(color='darkblue', linewidth=2)
        
        ax.set_xlabel('分类器', fontsize=13, fontweight='bold')
        ax.set_ylabel('交叉验证准确率', fontsize=13, fontweight='bold')
        ax.set_title('5折交叉验证准确率分布\n(菱形标记表示均值)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_ylim([0.82, 1.0])
        plt.xticks(rotation=20, ha='right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('cv_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_pipeline(self, n_components=None, test_size=0.2):
        """
        运行完整的分析流程
        
        参数:
            n_components: PCA主成分数量 (None=自动选择)
            test_size: 测试集比例
        """
        print("\n" + "="*80)
        print(" "*22 + "种子数据集PCA分类分析系统")
        print("="*80)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据预处理
        self.preprocess_data(test_size=test_size)
        
        # 3. 训练直接分类器
        self.train_direct_classifiers(use_cv=True)
        
        # 4. PCA分析
        pca_full = self.analyze_pca_components()
        
        # 5. 应用PCA降维
        self.apply_pca(n_components=n_components)
        
        # 6. 训练PCA降维后的分类器
        self.train_pca_classifiers(use_cv=True)
        
        # 7. 对比分析
        comparison_df = self.compare_methods()
        
        # 8. 保存对比结果到CSV
        comparison_df.to_csv('comparison_results.csv', index=False, encoding='utf-8-sig')
        print("\n✓ 对比结果已保存为 'comparison_results.csv'")
        
        print("\n" + "="*80)
        print("✓ 分析完成！所有图表和结果已保存到当前目录。")
        print("="*80)
        
        return comparison_df


# 主程序入口
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║                     种子数据集 PCA 分类分析系统                               ║
    ║                                                                            ║
    ║  功能特点:                                                                  ║
    ║    ✓ 完整的PCA主成分分析和可视化                                              ║
    ║    ✓ 7种分类算法                                                            ║
    ║    ✓ 直接分类 vs PCA降维分类对比                                              ║
    ║    ✓ 5折交叉验证评估                                                        ║
    ║    ✓ 详细的性能报告和图表                                                    ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 创建分类器实例
    classifier = SeedClassifier('seeds.tsv')
    
    print("\n" + "=" * 80)
    print(" " * 28 + "请选择分析模式")
    print("=" * 80)
    print("\n  1. 快速模式   - 2个主成分，快速可视化")
    print("  2. 平衡模式   - 自动选择主成分数，保留99%方差")
    print("  3. 完整模式   - 6个主成分，接近完整信息")
    print("  4. 自定义模式 - 自定义所有参数")
    
    choice = input("\n请输入选项 (1-4，默认2): ").strip() or "2"
    
    print("\n" + "="*80)
    
    if choice == "1":
        print(" " * 28 + ">>> 运行快速模式 <<<")
        print("="*80)
        results = classifier.run_pipeline(n_components=2, test_size=0.2)
        
    elif choice == "2":
        print(" " * 28 + ">>> 运行平衡模式 <<<")
        print("="*80)
        results = classifier.run_pipeline(n_components=None, test_size=0.2)
        
    elif choice == "3":
        print(" " * 28 + ">>> 运行完整模式 <<<")
        print("="*80)
        results = classifier.run_pipeline(n_components=6, test_size=0.2)
        
    elif choice == "4":
        print(" " * 27 + ">>> 运行自定义模式 <<<")
        print("="*80)
        print("\n请输入参数:")
        n_comp = int(input("  主成分数量 (1-7，推荐4): "))
        test_ratio = float(input("  测试集比例 (0.1-0.4，推荐0.2): "))
        print(f"\n>>> 使用自定义参数: {n_comp}个主成分, {test_ratio:.0%}测试集")
        results = classifier.run_pipeline(n_components=n_comp, test_size=test_ratio)
    
    print("\n" + "="*80)
    print(" " * 30 + "系统程序结束！")
    print("="*80)
    print("\n此系统仅用于个人学习和研究目的，如有纰漏联系邮箱1394872816@qq.com。")
    print("\n" + "="*80)