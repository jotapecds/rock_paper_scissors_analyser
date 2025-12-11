"""
Análise de Dados com PCA e K-Means para o Dataset Rock-Paper-Scissors
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
import cv2
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class RockPaperScissorsAnalyzer:
    def __init__(self, data_path='data'):
        """
        Inicializa o analisador com o caminho dos dados
        
        Args:
            data_path: Caminho para a pasta contendo as subpastas rock, paper, scissors
        """
        self.data_path = data_path
        self.images = []
        self.labels = []
        self.features = None
        self.labels_numeric = None
        self.label_mapping = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.reverse_mapping = {0: 'rock', 1: 'paper', 2: 'scissors'}
        
    def load_and_preprocess_images(self, target_size=(100, 100), max_images_per_class=200):
        """
        Carrega e pré-processa as imagens do dataset
        
        Args:
            target_size: Tamanho para redimensionar as imagens
            max_images_per_class: Número máximo de imagens por classe (para desenvolvimento rápido)
        """
        print("Carregando imagens...")
        
        for class_name in ['rock', 'paper', 'scissors']:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                print(f"Aviso: Pasta {class_path} não encontrada!")
                continue
                
            image_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
            image_files = image_files[:max_images_per_class]
            
            print(f"  {class_name}: {len(image_files)} imagens")
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                # Carrega imagem
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Redimensiona
                img_resized = cv2.resize(img, target_size)
                
                # Adiciona à lista
                self.images.append(img_resized)
                self.labels.append(class_name)
        
        print(f"Total de imagens carregadas: {len(self.images)}")
        print(f"Distribuição: Rock={self.labels.count('rock')}, "
              f"Paper={self.labels.count('paper')}, "
              f"Scissors={self.labels.count('scissors')}")
        
        # Converte labels para numérico
        self.labels_numeric = np.array([self.label_mapping[label] for label in self.labels])
        
        return self.images, self.labels_numeric
    
    def extract_features(self):
        """
        Extrai características das imagens para análise
        """
        print("\nExtraindo características das imagens...")
        
        features_list = []
        
        for img in self.images:
            # 1. Características de cor (histogramas RGB)
            hist_r = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
            hist_g = cv2.calcHist([img], [1], None, [16], [0, 256]).flatten()
            hist_b = cv2.calcHist([img], [2], None, [16], [0, 256]).flatten()
            
            # 2. Características de textura (gradientes)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3).flatten()
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3).flatten()
            
            # 3. Estatísticas simples
            mean_color = np.mean(img, axis=(0, 1))
            std_color = np.std(img, axis=(0, 1))
            
            # 4. Forma da mão (thresholding básico)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            hand_area = np.sum(mask > 0) / mask.size
            
            # Combina todas as características
            features = np.concatenate([
                hist_r, hist_g, hist_b,
                sobelx[:50],  # Amostra dos gradientes
                sobely[:50],
                mean_color,
                std_color,
                [hand_area]
            ])
            
            features_list.append(features)
        
        self.features = np.array(features_list)
        print(f"Forma das características: {self.features.shape}")
        
        return self.features
    
    def apply_pca(self, n_components=None, variance_threshold=0.95):
        """
        Aplica PCA para redução de dimensionalidade
        
        Args:
            n_components: Número de componentes (None para automático)
            variance_threshold: Variância mínima a ser explicada
        """
        print("\nAplicando PCA...")
        
        # Padronização dos dados
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Determina número de componentes
        if n_components is None:
            pca_full = PCA()
            pca_full.fit(features_scaled)
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            print(f"  Componentes para {variance_threshold*100:.1f}% de variância: {n_components}")
        
        # Aplica PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)
        
        # Estatísticas do PCA
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"  Forma após PCA: {features_pca.shape}")
        print(f"  Variância explicada por componente: {explained_variance[:5]}")
        print(f"  Variância acumulada total: {cumulative_variance[-1]:.4f}")
        
        # Visualiza resultados do PCA
        self._plot_pca_results(pca, features_pca)
        
        return features_pca, pca
    
    def _plot_pca_results(self, pca, features_pca):
        """
        Visualiza resultados do PCA
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Variância explicada
        axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                       np.cumsum(pca.explained_variance_ratio_), 'bo-')
        axes[0, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Número de Componentes')
        axes[0, 0].set_ylabel('Variância Explicada Acumulada')
        axes[0, 0].set_title('Variância Explicada pelo PCA')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Primeiros dois componentes
        scatter = axes[0, 1].scatter(features_pca[:, 0], features_pca[:, 1],
                                    c=self.labels_numeric, cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Primeiro Componente Principal')
        axes[0, 1].set_ylabel('Segundo Componente Principal')
        axes[0, 1].set_title('Projeção nos Dois Primeiros Componentes')
        
        # Legenda para cores
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='yellow', markersize=10, label='Rock'),
                          plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='green', markersize=10, label='Paper'),
                          plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='blue', markersize=10, label='Scissors')]
        axes[0, 1].legend(handles=legend_elements)
        
        # 3. Heatmap dos componentes principais
        im = axes[1, 0].imshow(pca.components_[:10, :50], cmap='coolwarm',
                              aspect='auto', interpolation='nearest')
        axes[1, 0].set_xlabel('Características Originais')
        axes[1, 0].set_ylabel('Componentes PCA')
        axes[1, 0].set_title('Heatmap dos Componentes Principais')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Distribuição 3D (se houver pelo menos 3 componentes)
        if features_pca.shape[1] >= 3:
            ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
            scatter_3d = ax_3d.scatter(features_pca[:, 0], features_pca[:, 1],
                                      features_pca[:, 2], c=self.labels_numeric,
                                      cmap='viridis', alpha=0.6)
            ax_3d.set_xlabel('PC1')
            ax_3d.set_ylabel('PC2')
            ax_3d.set_zlabel('PC3')
            ax_3d.set_title('Projeção 3D - Três Primeiros Componentes')
        
        plt.tight_layout()
        plt.savefig('results/pca_results/pca_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def apply_kmeans(self, features_pca, n_clusters=3, random_state=42):
        """
        Aplica K-Means clustering
        
        Args:
            features_pca: Características após PCA
            n_clusters: Número de clusters
            random_state: Seed para reprodutibilidade
        """
        print(f"\nAplicando K-Means com {n_clusters} clusters...")
        
        # Método do cotovelo para determinar número ideal de clusters
        self._elbow_method(features_pca, max_clusters=10)
        
        # Aplica K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features_pca)
        
        # Avaliação
        silhouette_avg = silhouette_score(features_pca, cluster_labels)
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        
        # Análise de correspondência clusters vs classes reais
        self._analyze_clusters(cluster_labels, kmeans, features_pca)
        
        return cluster_labels, kmeans
    
    def _elbow_method(self, features_pca, max_clusters=10):
        """
        Aplica método do cotovelo para determinar número ideal de clusters
        """
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_pca)
            inertias.append(kmeans.inertia_)
            
            if len(set(kmeans.labels_)) > 1:  # Silhouette precisa de pelo menos 2 clusters
                silhouette_scores.append(silhouette_score(features_pca, kmeans.labels_))
            else:
                silhouette_scores.append(0)
        
        # Plot do método do cotovelo
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de inércia
        axes[0].plot(range(2, max_clusters + 1), inertias, 'bo-')
        axes[0].set_xlabel('Número de Clusters (k)')
        axes[0].set_ylabel('Inércia')
        axes[0].set_title('Método do Cotovelo para K-Means')
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico de silhouette score
        axes[1].plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
        axes[1].set_xlabel('Número de Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score para Diferentes Valores de k')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/clusters/elbow_method.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _analyze_clusters(self, cluster_labels, kmeans, features_pca):
        """
        Analisa correspondência entre clusters e classes reais
        """
        print("\nAnálise de Clusters vs Classes Reais:")
        
        # Cria DataFrame para análise
        analysis_df = pd.DataFrame({
            'true_label': self.labels,
            'true_label_numeric': self.labels_numeric,
            'cluster': cluster_labels,
            'pc1': features_pca[:, 0],
            'pc2': features_pca[:, 1]
        })
        
        # Matriz de confusão clusters vs classes
        confusion = pd.crosstab(analysis_df['true_label'], analysis_df['cluster'],
                               normalize='index')
        
        print("\nDistribuição de Classes por Cluster (%):")
        print(confusion.round(3) * 100)
        
        # Visualização
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Clusters vs Classes reais
        for cluster in sorted(analysis_df['cluster'].unique()):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster]
            axes[0].scatter(cluster_data['pc1'], cluster_data['pc2'],
                          label=f'Cluster {cluster}', alpha=0.6, s=50)
        
        axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                       c='black', marker='X', s=200, label='Centróides')
        axes[0].set_xlabel('Primeiro Componente Principal')
        axes[0].set_ylabel('Segundo Componente Principal')
        axes[0].set_title('Clusters K-Means')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Heatmap da matriz de confusão
        sns.heatmap(confusion, annot=True, fmt='.2%', cmap='Blues', ax=axes[1])
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Classe Real')
        axes[1].set_title('Distribuição de Classes por Cluster')
        
        plt.tight_layout()
        plt.savefig('results/clusters/cluster_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Estatísticas por cluster
        print("\nEstatísticas por Cluster:")
        for cluster in sorted(analysis_df['cluster'].unique()):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster]
            print(f"\nCluster {cluster}:")
            print(f"  Tamanho: {len(cluster_data)} imagens")
            print(f"  Distribuição de classes:")
            class_counts = cluster_data['true_label'].value_counts()
            for class_name, count in class_counts.items():
                percentage = count / len(cluster_data) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
    
    def visualize_sample_images(self, n_samples=5):
        """
        Visualiza amostras de imagens de cada classe
        """
        print("\nVisualizando amostras de imagens...")
        
        fig, axes = plt.subplots(3, n_samples, figsize=(15, 9))
        
        for i, class_name in enumerate(['rock', 'paper', 'scissors']):
            # Encontra índices das imagens desta classe
            class_indices = [idx for idx, label in enumerate(self.labels) if label == class_name]
            sample_indices = class_indices[:n_samples]
            
            for j, idx in enumerate(sample_indices):
                axes[i, j].imshow(self.images[idx])
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_ylabel(class_name.capitalize(), fontsize=14)
        
        plt.suptitle('Amostras das Imagens por Classe', fontsize=16)
        plt.tight_layout()
        plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_full_analysis(self, max_images_per_class=200):
        """
        Executa análise completa
        """
        print("=" * 60)
        print("ANÁLISE DE DADOS - ROCK-PAPER-SCISSORS")
        print("=" * 60)
        
        # Cria diretórios de resultados
        os.makedirs('results/pca_results', exist_ok=True)
        os.makedirs('results/clusters', exist_ok=True)
        
        # 1. Carrega e pré-processa imagens
        self.load_and_preprocess_images(max_images_per_class=max_images_per_class)
        
        # 2. Visualiza amostras
        self.visualize_sample_images()
        
        # 3. Extrai características
        self.extract_features()
        
        # 4. Aplica PCA
        features_pca, pca_model = self.apply_pca(variance_threshold=0.95)
        
        # 5. Aplica K-Means
        cluster_labels, kmeans_model = self.apply_kmeans(features_pca, n_clusters=3)
        
        print("\n" + "=" * 60)
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        
        return {
            'images': self.images,
            'labels': self.labels,
            'features': self.features,
            'features_pca': features_pca,
            'pca_model': pca_model,
            'cluster_labels': cluster_labels,
            'kmeans_model': kmeans_model
        }


def main():
    """Função principal"""
    # Inicializa analisador
    analyzer = RockPaperScissorsAnalyzer(data_path='data')
    
    # Executa análise completa
    results = analyzer.run_full_analysis(max_images_per_class=200)
    
    # Salva resultados
    print("\nResultados salvos na pasta 'results/'")
    print("Arquivos gerados:")
    print("  - results/sample_images.png")
    print("  - results/pca_results/pca_analysis.png")
    print("  - results/clusters/elbow_method.png")
    print("  - results/clusters/cluster_analysis.png")


if __name__ == "__main__":
    main()