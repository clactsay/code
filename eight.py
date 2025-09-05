
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, pearsonr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("Bioinformatics Research Code: Music Therapy Impact on Schizophrenia Brain Imaging Analysis")
print("=" * 80)
print()

class BrainImagingAnalyzer:
    """
    A comprehensive class for analyzing brain imaging data in the context of 
    music therapy interventions for schizophrenia research.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.pca = PCA()
        
    def simulate_brain_imaging_data(self, n_participants=100, n_regions=68):
        """
        Simulate brain imaging data for demonstration purposes.
        In real research, this would be replaced with actual neuroimaging data loading.
        """
        print(f"Simulating brain imaging data for {n_participants} participants across {n_regions} brain regions...")
        
        # Simulate participant demographics
        participants = pd.DataFrame({
            'participant_id': range(1, n_participants + 1),
            'age': np.random.normal(35, 8, n_participants),
            'gender': np.random.choice(['M', 'F'], n_participants),
            'diagnosis_years': np.random.exponential(5, n_participants),
            'medication_dosage': np.random.gamma(2, 2, n_participants),
            'baseline_panss': np.random.normal(75, 15, n_participants)  # PANSS baseline scores
        })
        
        # Simulate pre-therapy brain imaging data
        # Key regions affected in schizophrenia: prefrontal cortex, temporal lobe, limbic system
        pre_therapy_data = np.random.multivariate_normal(
            mean=np.zeros(n_regions),
            cov=np.eye(n_regions) * 0.5,
            size=n_participants
        )
        
        # Simulate post-therapy data with treatment effects
        # Music therapy typically shows improvements in emotional processing regions
        treatment_effect = np.zeros(n_regions)
        # Simulate stronger effects in emotion-related regions (hypothetical indices)
        emotional_regions = [10, 15, 25, 30, 45, 50]  # Amygdala, hippocampus, etc.
        auditory_regions = [20, 35, 40]  # Auditory cortex regions
        
        for region in emotional_regions:
            treatment_effect[region] = np.random.normal(0.3, 0.1)  # Positive effect
        for region in auditory_regions:
            treatment_effect[region] = np.random.normal(0.4, 0.1)  # Strong auditory effect
            
        post_therapy_data = pre_therapy_data + treatment_effect + np.random.normal(0, 0.1, (n_participants, n_regions))
        
        # Create region names (simplified for demonstration)
        region_names = [f'Region_{i:02d}' for i in range(n_regions)]
        region_names[10] = 'Amygdala_L'
        region_names[15] = 'Hippocampus_L'
        region_names[20] = 'Auditory_Cortex_L'
        region_names[25] = 'Amygdala_R'
        region_names[30] = 'Hippocampus_R'
        region_names[35] = 'Auditory_Cortex_R'
        region_names[40] = 'Superior_Temporal_Gyrus'
        region_names[45] = 'Prefrontal_Cortex_L'
        region_names[50] = 'Prefrontal_Cortex_R'
        
        return participants, pre_therapy_data, post_therapy_data, region_names
    
    def statistical_analysis(self, pre_data, post_data, region_names, alpha=0.05):
        """
        Perform comprehensive statistical analysis comparing pre- and post-therapy brain activity.
        """
        print("\nPerforming Statistical Analysis...")
        print("-" * 50)
        
        n_regions = len(region_names)
        results = []
        
        for i, region in enumerate(region_names):
            # Paired t-test for each brain region
            t_stat, p_value = ttest_rel(pre_data[:, i], post_data[:, i])
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((np.std(pre_data[:, i])**2 + np.std(post_data[:, i])**2) / 2))
            cohens_d = (np.mean(post_data[:, i]) - np.mean(pre_data[:, i])) / pooled_std
            
            # Mean change
            mean_change = np.mean(post_data[:, i] - pre_data[:, i])
            
            results.append({
                'region': region,
                'pre_mean': np.mean(pre_data[:, i]),
                'post_mean': np.mean(post_data[:, i]),
                'mean_change': mean_change,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'cohens_d': cohens_d,
                'effect_size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
            })
        
        results_df = pd.DataFrame(results)
        
        # Bonferroni correction for multiple comparisons
        results_df['p_value_corrected'] = results_df['p_value'] * n_regions
        results_df['significant_corrected'] = results_df['p_value_corrected'] < alpha
        
        # Display significant results
        significant_regions = results_df[results_df['significant_corrected']].sort_values('p_value')
        
        print(f"Significant regions after Bonferroni correction (α = {alpha}):")
        if len(significant_regions) > 0:
            for _, row in significant_regions.iterrows():
                print(f"  {row['region']}: t={row['t_statistic']:.3f}, p={row['p_value']:.4f}, "
                      f"Cohen's d={row['cohens_d']:.3f} ({row['effect_size']} effect)")
        else:
            print("  No regions survived multiple comparison correction")
            
        return results_df
    
    def visualization_analysis(self, pre_data, post_data, region_names, results_df):
        """
        Create comprehensive visualizations for the brain imaging analysis.
        """
        print("\nGenerating Visualizations...")
        print("-" * 30)
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Heatmap of brain activity changes
        ax1 = plt.subplot(3, 3, 1)
        change_data = post_data - pre_data
        mean_changes = np.mean(change_data, axis=0)
        
        # Create a simplified "brain map" visualization
        brain_matrix = mean_changes.reshape(8, -1)  # Reshape for visualization
        im1 = plt.imshow(brain_matrix, cmap='RdBu_r', aspect='auto')
        plt.colorbar(im1, ax=ax1)
        plt.title('Mean Activity Changes Across Brain Regions\n(Post - Pre Therapy)', fontsize=12)
        plt.xlabel('Region Groups')
        plt.ylabel('Brain Areas')
        
        # 2. Distribution of effect sizes
        ax2 = plt.subplot(3, 3, 2)
        plt.hist(results_df['cohens_d'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='No Effect')
        plt.axvline(0.2, color='orange', linestyle='--', label='Small Effect')
        plt.axvline(0.5, color='green', linestyle='--', label='Medium Effect')
        plt.axvline(0.8, color='purple', linestyle='--', label='Large Effect')
        plt.xlabel("Cohen's d (Effect Size)")
        plt.ylabel('Number of Regions')
        plt.title('Distribution of Effect Sizes Across Brain Regions')
        plt.legend()
        
        # 3. Volcano plot (effect size vs significance)
        ax3 = plt.subplot(3, 3, 3)
        x = results_df['cohens_d']
        y = -np.log10(results_df['p_value'])
        colors = ['red' if sig else 'gray' for sig in results_df['significant_corrected']]
        plt.scatter(x, y, c=colors, alpha=0.6)
        plt.axhline(-np.log10(0.05), color='blue', linestyle='--', label='p = 0.05')
        plt.axvline(0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel("Cohen's d (Effect Size)")
        plt.ylabel('-log10(p-value)')
        plt.title('Volcano Plot: Effect Size vs Statistical Significance')
        plt.legend()
        
        # 4. Before vs After comparison for top regions
        ax4 = plt.subplot(3, 3, 4)
        top_regions = results_df.nlargest(5, 'cohens_d')
        top_indices = [i for i, name in enumerate(region_names) if name in top_regions['region'].values]
        
        pre_means = [np.mean(pre_data[:, i]) for i in top_indices]
        post_means = [np.mean(post_data[:, i]) for i in top_indices]
        
        x_pos = np.arange(len(top_indices))
        width = 0.35
        
        plt.bar(x_pos - width/2, pre_means, width, label='Pre-Therapy', alpha=0.7, color='lightcoral')
        plt.bar(x_pos + width/2, post_means, width, label='Post-Therapy', alpha=0.7, color='lightblue')
        
        plt.xlabel('Brain Regions')
        plt.ylabel('Mean Activity Level')
        plt.title('Top 5 Regions with Largest Treatment Effects')
        plt.xticks(x_pos, [region_names[i].replace('_', '\n') for i in top_indices], rotation=45)
        plt.legend()
        
        # 5. Correlation matrix of regional changes
        ax5 = plt.subplot(3, 3, 5)
        # Select a subset of key regions for correlation analysis
        key_regions_idx = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        key_names = [region_names[i].replace('_', ' ') for i in key_regions_idx]
        change_subset = change_data[:, key_regions_idx]
        
        corr_matrix = np.corrcoef(change_subset.T)
        im2 = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im2, ax=ax5)
        plt.xticks(range(len(key_names)), key_names, rotation=45, ha='right')
        plt.yticks(range(len(key_names)), key_names)
        plt.title('Correlation Matrix of Treatment Effects\n(Key Brain Regions)')
        
        # 6. Individual participant trajectories for key regions
        ax6 = plt.subplot(3, 3, 6)
        amygdala_idx = 10  # Amygdala left
        auditory_idx = 20  # Auditory cortex left
        
        participants_subset = np.random.choice(range(len(pre_data)), 20, replace=False)
        
        for p in participants_subset:
            plt.plot([0, 1], [pre_data[p, amygdala_idx], post_data[p, amygdala_idx]], 
                    'b-', alpha=0.3, linewidth=0.5)
        
        plt.plot([0, 1], [np.mean(pre_data[:, amygdala_idx]), np.mean(post_data[:, amygdala_idx])], 
                'r-', linewidth=3, label='Mean Trajectory')
        
        plt.xlim(-0.1, 1.1)
        plt.xticks([0, 1], ['Pre-Therapy', 'Post-Therapy'])
        plt.ylabel('Activity Level')
        plt.title(f'Individual Trajectories: {region_names[amygdala_idx]}')
        plt.legend()
        
        # 7. Effect size by brain region type
        ax7 = plt.subplot(3, 3, 7)
        
        # Categorize regions
        emotional_regions = ['Amygdala_L', 'Amygdala_R', 'Hippocampus_L', 'Hippocampus_R']
        auditory_regions = ['Auditory_Cortex_L', 'Auditory_Cortex_R', 'Superior_Temporal_Gyrus']
        cognitive_regions = ['Prefrontal_Cortex_L', 'Prefrontal_Cortex_R']
        
        categories = []
        for region in results_df['region']:
            if region in emotional_regions:
                categories.append('Emotional')
            elif region in auditory_regions:
                categories.append('Auditory')
            elif region in cognitive_regions:
                categories.append('Cognitive')
            else:
                categories.append('Other')
        
        results_df['category'] = categories
        
        category_effects = results_df.groupby('category')['cohens_d'].apply(list)
        
        plt.boxplot([category_effects[cat] for cat in category_effects.index], 
                   labels=category_effects.index)
        plt.ylabel("Cohen's d (Effect Size)")
        plt.title('Treatment Effects by Brain Region Category')
        plt.xticks(rotation=45)
        
        # 8. Statistical significance overview
        ax8 = plt.subplot(3, 3, 8)
        sig_counts = results_df['significant_corrected'].value_counts()
        labels = ['Non-Significant', 'Significant']
        sizes = [sig_counts.get(False, 0), sig_counts.get(True, 0)]
        colors = ['lightgray', 'lightgreen']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Statistical Significance\n(Bonferroni Corrected, α=0.05)')
        
        # 9. Treatment response prediction
        ax9 = plt.subplot(3, 3, 9)
        
        # Create a composite score for treatment response
        key_regions_idx = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        treatment_response = np.mean(change_data[:, key_regions_idx], axis=1)
        
        plt.hist(treatment_response, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(treatment_response), color='red', linestyle='--', 
                   label=f'Mean Response: {np.mean(treatment_response):.3f}')
        plt.axvline(0, color='black', linestyle=':', alpha=0.5, label='No Change')
        plt.xlabel('Treatment Response Score')
        plt.ylabel('Number of Participants')
        plt.title('Distribution of Overall Treatment Response')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('brain_imaging_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results_df
    
    def machine_learning_analysis(self, pre_data, post_data, participants_df):
        """
        Apply machine learning techniques to predict treatment response and classify participants.
        """
        print("\nMachine Learning Analysis...")
        print("-" * 35)
        
        # Prepare features: combine pre-therapy brain data with demographic info
        demographic_features = participants_df[['age', 'diagnosis_years', 'medication_dosage', 'baseline_panss']].values
        
        # Normalize demographic features
        demographic_features = self.scaler.fit_transform(demographic_features)
        
        # Combine brain imaging and demographic data
        X = np.hstack([pre_data, demographic_features])
        
        # Create target variable: treatment response (binary classification)
        treatment_response = np.mean(post_data - pre_data, axis=1)
        y = (treatment_response > np.median(treatment_response)).astype(int)  # High vs Low responders
        
        print(f"Dataset: {X.shape[0]} participants, {X.shape[1]} features")
        print(f"Treatment response distribution: {np.sum(y)} high responders, {np.sum(1-y)} low responders")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=self.random_state, stratify=y)
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': (y_pred == y_test).mean(),
                'predictions': y_pred
            }
            
            print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Test accuracy: {results[name]['test_accuracy']:.3f}")
            
            # Classification report
            print("  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Low Responder', 'High Responder'], 
                                      zero_division=0))
        
        # Feature importance analysis (using Random Forest)
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_importance = rf_model.feature_importances_
            
            # Create feature names
            n_brain_regions = pre_data.shape[1]
            feature_names = [f'Brain_Region_{i:02d}' for i in range(n_brain_regions)]
            feature_names.extend(['Age', 'Diagnosis_Years', 'Medication_Dosage', 'Baseline_PANSS'])
            
            # Get top 15 most important features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head(15)
            
            print("\nTop 15 Most Important Features for Treatment Response Prediction:")
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return results, X_test, y_test
    
    def dimensionality_analysis(self, pre_data, post_data, region_names):
        """
        Perform dimensionality reduction and clustering analysis.
        """
        print("\nDimensionality Reduction and Clustering Analysis...")
        print("-" * 55)
        
        # Combine pre and post data for comprehensive analysis
        combined_data = np.hstack([pre_data, post_data])
        
        # Principal Component Analysis
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(combined_data)
        
        print(f"PCA Results:")
        print(f"  Explained variance ratio (first 5 components): {pca.explained_variance_ratio_[:5]}")
        print(f"  Cumulative explained variance (10 components): {np.sum(pca.explained_variance_ratio_):.3f}")
        
        # K-means clustering
        optimal_k = 3  # Based on clinical expectation: high, medium, low responders
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state)
        clusters = kmeans.fit_predict(pca_result[:, :5])  # Use first 5 PCs
        
        print(f"\nK-means Clustering (k={optimal_k}):")
        unique, counts = np.unique(clusters, return_counts=True)
        for i, (cluster, count) in enumerate(zip(unique, counts)):
            print(f"  Cluster {cluster}: {count} participants ({count/len(clusters)*100:.1f}%)")
        
        # Visualize clustering results
        plt.figure(figsize=(15, 5))
        
        # PCA scree plot
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Scree Plot')
        plt.grid(True, alpha=0.3)
        
        # Clustering visualization
        plt.subplot(1, 3, 2)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i in range(optimal_k):
            mask = clusters == i
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=colors[i], alpha=0.6, label=f'Cluster {i}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Participant Clustering (PCA Space)')
        plt.legend()
        
        # Treatment response by cluster
        plt.subplot(1, 3, 3)
        treatment_response = np.mean(post_data - pre_data, axis=1)
        
        cluster_responses = []
        for i in range(optimal_k):
            cluster_response = treatment_response[clusters == i]
            cluster_responses.append(cluster_response)
            plt.boxplot([cluster_response], positions=[i], widths=0.6)
        
        plt.xlabel('Cluster')
        plt.ylabel('Treatment Response Score')
        plt.title('Treatment Response by Cluster')
        plt.xticks(range(optimal_k), [f'Cluster {i}' for i in range(optimal_k)])
        
        plt.tight_layout()
        plt.savefig('dimensionality_clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca_result, clusters, treatment_response

# Execute the comprehensive analysis
if __name__ == "__main__":
    print("Initializing Brain Imaging Analysis for Music Therapy Research...")
    
    # Initialize the analyzer
    analyzer = BrainImagingAnalyzer(random_state=42)
    
    # Step 1: Load/simulate data
    participants_df, pre_therapy_data, post_therapy_data, region_names = analyzer.simulate_brain_imaging_data(
        n_participants=120, n_regions=68
    )
    
    print(f"\nDataset Overview:")
    print(f"  Participants: {len(participants_df)}")
    print(f"  Brain regions analyzed: {len(region_names)}")
    print(f"  Age range: {participants_df['age'].min():.1f} - {participants_df['age'].max():.1f} years")
    print(f"  Mean baseline PANSS: {participants_df['baseline_panss'].mean():.1f} ± {participants_df['baseline_panss'].std():.1f}")
