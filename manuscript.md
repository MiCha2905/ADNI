# Non-Invasive Prediction of Alzheimer's Disease Conversion using FastSurfer-Derived Longitudinal Atrophy Velocity

**Sonali**
*Independent Researcher*
k2905gupta@gmail.com

---

## Abstract

**Background:** The early prediction of Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) conversion is critical for clinical trials and early intervention. However, current predictive models rely heavily on invasive cerebrospinal fluid (CSF) assays, expensive Positron Emission Tomography (PET) scans, or static structural MRI volumes, which are often confounded by individual cognitive reserve.

**Methods:** We propose a purely non-invasive machine learning pipeline leveraging longitudinal structural MRI. Using FastSurfer, we calculated the annualized atrophy velocity of 96 distinct brain regions. To prevent overfitting on small medical cohorts, we integrated baseline regional volumes (cognitive reserve) with longitudinal velocity and clinical demographics. We applied strict Recursive Feature Elimination (RFE) to isolate the top 10 most critical neurodegenerative biomarkers and evaluated the model using Leave-One-Out Cross-Validation (LOOCV).

**Results:** Evaluated on the Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort with 10-year ground-truth diagnostic tracking, our model achieved a state-of-the-art (SOTA) accuracy of 94.7%. Furthermore, the pipeline demonstrated a precision of 96.4%, a recall of 96.4%, and a Receiver Operating Characteristic Area Under the Curve (ROC-AUC) of 0.961 in predicting MCI-to-AD conversion.

**Conclusion:** Tracking the temporal velocity of neurodegeneration, rather than static snapshots, provides a highly accurate and deployable framework for identifying high-risk MCI patients. This method achieves SOTA predictive performance while completely bypassing the need for invasive or radioactive biomarkers.

**Keywords:** Alzheimer's Disease, Mild Cognitive Impairment, Machine Learning, FastSurfer, Structural MRI, Longitudinal Velocity, Feature Selection

---

## 1. Introduction
Alzheimer's Disease (AD) is the most common neurodegenerative disorder globally, characterized by irreversible cognitive decline and severe cerebral atrophy. The transitional state between expected cognitive aging and full-blown dementia is defined as Mild Cognitive Impairment (MCI). Because pharmaceutical interventions are most effective before massive neuronal death occurs, predicting which MCI patients will rapidly convert to AD is currently one of the most critical challenges in clinical neuroimaging.

However, current State-of-the-Art (SOTA) predictive models face significant limitations. Highly accurate pipelines often rely heavily on expensive Positron Emission Tomography (PET) or invasive cerebrospinal fluid (CSF) assays, making them impractical for standard hospital screenings. Conversely, models utilizing non-invasive Structural MRI typically rely on static, single-timepoint volumetric snapshots (e.g., total hippocampal volume at baseline). These static models are deeply confounded by the biological phenomenon of "Cognitive Reserve"; a patient with naturally large ventricles may be misclassified as diseased, while a patient with a naturally large hippocampus experiencing rapid decay may be misclassified as healthy.

To address these limitations, we propose a paradigm shift from static volumetric snapshots to *dynamic longitudinal velocity*. By utilizing FastSurfer to extract detailed parcellations from baseline and follow-up MRI scans, we calculated the annualized atrophy velocity of 96 distinct brain regions. We hypothesize that the *rate of structural change* is a far more reliable biological marker for aggressive disease trajectory than absolute volume alone.

In this paper, we present a multi-modal machine learning pipeline that integrates baseline cognitive reserve, clinical demographics, and annualized atrophy velocity. Utilizing Recursive Feature Elimination (RFE) and rigorous Leave-One-Out Cross-Validation (LOOCV), we demonstrate that tracking temporal velocity allows for the highly accurate, non-invasive prediction of MCI-to-AD conversion, bypassing the need for invasive testing.

## 2. Related Work
The prediction of MCI-to-AD conversion has been extensively studied, with existing literature largely divided into three methodological paradigms:

**Traditional Volumetric Approaches:** Early machine learning models focused on absolute brain volumes extracted from structural MRI (e.g., hippocampal or ventricular volume). For instance, Moradi et al. [5] utilized baseline MRI and cognitive scores to achieve approximately 82% accuracy. However, static volumes fail to account for patient-specific brain sizes, often leading to false positives in patients with naturally smaller anatomy (a biological phenomenon tied to cognitive reserve).

**Deep Learning on Static Imagery:** Recent advancements have leveraged 3D Convolutional Neural Networks (3D-CNN) on raw, single-timepoint MRI scans. Basaia et al. [6] demonstrated the utility of deep learning for automated classification, yet these models typically plateau between 75% and 80% accuracy for predicting future MCI conversion. Furthermore, CNNs act as "black boxes," offering limited neurobiological interpretability, and by relying on single timepoints, they fundamentally ignore the temporal element of neurodegeneration.

**Multi-Modal Complex Pipelines:** To break the 85% accuracy barrier, researchers often incorporate highly invasive or radioactive modalities. Spasov et al. [7] utilized a complex ensemble of structural MRI, FDG-PET scans, and cerebrospinal fluid (CSF) assays to achieve an accuracy of ~86%. While clinically effective, the requirement for painful lumbar punctures and expensive PET imaging makes these pipelines highly impractical for standard, scalable hospital screenings.

**Our Contribution:** Our proposed pipeline completely bypasses the need for PET and CSF data. By transitioning from static MRI snapshots to dynamic longitudinal velocity, we achieve SOTA accuracy purely through non-invasive structural MRI and clinical demographics.

## 3. Proposed Methodology

### 3.1 Dataset and Cohort Selection
Data used in the preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database. The clinical demographics, including age and sex, were extracted from the baseline clinical dataset. To perform rigorous survival analysis and validate our predictive models, long-term diagnostic tracking data (spanning up to 10 years) was extracted from the ADNI Diagnostic Summary (`DXSUM`) repository. The primary cohort analyzed in this study consisted of patients diagnosed with Mild Cognitive Impairment (MCI) at baseline who possessed both a baseline structural MRI scan and a follow-up scan within a 1-to-2 year window. 

### 3.2 Image Processing and Feature Extraction
Raw T1-weighted structural MRI scans were processed using FastSurfer [10], a modern, highly-scalable, deep-learning alternative to FreeSurfer. FastSurfer was utilized to perform whole-brain segmentation and parcellation, automatically extracting the absolute volumes of 96 distinct cortical and subcortical regions. 

### 3.3 Longitudinal Velocity Calculation
To mitigate the confounding variable of individual brain size (cognitive reserve), we transitioned from static volumetric analysis to dynamic velocity tracking. For each patient and for each of the 96 brain regions, the annualized velocity of atrophy ($V$) was calculated using the absolute volumes ($Vol$) extracted at the baseline ($T1$) and follow-up ($T2$) scans, normalized by the exact time elapsed in years:

$$V_i = \frac{Vol_{T2, i} - Vol_{T1, i}}{\Delta Years}$$

where $i$ represents a specific segmented brain region.

### 3.4 Multi-Modal Feature Enrichment and Selection
To build a highly robust predictive model, we integrated multiple modalities into a single feature space. The enriched feature set included the annualized velocity of all 96 regions, the absolute baseline volumes of all 96 regions (to account for cognitive reserve capacity), and baseline clinical demographics (Age and Sex), totaling 194 initial features.

Given the high dimensionality of the feature space relative to the cohort size, we employed Recursive Feature Elimination (RFE) utilizing an L2-penalized Logistic Regression estimator. This strict feature selection process isolated the top 10 most powerful and stable neurodegenerative biomarkers, successfully mitigating the curse of dimensionality.

### 3.5 Predictive Modeling and Validation
A strict, L2-regularized Logistic Regression classifier was trained on the top 10 extracted biomarkers to predict the long-term clinical conversion from MCI to AD. To ensure rigorous, unbiased evaluation and to prevent the overfitting common in small-N medical machine learning studies, the model was evaluated exclusively using Leave-One-Out Cross-Validation (LOOCV). Performance was measured using Accuracy, Precision, Recall, F1-Score, and the Area Under the Receiver Operating Characteristic Curve (ROC-AUC).

## 4. Results

### 4.1 Longitudinal Velocity as a Discriminative Biomarker
To validate the hypothesis that the rate of structural change is more discriminative than baseline static volume, we analyzed the annualized velocity across the entire cohort. As shown in **Figure 1**, the velocity of neurodegeneration perfectly separated the extreme phenotypes (Cognitively Normal vs. Alzheimer's Disease), whereas baseline cognitive reserve showed significant overlap.

![Distribution of annualized atrophy velocity across clinical groups. The rate of change effectively separates healthy subjects from those with active Alzheimer's Disease.]<img width="1527" height="533" alt="Change_in_velocity" src="https://github.com/user-attachments/assets/67ac8eb5-96c2-4c97-8c6c-c978ae5ed0f0" />

*Figure 1: Distribution of annualized atrophy velocity across clinical groups.*

### 4.2 Unsupervised Risk Stratification and Feature Importance
A Random Forest classifier trained on the CN vs. AD phenotypes was utilized to perform Zero-Shot inference on the ambiguous MCI cohort. This unsupervised stratification successfully segmented the MCI patients into a "Low Risk" group and a "High Risk" group. The top predictive features driving this stratification, identified by Gini impurity, heavily favored bilateral temporal and parietal lobe velocities, aligning with established AD pathology (**Figure 2**).

![Feature Importance Plot. Bilateral temporal regions and ventricular expansion velocity dominated the decision trees.]<img width="797" height="540" alt="CriticalRegion" src="https://github.com/user-attachments/assets/0577dba9-c621-408e-bcf5-5a7228765c88" />

*Figure 2: Feature Importance Plot.*

### 4.3 Survival Analysis and Clinical Validation
To rigorously validate the unsupervised risk stratification, Kaplan-Meier survival analysis was performed using 10-year ground-truth clinical data. As shown in **Figure 3**, the survival curves demonstrated a profound divergence. Notably, 100% of the patients flagged as "High Risk" by the AI pipeline based solely on their 1-year brain velocity eventually converted to full clinical dementia (Positive Predictive Value = 1.0).

![Kaplan-Meier Survival Curves. The High-Risk MCI group demonstrates a 100% conversion rate to AD over the tracking period, validating the velocity biomarkers.]<img width="747" height="553" alt="Kapman-Meier" src="https://github.com/user-attachments/assets/f1b4a7d8-78ce-41e8-be41-0b08a4deaa69" />

*Figure 3: Kaplan-Meier Survival Curves.*

### 4.4 State-of-the-Art Multi-Modal Prediction
The final supervised model was trained to predict MCI-to-AD conversion directly, utilizing the top 10 biomarkers isolated via RFE (integrating baseline volume, demographic data, and velocity). Evaluated under strict Leave-One-Out Cross-Validation (LOOCV), the pipeline achieved an overall accuracy of 94.7% (**Figure 4**). The precision and recall for predicting AD conversion were both 96.4%, yielding an F1-Score of 96.4% and an ROC-AUC of 0.961.

![Confusion Matrix of the SOTA Multi-Modal Classifier evaluated using LOOCV.]<img width="542" height="446" alt="SOOTA_confusion_matrix" src="https://github.com/user-attachments/assets/2d250d6b-edb2-4440-9e73-9fbbc15aad70" />

*Figure 4: Confusion Matrix of the SOTA Multi-Modal Classifier evaluated using LOOCV.*

**Table 1:** Comparison of the proposed model against State-of-the-Art (SOTA) literature for MCI-to-AD conversion prediction.

| Research Approach | Modalities Used | Methodology | Accuracy |
| :--- | :--- | :--- | :---: |
| Traditional Volumetric (Moradi et al.) [5] | Baseline MRI + Cognitive | Cross-sectional SVM | ~82% |
| Deep Learning (Basaia et al.) [6] | Raw Baseline MRI | 3D-CNN | 75--80% |
| Multi-Modal (Spasov et al.) [7] | MRI + PET + CSF + Clinical | Complex Ensembles | ~86% |
| **Proposed Pipeline** | **MRI (Velocity) + Demographics** | **RFE + LOOCV** | **94.7%** |

## 5. Discussion
The accurate and early prediction of MCI-to-AD conversion remains one of the most pressing challenges in neuroinformatics and clinical neurology. The results of this study strongly support the hypothesis that tracking the dynamic longitudinal velocity of brain atrophy is fundamentally superior to relying on static volumetric snapshots. By normalizing structural volume changes over time, our proposed pipeline effectively controls for the critical confounding variable of cognitive reserve. This ensures that natural biological variations in human brain size---which often cause false positives in traditional cross-sectional machine learning models---are eliminated from the predictive decision boundary.

A key strength of our approach lies in its neurobiological interpretability. Unlike many deep learning architectures that function as "black boxes" [6], our pipeline utilized Recursive Feature Elimination (RFE) to isolate the most critical biomarkers driving the prediction. The top predictive features identified by the model heavily favored the annualized velocity of the bilateral temporal lobes, the parietal lobes, and the expansion of the lateral ventricles. This data-driven feature selection perfectly aligns with established neuropathological timelines, such as Braak staging [3], which dictates that Alzheimer's-related neurodegeneration typically originates in the medial temporal structures before propagating to the parietal and frontal neocortex [4]. 

Furthermore, the integration of unsupervised Zero-Shot inference with Kaplan-Meier survival analysis provided rigorous validation of our risk stratification. The observation that 100% of the MCI patients flagged as "High Risk" by the AI pipeline eventually converted to full clinical dementia over the 10-year tracking period demonstrates the profound clinical utility of utilizing 1-year velocity metrics to forecast long-term outcomes. 

Crucially, our approach successfully circumvents the need for invasive, painful, and expensive testing. While current SOTA models frequently depend on FDG-PET imaging or cerebrospinal fluid (CSF) tau and amyloid-$\beta$ assays to breach the 85% accuracy threshold [7], our multi-modal pipeline achieved 94.7% accuracy utilizing only standard, non-invasive structural MRI and clinical demographics. This positions our model not merely as an academic exercise, but as a highly deployable, cost-effective screening tool suited for standard hospital environments, where widespread PET scanning and lumbar punctures are logistically and financially unfeasible.

Despite the robust predictive performance, this study has limitations. The reliance on structural MRI alone means the model captures macroscopic neurodegeneration rather than the underlying molecular pathology (amyloid plaques and tau tangles) [2]. Future research should investigate the fusion of longitudinal MRI velocity with longitudinal cognitive assessments (e.g., ADAS-Cog scores) to capture both the structural and behavioral dimensions of the disease trajectory simultaneously.

## 6. Conclusion
In this study, we developed, evaluated, and rigorously validated a multi-modal machine learning pipeline designed for the early prediction of Alzheimer's Disease conversion in Mild Cognitive Impairment cohorts. By leveraging FastSurfer to extract detailed parcellations from longitudinal MRI data, we calculated the annualized atrophy velocity across the brain, effectively neutralizing the confounding effects of cognitive reserve. 

Through the combination of this longitudinal velocity with baseline absolute volumes and clinical demographics, and the application of strict Recursive Feature Elimination, we isolated the top 10 most powerful structural biomarkers. Evaluated via Leave-One-Out Cross-Validation on the ADNI cohort---validated against 10-year ground-truth clinical tracking data---the model achieved a state-of-the-art accuracy of 94.7%, a precision of 96.4%, and an ROC-AUC of 0.961. This pipeline demonstrates that tracking the temporal dynamics of structural neurodegeneration provides a highly accurate, non-invasive, and clinically viable framework for identifying high-risk MCI patients, completely bypassing the need for invasive or radioactive diagnostic procedures.

## Data and Code Availability
The MRI datasets, clinical demographics, and 10-year longitudinal clinical tracking data analyzed in this study are publicly available via the Alzheimer's Disease Neuroimaging Initiative (ADNI) repository (adni.loni.usc.edu). The complete codebase, machine learning pipeline, and FastSurfer feature processing scripts used to generate the results in this manuscript are available upon reasonable request.

## Acknowledgments
Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012). ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from many private pharmaceutical entities.

## References

[1] Petersen, R. C. (2004). Mild cognitive impairment as a diagnostic entity. *Journal of Internal Medicine*, 256(3), 183-194.

[2] Jack Jr, C. R., Knopman, D. S., Jagust, W. J., Shaw, L. M., Aisen, P. S., Weiner, M. W., ... & Trojanowski, J. Q. (2010). Hypothetical model of dynamic biomarkers of the Alzheimer's pathological cascade. *The Lancet Neurology*, 9(1), 119-128.

[3] Braak, H., & Braak, E. (1991). Neuropathological stageing of Alzheimer-related changes. *Acta Neuropathologica*, 82(4), 239-259.

[4] Dickerson, B. C., Bakkour, A., Salat, D. H., Feczko, E., Pacheco, J., Greve, D. N., ... & Buckner, R. L. (2009). The cortical signature of Alzheimer's disease: regionally specific cortical thinning relates to symptom severity in very mild to mild AD dementia and is detectable in asymptomatic amyloid-positive individuals. *Cerebral Cortex*, 19(3), 497-510.

[5] Moradi, E., Pepe, A., Gaser, C., Huttunen, H., Tohka, J., & ADNI. (2015). Machine learning framework for early MRI-based Alzheimer's conversion prediction in MCI subjects. *Neuroimage*, 104, 398-412.

[6] Basaia, S., Agosta, F., Wagner, L., Canu, E., Magnani, G., Santangelo, R., ... & Filippi, M. (2019). Automated classification of Alzheimer's disease and mild cognitive impairment using a single MRI and deep neural networks. *NeuroImage: Clinical*, 21, 101645.

[7] Spasov, S., Passamonti, L., Duggento, A., Lio, P., Toschi, N., & ADNI. (2019). A parameter-efficient deep learning approach to predict conversion from mild cognitive impairment to Alzheimer's disease. *Neuroimage*, 189, 276-287.

[8] Eskildsen, S. F., Coupé, P., García-Lorenzo, D., Fonov, V., Pruessner, J. C., Collins, D. L., & ADNI. (2013). Prediction of Alzheimer's disease in subjects with mild cognitive impairment from the ADNI cohort using patterns of cortical thinning. *Neuroimage*, 65, 511-521.

[9] Cuingnet, R., Gerardin, E., Tessieras, J., Auzias, G., Lehéricy, S., Habert, M. O., ... & ADNI. (2011). Automatic classification of patients with Alzheimer's disease from structural MRI: a comparison of ten methods using the ADNI database. *Neuroimage*, 56(2), 766-781.

[10] Henschel, L., Conjeti, S., Estrada, S., Diers, K., Fischl, B., & Reuter, M. (2020). FastSurfer-A fast and accurate deep learning based neuroimaging pipeline. *NeuroImage*, 219, 117012.
