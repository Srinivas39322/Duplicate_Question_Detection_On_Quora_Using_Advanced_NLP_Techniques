### Duplicate Question Detection on Quora Using Advanced NLP Techniques 🧠💬

---

#### **Introduction**
This project applies advanced natural language processing (NLP) techniques to detect duplicate questions on Quora. By leveraging transformer-based embeddings, LSTM-based architectures, and ensemble models, it aims to enhance the accuracy of identifying semantically similar questions. The methodology addresses challenges in duplicate content detection, improving user experience and scalability for real-world applications.

---

#### **Objective**
To develop robust NLP models for detecting duplicate questions on Quora, enhancing content quality and user engagement by minimizing redundant questions.

---

#### **SMART Questions**
1. How effectively can transformer-based embeddings capture semantic similarities between questions?
2. Does leveraging LSTM-based architectures improve the model's ability to detect duplicates?
3. What role do features like word overlap and n-gram similarity play in improving model accuracy?
4. Can ensemble models enhance performance by combining multiple classifiers?
5. How scalable is the solution for real-world applications with millions of queries?

---

#### **Dataset**
- **Source**: [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/overview)
- **Key Features**:
  - Textual features: Word overlap, length differences
  - Semantic features: Embedding similarity using MiniLM
  - Distance metrics: Cosine similarity
- **Structure**: Training, validation, and test datasets

---

#### **Key Findings/Conclusion**
1. **Model Performance**:
   - Baseline Random Forest achieved 79% accuracy.
   - Advanced models like XGBoost and LightGBM improved accuracy to 81.3%.
   - LSTM-based Siamese Neural Network achieved 88.2% accuracy with strong generalization.

2. **Feature Insights**:
   - Semantic features using MiniLM embeddings and cosine similarity significantly improved duplicate detection.
   - N-gram overlap captured phrasal similarities, complementing semantic features.

3. **Impact**:
   - Improved user experience by reducing redundant content on Quora.
   - Broader applicability in forums, customer support, and e-commerce for deduplication tasks.

4. **Limitations**:
   - Difficulty distinguishing similar but non-duplicate questions.
   - High computational resource requirements for transformer-based models.

5. **Future Work**:
   - Explore unsupervised methods for unseen duplicate detection.
   - Expand datasets to test model scalability across platforms.
   - Implement real-time duplicate detection for large-scale applications.

---
