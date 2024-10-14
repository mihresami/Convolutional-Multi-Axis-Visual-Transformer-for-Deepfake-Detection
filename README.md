Convolutional Multi-Axis Visual Transformer for Deepfake Detection
Abstract The recent advancement of DeepFake technology has raised several concerns due to its potential to spread
misinformation. Although existing methods relying on Deep Neural Networks (DNNs) achieve promising results, they still
struggle to identify deepfakes with subtle visual defects and detect compressed formats. Additionally, many detection methods are computationally demanding, limiting their use in resource-constrained environments. We propose a novel lightweight
Convolutional Multi-Axis Vision Transformer (CMViT) for deepfake detection to address these challenges. CMViT leverages
the strengths of CNNs and Visual Transformer: a CNN extracts overall image features, while a window-based attention
mechanism in the Multi-Axis Visual Transformer analyzes patch-level details. This combined approach allows CMViT to
effectively capture high-level and fine-grained information crucial for detecting deepfakes. Furthermore, we introduce a
frequency filtering method that extracts spatial features in the frequency domain, specifically targeting compressed deepfakes. Extensive experiments demonstrate that CMViT achieves superior performance while maintaining low computational
cost. Notably, on the FF++ c40 compressed dataset, CMViT achieves an impressive 97.44% Area Under the Curve score,
significantly outperforming existing methods.
Keywords Deepfake, Convolutional neural network, global features, visual transformers, attention mechanism
