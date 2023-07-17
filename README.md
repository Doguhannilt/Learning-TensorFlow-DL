
<strong>Acknowledgment: </strong> I would like to express my gratitude to Mr. Daniel Bourke (mrdbourke) for his exceptional TensorFlow tutorial. <br> His comprehensive and well-structured tutorial served as an invaluable resource in my learning journey. Mr. Bourke's expertise in TensorFlow and his ability to explain complex concepts in a clear and concise manner greatly facilitated my understanding of this powerful deep learning framework. <br> His dedication to providing high-quality educational content has been instrumental in enhancing my skills and knowledge. 
<p><strong>GitHub Repository: https://github.com/mrdbourke/tensorflow-deep-learning/</strong>


<br>
<br>

<h1>What is Deep Learning?</h1>

<lu>
  <li>A type of machine learning based on artificial neural networks in which multiple layers of processing are used to extract progressively higher level features from data.</li>
</lu>

<h3>Algorithms of Deep Larning</h3>

<lu>
	<li><strong>ANN: </strong> ANN stands for Artificial Neural Network. It is a computational model inspired by the structure and functioning of the human brain's neural networks. An ANN consists of interconnected artificial neurons or nodes that work together to process and transmit information. The basic building block of an artificial neural network is an artificial neuron, also known as a perceptron. Each neuron receives input signals, applies a mathematical transformation to these inputs, and produces an output signal. The output signal of one neuron can serve as an input to other neurons, forming a network of interconnected neurons.</li>
	<li><strong>RNN: </strong> RNN stands for Recurrent Neural Network. It is a type of artificial neural network designed to process sequential data by utilizing feedback connections, allowing the network to retain information from previous steps or time points. Unlike feedforward neural networks, which process data in a single forward direction, RNNs have loops within their structure, enabling them to maintain a memory of past inputs. This memory aspect makes RNNs well-suited for tasks involving sequential or time-dependent data, such as natural language processing, speech recognition, handwriting recognition, and time series analysis.</li>
	<li><strong>CNN: </strong> CNN stands for Convolutional Neural Network. It is a type of artificial neural network architecture specifically designed for processing and analyzing structured grid-like data, such as images and videos. CNNs are particularly effective in computer vision tasks due to their ability to automatically learn and extract meaningful features from images. They are composed of multiple layers, including convolutional layers, pooling layers, and fully connected layers.</li>
</lu>


<br>
<h3>Pons & Cons of ANN RNN CNN</h3>

<br>
<lu>
	<li><strong>Artificial Neural Networks (ANNs):</strong></li>
</lu>

<br>

<strong>Pons: </strong> <br>

<ol>
	<li>Versatility: ANNs can be applied to a wide range of tasks, including classification, regression, and pattern recognition.</li>
	<li>Non-linearity: ANNs can model complex relationships between input and output variables, allowing for non-linear mappings.</li>
	<li>Generalization: ANNs have the ability to generalize from training data and make predictions on unseen data.</li>
	<li>Parallel Processing: ANNs can perform computations in parallel, enabling efficient processing of large datasets.</li>
	<li>Adaptability: ANNs can learn and adapt to changing data patterns by adjusting their internal weights.</li>
</ol>


<strong>Cons: </strong> <br>

<ol>
	<li>Black Box Nature: ANNs often lack interpretability, making it challenging to understand the reasoning behind their predictions.</li>
	<li>Overfitting: ANNs can be prone to overfitting, meaning they may perform well on training data but struggle with unseen data.</li>
	<li>Computational Intensity: Training ANNs can be computationally expensive, requiring significant processing power and time.</li>
	<li>Large Data Requirements: ANNs may require large amounts of labeled training data to achieve optimal performance.</li>
	<li>Sensitivity to Noise: ANNs can be sensitive to noisy or irrelevant features in the input data, which can affect their performance.</li>

</ol>



<br>
<lu>
	<li><strong>Convolutional Neural Networks (CNNs): </strong></li>
</lu>

<br>

<strong>Pons: </strong> <br>

<ol>
	<li>Hierarchical Feature Extraction: CNNs automatically learn hierarchical representations of visual features, enabling effective image analysis.</li>
	<li>Translation Invariance: CNNs are capable of recognizing patterns regardless of their position in an image, making them robust to translations.</li>
	<li>Parameter Sharing: CNNs utilize shared weights across different spatial locations, reducing the number of parameters and improving efficiency.</li>
	<li>Local Connectivity: CNNs capture local spatial dependencies, allowing them to efficiently process large images without requiring excessive computation.</li>
	<li>State-of-the-Art Performance: CNNs have demonstrated outstanding performance in various computer vision tasks, including image classification, object detection, and image segmentation.</li>
</ol>


<strong>Cons: </strong> <br>

<ol>
	<li>Limited Contextual Information: CNNs have a limited receptive field, which may lead to difficulties in capturing long-range dependencies or global context.</li>
	<li>Lack of Temporal Information: CNNs are not inherently designed for sequential data analysis and may not effectively model temporal dependencies.</li>
	<li>Difficulty with Variable-Sized Inputs: CNNs typically require fixed-size inputs, making it challenging to handle inputs with varying dimensions.</li>
	<li>Large Memory Requirements: Deep CNN architectures with numerous layers can demand significant memory resources for training and inference.</li>
	<li>Limited Interpretability: Similar to ANNs, CNNs often lack interpretability, making it difficult to understand their decision-making process.</li>

</ol>



<br>
<lu>
	<li><strong>Recurrent Neural Networks (RNNs): </strong></li>
</lu>

<br>

<strong>Pons: </strong> <br>

<ol>
	<li>Sequential Modeling: RNNs excel at modeling and predicting sequential data by maintaining a memory of previous inputs.</li>
	<li>Variable-Length Inputs: RNNs can handle inputs of varying lengths, making them suitable for tasks like natural language processing.</li>
	<li>Temporal Dependencies: RNNs capture temporal dependencies and can effectively model time series data or sequences.</li>
	<li>Feedback Connections: RNNs can propagate information backward through time, allowing for context preservation and long-term dependencies.</li>
	<li>Language Generation: RNNs are widely used for tasks such as language generation, machine translation, and speech recognition.</li>
</ol>


<strong>Cons: </strong> <br>

<ol>
	<li>Vanishing/Exploding Gradients: RNNs can suffer from vanishing or exploding gradients during training, affecting their ability to learn long-term dependencies.</li>
	<li>Computationally Demanding: RNNs can be computationally expensive, especially when dealing with long sequences or complex models.</li>
	<li>Lack of Parallelism: The sequential nature of RNNs limits their ability to process inputs in parallel, potentially reducing efficiency.</li>
	<li>Difficulty with Long-Term Memory: Traditional RNNs may struggle with capturing long-term dependencies due to the vanishing gradient problem.</li>
	<li>Limited Context Window: RNNs have a finite memory capacity and may struggle to capture dependencies that span a large number of time steps.</li>

</ol>

<br>
<br>

<h1>Common Terminologies in Deep Learning</h1>

<ol>
	<li>Deep Learning: A subfield of machine learning that focuses on training deep neural networks with multiple hidden layers to learn and extract high-level representations from complex data.</li>
	<li>Activation Function: A mathematical function applied to the output of a neuron to introduce non-linearity and determine the neuron's output value.</li>
	<li>Backpropagation: A training algorithm used to update the weights of a neural network by propagating the error from the output layer back to the input layer.</li>
	<li>Gradient Descent: An optimization algorithm used to minimize the loss or error of a neural network by iteratively adjusting the weights in the direction of the steepest descent of the loss function.</li>
	<li>Loss Function: A function that measures the difference between the predicted output of a neural network and the actual output, used as a guide for updating the network's weights during training.</li>
	<li>Overfitting: A situation where a neural network performs exceptionally well on the training data but fails to generalize well on unseen data due to excessive complexity or lack of regularization.</li>
	<li>Pooling: A downsampling operation used in CNNs to reduce the spatial dimensions of feature maps, preserving the most important information.</li>
	<li>Long Short-Term Memory (LSTM): A type of RNN architecture that addresses the vanishing gradient problem and can effectively capture long-term dependencies in sequential data.</li>
	<li>Generative Adversarial Network (GAN): A type of neural network architecture consisting of a generator and a discriminator, working together in a competitive manner to generate realistic synthetic data.</li>
	<li>Transfer Learning: A technique where a pre-trained neural network, trained on a large dataset, is used as a starting point for a new task or dataset, allowing for faster and more effective training.</li>
	<li>Dropout: A regularization technique commonly used in deep learning, where a random subset of neurons is temporarily "dropped out" during training, preventing co-adaptation and improving generalization.</li>
	<li>Batch Normalization: A technique used to normalize the activations of neurons in a neural network by applying a normalization operation to each mini-batch of data, improving the stability and convergence of the network.</li>

</ol>

<br>
<br>


![unnamed](https://github.com/Doguhannilt/Learning-TensorFlow-DL/assets/77373443/b40bc341-1f4f-40d5-891f-2f1ea98099a9)
