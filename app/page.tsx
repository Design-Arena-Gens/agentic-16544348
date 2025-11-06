'use client';

import { useState } from 'react';

const ebookPages = [
  {
    title: "Table of Contents",
    content: (
      <div>
        <h2>Understanding Artificial Intelligence and Machine Learning</h2>
        <p style={{ fontSize: '1.2rem', marginBottom: '2rem', color: '#4a5568' }}>
          A Comprehensive Guide to the Future of Technology
        </p>

        <div className="table-of-contents">
          {[
            "Introduction to AI and ML",
            "The History of Artificial Intelligence",
            "Core Concepts and Terminology",
            "Types of Machine Learning",
            "Neural Networks and Deep Learning",
            "Natural Language Processing",
            "Computer Vision Applications",
            "AI in Business and Industry",
            "Ethical Considerations",
            "AI Development Tools",
            "Training AI Models",
            "Real-World Applications",
            "Challenges and Limitations",
            "The Future of AI",
            "Getting Started with AI"
          ].map((title, idx) => (
            <div key={idx} className="toc-item">
              <div className="toc-number">Page {idx + 2}</div>
              <div className="toc-title">{title}</div>
            </div>
          ))}
        </div>
      </div>
    )
  },
  {
    title: "Introduction to AI and ML",
    content: (
      <div>
        <h2>Introduction to Artificial Intelligence and Machine Learning</h2>

        <p>
          Artificial Intelligence (AI) and Machine Learning (ML) represent two of the most transformative
          technologies of the 21st century. These fields are revolutionizing how we interact with technology,
          make decisions, and solve complex problems across virtually every industry.
        </p>

        <div className="highlight-box">
          <strong>What is Artificial Intelligence?</strong>
          <p style={{ marginTop: '1rem', marginBottom: 0 }}>
            Artificial Intelligence refers to the simulation of human intelligence in machines that are
            programmed to think, learn, and problem-solve. AI systems can perceive their environment,
            process information, and take actions to achieve specific goals.
          </p>
        </div>

        <h3>The Distinction Between AI and ML</h3>
        <p>
          While often used interchangeably, AI and ML are distinct concepts:
        </p>

        <ul>
          <li>
            <strong>Artificial Intelligence</strong> is the broader concept of machines being able to carry
            out tasks in a way that we would consider "smart" or "intelligent"
          </li>
          <li>
            <strong>Machine Learning</strong> is a specific subset of AI that enables machines to learn from
            data without being explicitly programmed
          </li>
        </ul>

        <p>
          Machine Learning is the engine that powers most modern AI applications. It allows systems to
          automatically improve their performance through experience, making them increasingly accurate
          and effective over time.
        </p>

        <div className="key-takeaway">
          <strong>Key Takeaway:</strong>
          <p style={{ marginTop: '1rem', marginBottom: 0 }}>
            AI is the destination, while ML is one of the most important vehicles getting us there.
            Understanding this relationship is fundamental to grasping how modern intelligent systems work.
          </p>
        </div>
      </div>
    )
  },
  {
    title: "The History of Artificial Intelligence",
    content: (
      <div>
        <h2>The History of Artificial Intelligence</h2>

        <p>
          The journey of artificial intelligence spans over seven decades, marked by periods of intense
          optimism, followed by challenging "AI winters," and culminating in today's unprecedented breakthroughs.
        </p>

        <h3>The Birth of AI (1950s)</h3>
        <p>
          The field of AI was officially born in 1956 at the Dartmouth Conference, where John McCarthy
          coined the term "artificial intelligence." However, the conceptual foundations were laid earlier:
        </p>

        <ul>
          <li>
            <strong>1950:</strong> Alan Turing published "Computing Machinery and Intelligence,"
            introducing the famous Turing Test
          </li>
          <li>
            <strong>1956:</strong> The Dartmouth Conference brought together pioneers who believed
            machines could simulate human intelligence
          </li>
        </ul>

        <h3>Early Enthusiasm and the First AI Winter (1960s-1970s)</h3>
        <p>
          The 1960s saw significant progress with early AI programs that could solve algebra problems
          and prove mathematical theorems. However, limitations in computing power and overpromising
          led to reduced funding and the first "AI winter" in the 1970s.
        </p>

        <h3>Expert Systems Era (1980s)</h3>
        <p>
          The 1980s brought renewed interest through expert systems—programs that mimicked human
          decision-making in specific domains. Companies invested heavily, though another AI winter
          followed when these systems proved brittle and expensive to maintain.
        </p>

        <h3>The Machine Learning Revolution (1990s-2000s)</h3>
        <p>
          The focus shifted from rule-based systems to data-driven machine learning. Key milestones included:
        </p>

        <ul>
          <li>Development of support vector machines and ensemble methods</li>
          <li>IBM's Deep Blue defeating world chess champion Garry Kasparov in 1997</li>
          <li>The rise of the internet providing vast amounts of training data</li>
        </ul>

        <h3>Deep Learning Breakthrough (2010s-Present)</h3>
        <p>
          The convergence of big data, powerful GPUs, and refined neural network architectures
          triggered an AI renaissance. Deep learning achieved superhuman performance in image
          recognition, natural language processing, and game playing, bringing AI into mainstream
          applications we use daily.
        </p>
      </div>
    )
  },
  {
    title: "Core Concepts and Terminology",
    content: (
      <div>
        <h2>Core Concepts and Terminology</h2>

        <p>
          Understanding AI and ML requires familiarity with several fundamental concepts and terms
          that form the foundation of the field.
        </p>

        <h3>Essential AI/ML Vocabulary</h3>

        <div className="highlight-box">
          <strong>Algorithm:</strong> A set of rules or instructions that a computer follows to solve
          a problem or complete a task. In ML, algorithms learn patterns from data.
        </div>

        <div className="highlight-box">
          <strong>Dataset:</strong> A collection of data used to train, validate, or test machine
          learning models. Quality and quantity of data are crucial for model performance.
        </div>

        <div className="highlight-box">
          <strong>Model:</strong> The mathematical representation of a real-world process created by
          an ML algorithm after training on data. The model makes predictions on new, unseen data.
        </div>

        <div className="highlight-box">
          <strong>Training:</strong> The process of feeding data to an algorithm so it can learn
          patterns and relationships. The model adjusts its internal parameters to minimize errors.
        </div>

        <div className="highlight-box">
          <strong>Features:</strong> Individual measurable properties or characteristics of the data
          being observed. Selecting relevant features is critical for model accuracy.
        </div>

        <div className="highlight-box">
          <strong>Labels:</strong> The output or target variable that a supervised learning model is
          trying to predict. For example, "spam" or "not spam" for email classification.
        </div>

        <h3>Performance Metrics</h3>
        <p>
          Machine learning models are evaluated using various metrics:
        </p>

        <ul>
          <li><strong>Accuracy:</strong> The percentage of correct predictions</li>
          <li><strong>Precision:</strong> The accuracy of positive predictions</li>
          <li><strong>Recall:</strong> The ability to find all positive instances</li>
          <li><strong>F1 Score:</strong> The harmonic mean of precision and recall</li>
        </ul>

        <p>
          Choosing the right metric depends on the specific problem and the costs associated with
          different types of errors.
        </p>
      </div>
    )
  },
  {
    title: "Types of Machine Learning",
    content: (
      <div>
        <h2>Types of Machine Learning</h2>

        <p>
          Machine learning approaches can be categorized into three main types, each suited for
          different kinds of problems and data scenarios.
        </p>

        <h3>1. Supervised Learning</h3>
        <div className="highlight-box">
          <p>
            <strong>Definition:</strong> Learning from labeled data where the correct answer is provided
            during training. The algorithm learns to map inputs to outputs based on example pairs.
          </p>
        </div>

        <p><strong>Common Applications:</strong></p>
        <ul>
          <li>Email spam detection</li>
          <li>Image classification</li>
          <li>Price prediction</li>
          <li>Medical diagnosis</li>
          <li>Credit scoring</li>
        </ul>

        <p><strong>Popular Algorithms:</strong> Linear regression, logistic regression, decision trees,
        random forests, support vector machines, and neural networks.</p>

        <h3>2. Unsupervised Learning</h3>
        <div className="highlight-box">
          <p>
            <strong>Definition:</strong> Learning from unlabeled data where the algorithm must find
            patterns and structure on its own without predefined categories or labels.
          </p>
        </div>

        <p><strong>Common Applications:</strong></p>
        <ul>
          <li>Customer segmentation</li>
          <li>Anomaly detection</li>
          <li>Recommendation systems</li>
          <li>Data compression</li>
          <li>Topic modeling in text</li>
        </ul>

        <p><strong>Popular Algorithms:</strong> K-means clustering, hierarchical clustering, principal
        component analysis (PCA), and autoencoders.</p>

        <h3>3. Reinforcement Learning</h3>
        <div className="highlight-box">
          <p>
            <strong>Definition:</strong> Learning through interaction with an environment, receiving
            rewards or penalties for actions taken. The algorithm learns to maximize cumulative rewards.
          </p>
        </div>

        <p><strong>Common Applications:</strong></p>
        <ul>
          <li>Game playing (Chess, Go, video games)</li>
          <li>Robotics and autonomous vehicles</li>
          <li>Resource management</li>
          <li>Personalized recommendations</li>
          <li>Trading strategies</li>
        </ul>

        <p><strong>Key Concepts:</strong> Agent, environment, state, action, reward, and policy.</p>
      </div>
    )
  },
  {
    title: "Neural Networks and Deep Learning",
    content: (
      <div>
        <h2>Neural Networks and Deep Learning</h2>

        <p>
          Neural networks, inspired by the biological neural networks in animal brains, form the
          foundation of deep learning and have revolutionized AI capabilities in recent years.
        </p>

        <h3>Understanding Neural Networks</h3>
        <p>
          An artificial neural network consists of interconnected nodes (neurons) organized in layers:
        </p>

        <ul>
          <li>
            <strong>Input Layer:</strong> Receives the raw data or features
          </li>
          <li>
            <strong>Hidden Layers:</strong> Process the information through weighted connections,
            learning increasingly abstract representations
          </li>
          <li>
            <strong>Output Layer:</strong> Produces the final prediction or classification
          </li>
        </ul>

        <div className="highlight-box">
          <strong>How Neural Networks Learn</strong>
          <p style={{ marginTop: '1rem', marginBottom: 0 }}>
            Neural networks learn through a process called backpropagation. During training, the network
            makes predictions, calculates the error, and adjusts its weights backward through the layers
            to minimize that error. This iterative process continues until the network achieves
            satisfactory performance.
          </p>
        </div>

        <h3>Deep Learning: Networks That Go Deeper</h3>
        <p>
          Deep learning refers to neural networks with many hidden layers (hence "deep"). The depth
          allows these networks to learn hierarchical representations of data:
        </p>

        <ul>
          <li>Early layers might detect simple features (edges, colors)</li>
          <li>Middle layers combine these into more complex patterns (shapes, textures)</li>
          <li>Deeper layers recognize high-level concepts (objects, faces)</li>
        </ul>

        <h3>Types of Deep Neural Networks</h3>

        <p><strong>Convolutional Neural Networks (CNNs):</strong></p>
        <p>
          Specialized for processing grid-like data such as images. CNNs use convolutional layers
          that automatically learn spatial hierarchies of features, making them incredibly effective
          for computer vision tasks.
        </p>

        <p><strong>Recurrent Neural Networks (RNNs):</strong></p>
        <p>
          Designed for sequential data like text or time series. RNNs maintain an internal state
          (memory) that allows them to process sequences of varying length and capture temporal dependencies.
        </p>

        <p><strong>Transformers:</strong></p>
        <p>
          The architecture behind modern language models like GPT and BERT. Transformers use
          attention mechanisms to process entire sequences simultaneously, achieving unprecedented
          performance in natural language tasks.
        </p>
      </div>
    )
  },
  {
    title: "Natural Language Processing",
    content: (
      <div>
        <h2>Natural Language Processing</h2>

        <p>
          Natural Language Processing (NLP) enables computers to understand, interpret, and generate
          human language. It sits at the intersection of AI, computational linguistics, and computer science.
        </p>

        <h3>Core NLP Tasks</h3>

        <p><strong>Text Classification:</strong></p>
        <p>
          Assigning categories to text documents. Applications include spam detection, sentiment
          analysis, topic categorization, and intent recognition in chatbots.
        </p>

        <p><strong>Named Entity Recognition (NER):</strong></p>
        <p>
          Identifying and classifying named entities (people, organizations, locations, dates) in text.
          Essential for information extraction and knowledge graph construction.
        </p>

        <p><strong>Machine Translation:</strong></p>
        <p>
          Automatically translating text from one language to another. Modern neural machine translation
          systems have achieved near-human quality for many language pairs.
        </p>

        <p><strong>Question Answering:</strong></p>
        <p>
          Systems that can understand questions posed in natural language and provide accurate answers,
          either by extracting information from documents or generating responses.
        </p>

        <h3>The Transformer Revolution</h3>
        <div className="highlight-box">
          <p>
            The introduction of the Transformer architecture in 2017 revolutionized NLP. These models
            use attention mechanisms to weigh the importance of different words in context, enabling:
          </p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li>Better understanding of long-range dependencies in text</li>
            <li>Parallel processing for faster training</li>
            <li>Transfer learning through pre-trained models</li>
            <li>State-of-the-art performance across numerous NLP tasks</li>
          </ul>
        </div>

        <h3>Large Language Models (LLMs)</h3>
        <p>
          Modern LLMs like GPT-4, Claude, and PaLM are trained on massive amounts of text data and
          contain billions of parameters. They demonstrate emergent capabilities including:
        </p>

        <ul>
          <li>Coherent long-form text generation</li>
          <li>Few-shot learning from examples</li>
          <li>Complex reasoning and problem-solving</li>
          <li>Code generation and analysis</li>
          <li>Multilingual understanding</li>
        </ul>

        <p>
          These models are transforming how we interact with computers, making natural language the
          interface for accessing AI capabilities.
        </p>
      </div>
    )
  },
  {
    title: "Computer Vision Applications",
    content: (
      <div>
        <h2>Computer Vision Applications</h2>

        <p>
          Computer vision enables machines to derive meaningful information from digital images, videos,
          and other visual inputs. Deep learning has dramatically advanced the field, enabling human-level
          or superhuman performance on many visual tasks.
        </p>

        <h3>Image Classification and Recognition</h3>
        <p>
          The foundation of computer vision, where models identify what objects or scenes are present
          in an image. Applications include:
        </p>

        <ul>
          <li>Photo organization and search</li>
          <li>Content moderation on social media</li>
          <li>Product recognition in retail</li>
          <li>Plant and animal species identification</li>
          <li>Medical image analysis (X-rays, MRIs, CT scans)</li>
        </ul>

        <h3>Object Detection and Segmentation</h3>
        <div className="highlight-box">
          <p>
            <strong>Object Detection:</strong> Locating and classifying multiple objects within an image,
            drawing bounding boxes around each detected object.
          </p>
          <p style={{ marginTop: '1rem', marginBottom: 0 }}>
            <strong>Semantic Segmentation:</strong> Classifying each pixel in an image, creating
            detailed masks that precisely outline objects and regions.
          </p>
        </div>

        <p><strong>Real-world applications:</strong></p>
        <ul>
          <li>Autonomous vehicles detecting pedestrians, vehicles, and road signs</li>
          <li>Surveillance systems identifying suspicious activities</li>
          <li>Retail analytics tracking customer behavior</li>
          <li>Medical imaging for tumor detection and organ segmentation</li>
        </ul>

        <h3>Facial Recognition and Analysis</h3>
        <p>
          AI systems can now identify individuals from facial features with high accuracy, as well as
          analyze attributes like age, emotion, and attention. While powerful, these technologies raise
          important privacy and ethical considerations.
        </p>

        <h3>Optical Character Recognition (OCR)</h3>
        <p>
          Converting images of text into machine-readable text. Modern OCR systems handle:
        </p>

        <ul>
          <li>Multiple languages and scripts</li>
          <li>Handwritten text</li>
          <li>Text in complex layouts</li>
          <li>Text in natural scenes (signs, documents)</li>
        </ul>

        <h3>Video Understanding</h3>
        <p>
          Extending image analysis to video streams, enabling:
        </p>

        <ul>
          <li>Action recognition and activity detection</li>
          <li>Video summarization and highlight generation</li>
          <li>Anomaly detection in surveillance footage</li>
          <li>Sports analytics and performance tracking</li>
        </ul>
      </div>
    )
  },
  {
    title: "AI in Business and Industry",
    content: (
      <div>
        <h2>AI in Business and Industry</h2>

        <p>
          Artificial intelligence is transforming businesses across every sector, driving efficiency,
          enabling new capabilities, and creating competitive advantages for early adopters.
        </p>

        <h3>Healthcare and Medicine</h3>
        <p>
          AI is revolutionizing healthcare through:
        </p>

        <ul>
          <li><strong>Diagnostic Assistance:</strong> AI systems analyze medical images to detect diseases
          like cancer, often matching or exceeding specialist accuracy</li>
          <li><strong>Drug Discovery:</strong> ML accelerates the identification of promising drug
          candidates, reducing development time and costs</li>
          <li><strong>Personalized Treatment:</strong> Algorithms analyze patient data to recommend
          tailored treatment plans</li>
          <li><strong>Predictive Analytics:</strong> Forecasting disease progression and patient outcomes</li>
          <li><strong>Administrative Automation:</strong> Streamlining scheduling, billing, and
          documentation</li>
        </ul>

        <h3>Finance and Banking</h3>
        <div className="highlight-box">
          <p><strong>Key Applications:</strong></p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li>Fraud detection in real-time transactions</li>
            <li>Credit risk assessment and loan approval</li>
            <li>Algorithmic trading and portfolio optimization</li>
            <li>Customer service chatbots</li>
            <li>Anti-money laundering compliance</li>
          </ul>
        </div>

        <h3>Retail and E-commerce</h3>
        <p>
          AI enhances the shopping experience and optimizes operations:
        </p>

        <ul>
          <li>Personalized product recommendations</li>
          <li>Dynamic pricing optimization</li>
          <li>Inventory management and demand forecasting</li>
          <li>Visual search and virtual try-on</li>
          <li>Customer sentiment analysis</li>
        </ul>

        <h3>Manufacturing and Supply Chain</h3>
        <p>
          Industry 4.0 leverages AI for:
        </p>

        <ul>
          <li>Predictive maintenance reducing equipment downtime</li>
          <li>Quality control through computer vision inspection</li>
          <li>Supply chain optimization and logistics</li>
          <li>Demand forecasting and production planning</li>
          <li>Robotic automation of complex tasks</li>
        </ul>

        <h3>Marketing and Advertising</h3>
        <p>
          AI enables hyper-personalized marketing at scale:
        </p>

        <ul>
          <li>Customer segmentation and targeting</li>
          <li>Content generation and optimization</li>
          <li>Predictive lead scoring</li>
          <li>Chatbots for customer engagement</li>
          <li>Ad campaign optimization</li>
        </ul>
      </div>
    )
  },
  {
    title: "Ethical Considerations",
    content: (
      <div>
        <h2>Ethical Considerations in AI</h2>

        <p>
          As AI systems become more powerful and pervasive, addressing ethical challenges is crucial
          for ensuring these technologies benefit humanity while minimizing potential harms.
        </p>

        <h3>Bias and Fairness</h3>
        <div className="highlight-box">
          <p>
            <strong>The Challenge:</strong> AI systems can perpetuate or amplify existing biases present
            in training data, leading to unfair outcomes for certain groups.
          </p>
        </div>

        <p><strong>Examples of AI bias:</strong></p>
        <ul>
          <li>Facial recognition systems performing worse on people of color</li>
          <li>Hiring algorithms discriminating against women or minorities</li>
          <li>Credit scoring systems showing bias based on zip codes</li>
          <li>Criminal justice risk assessments producing racially disparate predictions</li>
        </ul>

        <p>
          <strong>Solutions:</strong> Diverse datasets, fairness-aware algorithms, regular audits,
          and diverse development teams can help mitigate bias.
        </p>

        <h3>Privacy and Surveillance</h3>
        <p>
          AI enables unprecedented data collection and analysis capabilities, raising concerns about:
        </p>

        <ul>
          <li>Mass surveillance and tracking</li>
          <li>Data breaches exposing sensitive information</li>
          <li>Inference of private information from public data</li>
          <li>Lack of transparency about data usage</li>
        </ul>

        <h3>Transparency and Explainability</h3>
        <div className="key-takeaway">
          <p>
            Many AI systems, especially deep learning models, operate as "black boxes"—their
            decision-making processes are opaque and difficult to interpret. This poses challenges when:
          </p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li>Decisions affect people's lives (healthcare, criminal justice, employment)</li>
            <li>Debugging and improving model performance</li>
            <li>Meeting regulatory requirements</li>
            <li>Building user trust</li>
          </ul>
        </div>

        <h3>Accountability and Responsibility</h3>
        <p>
          When AI systems make mistakes or cause harm, questions arise about who is responsible:
        </p>

        <ul>
          <li>The developers who created the system?</li>
          <li>The organization deploying it?</li>
          <li>The users operating it?</li>
          <li>The AI system itself?</li>
        </ul>

        <h3>Job Displacement and Economic Impact</h3>
        <p>
          Automation through AI will transform the labor market, potentially displacing workers while
          creating new opportunities. Society must address:
        </p>

        <ul>
          <li>Retraining and education programs</li>
          <li>Social safety nets</li>
          <li>Income inequality</li>
          <li>The distribution of AI-generated wealth</li>
        </ul>

        <h3>Autonomous Weapons and Dual-Use</h3>
        <p>
          AI technologies can be used for both beneficial and harmful purposes. The development of
          lethal autonomous weapons systems raises existential concerns about warfare and human control.
        </p>
      </div>
    )
  },
  {
    title: "AI Development Tools",
    content: (
      <div>
        <h2>AI Development Tools and Frameworks</h2>

        <p>
          The AI development ecosystem offers a rich array of tools, libraries, and frameworks that
          make building intelligent systems more accessible than ever.
        </p>

        <h3>Machine Learning Frameworks</h3>

        <p><strong>TensorFlow:</strong></p>
        <p>
          Google's open-source framework for machine learning and deep learning. TensorFlow offers
          flexibility for research and robust deployment capabilities for production. It includes
          Keras, a high-level API that simplifies model building.
        </p>

        <p><strong>PyTorch:</strong></p>
        <p>
          Developed by Meta (Facebook), PyTorch has become the preferred framework for research due
          to its intuitive, "Pythonic" interface and dynamic computation graphs. It excels at rapid
          prototyping and debugging.
        </p>

        <p><strong>Scikit-learn:</strong></p>
        <p>
          The go-to library for traditional machine learning algorithms in Python. It provides
          consistent interfaces for classification, regression, clustering, and preprocessing.
        </p>

        <h3>Cloud AI Platforms</h3>
        <div className="highlight-box">
          <p><strong>Major cloud providers offer comprehensive AI services:</strong></p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li><strong>Google Cloud AI:</strong> AutoML, Vertex AI, pre-trained models</li>
            <li><strong>AWS AI Services:</strong> SageMaker, Rekognition, Comprehend</li>
            <li><strong>Azure AI:</strong> Machine Learning Studio, Cognitive Services</li>
          </ul>
        </div>

        <h3>Development Tools</h3>

        <p><strong>Jupyter Notebooks:</strong></p>
        <p>
          Interactive computational environments that combine code, visualizations, and narrative text.
          Essential for data exploration, experimentation, and sharing reproducible research.
        </p>

        <p><strong>MLflow:</strong></p>
        <p>
          An open-source platform for managing the ML lifecycle, including experimentation,
          reproducibility, deployment, and model registry.
        </p>

        <p><strong>Weights & Biases:</strong></p>
        <p>
          Tools for experiment tracking, model visualization, hyperparameter optimization, and
          collaboration across ML teams.
        </p>

        <h3>Pre-trained Models and APIs</h3>
        <p>
          Leverage existing models to jumpstart development:
        </p>

        <ul>
          <li><strong>Hugging Face:</strong> Repository of thousands of pre-trained models for NLP,
          vision, and audio</li>
          <li><strong>OpenAI API:</strong> Access to GPT models and other AI capabilities</li>
          <li><strong>Google Cloud Vision, Speech, Translation APIs</strong></li>
          <li><strong>IBM Watson services</strong></li>
        </ul>

        <h3>Data Tools</h3>
        <p>
          Essential tools for data preparation and management:
        </p>

        <ul>
          <li><strong>Pandas:</strong> Data manipulation and analysis</li>
          <li><strong>NumPy:</strong> Numerical computing</li>
          <li><strong>Apache Spark:</strong> Big data processing</li>
          <li><strong>Label Studio:</strong> Data labeling and annotation</li>
        </ul>
      </div>
    )
  },
  {
    title: "Training AI Models",
    content: (
      <div>
        <h2>Training AI Models: The Process</h2>

        <p>
          Training an AI model involves multiple stages, from data preparation through evaluation.
          Understanding this process is essential for building effective AI systems.
        </p>

        <h3>1. Data Collection and Preparation</h3>
        <div className="highlight-box">
          <p><strong>Data is the foundation of machine learning.</strong> The quality and quantity of
          your data directly impact model performance.</p>
        </div>

        <p><strong>Key steps:</strong></p>
        <ul>
          <li><strong>Data Collection:</strong> Gather relevant data from various sources</li>
          <li><strong>Data Cleaning:</strong> Remove duplicates, handle missing values, fix errors</li>
          <li><strong>Feature Engineering:</strong> Create meaningful features from raw data</li>
          <li><strong>Data Augmentation:</strong> Generate additional training examples (especially
          for images and text)</li>
          <li><strong>Data Splitting:</strong> Divide into training, validation, and test sets</li>
        </ul>

        <h3>2. Model Selection and Architecture Design</h3>
        <p>
          Choose an appropriate model architecture based on:
        </p>

        <ul>
          <li>The type of problem (classification, regression, generation)</li>
          <li>The nature of your data (images, text, tabular)</li>
          <li>Available computational resources</li>
          <li>Latency requirements for deployment</li>
          <li>Interpretability needs</li>
        </ul>

        <h3>3. Training Process</h3>
        <p>
          The model learns by iteratively processing training data:
        </p>

        <ol>
          <li><strong>Forward Pass:</strong> Input data flows through the network to generate predictions</li>
          <li><strong>Loss Calculation:</strong> Compare predictions to actual labels using a loss function</li>
          <li><strong>Backward Pass:</strong> Calculate gradients showing how to adjust weights</li>
          <li><strong>Weight Update:</strong> Adjust model parameters using an optimizer</li>
          <li><strong>Repeat:</strong> Process multiple epochs until convergence</li>
        </ol>

        <h3>4. Hyperparameter Tuning</h3>
        <div className="highlight-box">
          <p>
            Hyperparameters are configuration settings that aren't learned from data. Examples include:
          </p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li>Learning rate</li>
            <li>Batch size</li>
            <li>Number of layers and neurons</li>
            <li>Regularization strength</li>
            <li>Dropout rate</li>
          </ul>
        </div>

        <p>
          Use techniques like grid search, random search, or Bayesian optimization to find optimal
          hyperparameters.
        </p>

        <h3>5. Evaluation and Validation</h3>
        <p>
          Assess model performance on validation and test sets using appropriate metrics. Watch for:
        </p>

        <ul>
          <li><strong>Overfitting:</strong> Model performs well on training data but poorly on new data</li>
          <li><strong>Underfitting:</strong> Model fails to capture patterns in the data</li>
        </ul>

        <h3>6. Deployment and Monitoring</h3>
        <p>
          Once satisfied with performance, deploy the model to production and continuously monitor
          for issues like data drift, where the distribution of input data changes over time.
        </p>
      </div>
    )
  },
  {
    title: "Real-World Applications",
    content: (
      <div>
        <h2>Real-World AI Applications Transforming Our Lives</h2>

        <p>
          AI has moved from research labs into everyday applications that millions of people use
          without even realizing they're interacting with artificial intelligence.
        </p>

        <h3>Virtual Assistants</h3>
        <p>
          Siri, Alexa, Google Assistant, and similar AI-powered assistants use natural language
          processing and speech recognition to:
        </p>

        <ul>
          <li>Understand voice commands</li>
          <li>Answer questions by searching the internet</li>
          <li>Control smart home devices</li>
          <li>Set reminders and manage schedules</li>
          <li>Provide personalized recommendations</li>
        </ul>

        <h3>Recommendation Systems</h3>
        <div className="highlight-box">
          <p>
            The algorithms powering Netflix, Spotify, YouTube, and Amazon recommendations analyze
            your behavior and preferences to suggest content you're likely to enjoy. These systems:
          </p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li>Drive significant user engagement and revenue</li>
            <li>Learn from billions of interactions</li>
            <li>Balance exploration (new content) and exploitation (known preferences)</li>
            <li>Account for contextual factors like time and device</li>
          </ul>
        </div>

        <h3>Autonomous Vehicles</h3>
        <p>
          Self-driving cars represent one of the most complex AI applications, integrating:
        </p>

        <ul>
          <li>Computer vision for perceiving the environment</li>
          <li>Sensor fusion combining data from cameras, lidar, and radar</li>
          <li>Path planning and decision-making</li>
          <li>Control systems for steering, acceleration, and braking</li>
        </ul>

        <p>
          Companies like Tesla, Waymo, and Cruise are making significant progress toward fully
          autonomous driving.
        </p>

        <h3>Content Creation</h3>
        <p>
          AI now assists with creative tasks:
        </p>

        <ul>
          <li><strong>Text Generation:</strong> ChatGPT, Claude, and similar models write articles,
          code, and creative content</li>
          <li><strong>Image Generation:</strong> DALL-E, Midjourney, and Stable Diffusion create
          images from text descriptions</li>
          <li><strong>Music Generation:</strong> AI composes original music in various styles</li>
          <li><strong>Video Editing:</strong> Automated editing, effects, and enhancement</li>
        </ul>

        <h3>Healthcare Diagnostics</h3>
        <p>
          AI assists medical professionals in:
        </p>

        <ul>
          <li>Detecting cancer in medical imaging</li>
          <li>Predicting patient deterioration</li>
          <li>Identifying drug interactions</li>
          <li>Analyzing genetic data</li>
          <li>Monitoring chronic conditions remotely</li>
        </ul>

        <h3>Fraud Detection</h3>
        <p>
          Financial institutions use AI to identify fraudulent transactions in real-time by analyzing
          patterns across millions of transactions, adapting to new fraud tactics continuously.
        </p>

        <h3>Language Translation</h3>
        <p>
          Tools like Google Translate use neural machine translation to provide instant translation
          across 100+ languages, enabling global communication and breaking down language barriers.
        </p>
      </div>
    )
  },
  {
    title: "Challenges and Limitations",
    content: (
      <div>
        <h2>Challenges and Limitations of Current AI</h2>

        <p>
          Despite remarkable progress, AI systems face significant challenges and limitations that
          researchers and practitioners must address.
        </p>

        <h3>Data Requirements and Quality</h3>
        <div className="highlight-box">
          <p><strong>The Data Challenge:</strong></p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li>Deep learning requires massive amounts of labeled data</li>
            <li>Data collection and labeling is expensive and time-consuming</li>
            <li>Biased or unrepresentative data leads to biased models</li>
            <li>Privacy concerns limit access to certain types of data</li>
            <li>Data distribution shifts can degrade model performance</li>
          </ul>
        </div>

        <h3>Computational Resources</h3>
        <p>
          Training state-of-the-art models requires:
        </p>

        <ul>
          <li>Expensive specialized hardware (GPUs, TPUs)</li>
          <li>Significant energy consumption (environmental concerns)</li>
          <li>Expertise in distributed computing</li>
          <li>Access to cloud infrastructure or on-premise clusters</li>
        </ul>

        <p>
          This creates barriers to entry and concentrates AI development power in well-funded
          organizations.
        </p>

        <h3>Lack of Common Sense and Reasoning</h3>
        <p>
          Current AI systems lack human-like understanding:
        </p>

        <ul>
          <li>They struggle with tasks that require common sense knowledge</li>
          <li>They can't reason about cause and effect like humans</li>
          <li>They lack genuine understanding of the concepts they manipulate</li>
          <li>They can be fooled by adversarial examples that wouldn't deceive humans</li>
        </ul>

        <h3>Brittleness and Lack of Robustness</h3>
        <div className="key-takeaway">
          <p>
            AI systems often fail in unpredictable ways when encountering situations different from
            their training data. They lack the robust generalization that humans demonstrate.
          </p>
        </div>

        <h3>Interpretability and Explainability</h3>
        <p>
          The "black box" nature of complex models makes it difficult to:
        </p>

        <ul>
          <li>Understand why a model made a particular decision</li>
          <li>Debug unexpected behavior</li>
          <li>Build trust with users and stakeholders</li>
          <li>Meet regulatory requirements in sensitive domains</li>
        </ul>

        <h3>Transfer Learning Limitations</h3>
        <p>
          While humans easily apply knowledge across domains, AI systems struggle to transfer learning
          from one task to another without significant retraining.
        </p>

        <h3>Hallucination and Reliability</h3>
        <p>
          Language models can generate plausible-sounding but factually incorrect information
          ("hallucinations"), making them unreliable for certain applications without human oversight.
        </p>

        <h3>Ethical and Social Challenges</h3>
        <p>
          Beyond technical limitations, AI faces challenges in:
        </p>

        <ul>
          <li>Ensuring fairness and avoiding discrimination</li>
          <li>Protecting privacy and security</li>
          <li>Maintaining human agency and control</li>
          <li>Addressing job displacement</li>
          <li>Preventing malicious use</li>
        </ul>
      </div>
    )
  },
  {
    title: "The Future of AI",
    content: (
      <div>
        <h2>The Future of Artificial Intelligence</h2>

        <p>
          As we look ahead, AI promises to become even more capable, ubiquitous, and transformative.
          Several key trends and developments will shape the future landscape.
        </p>

        <h3>Artificial General Intelligence (AGI)</h3>
        <div className="highlight-box">
          <p>
            <strong>AGI</strong> refers to AI systems with human-level intelligence across a wide range
            of tasks—not just narrow, specialized capabilities. While current AI excels at specific
            tasks, AGI would demonstrate flexible reasoning, learning, and problem-solving comparable
            to humans.
          </p>
        </div>

        <p>
          Timeline predictions vary widely, from a few years to several decades or never. Achieving
          AGI would represent a fundamental breakthrough with profound implications for society.
        </p>

        <h3>Multimodal AI Systems</h3>
        <p>
          Future AI will seamlessly integrate multiple modalities:
        </p>

        <ul>
          <li>Systems that understand and generate text, images, audio, and video together</li>
          <li>More natural human-AI interaction across sensory channels</li>
          <li>Better contextual understanding by combining information sources</li>
          <li>Unified models that handle diverse tasks without specialization</li>
        </ul>

        <h3>AI in Scientific Discovery</h3>
        <p>
          AI is accelerating research across fields:
        </p>

        <ul>
          <li><strong>Biology:</strong> Protein folding (AlphaFold), drug discovery, genomics</li>
          <li><strong>Physics:</strong> Simulation, particle physics, fusion energy</li>
          <li><strong>Materials Science:</strong> Discovering new materials with desired properties</li>
          <li><strong>Climate Science:</strong> Modeling, prediction, and optimization</li>
        </ul>

        <h3>Edge AI and Tiny ML</h3>
        <p>
          Moving AI processing from cloud servers to edge devices:
        </p>

        <ul>
          <li>Reduced latency for real-time applications</li>
          <li>Enhanced privacy by processing data locally</li>
          <li>Lower costs and energy consumption</li>
          <li>Offline functionality</li>
        </ul>

        <h3>AI-Augmented Human Capabilities</h3>
        <div className="highlight-box">
          <p>
            Rather than replacing humans, future AI will augment human intelligence and creativity:
          </p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li>AI assistants that enhance productivity and decision-making</li>
            <li>Creative tools that amplify human artistic expression</li>
            <li>Educational systems personalized to individual learning styles</li>
            <li>Accessibility tools helping people with disabilities</li>
          </ul>
        </div>

        <h3>Responsible AI Development</h3>
        <p>
          Growing emphasis on building AI systems that are:
        </p>

        <ul>
          <li>Fair and unbiased</li>
          <li>Transparent and explainable</li>
          <li>Privacy-preserving</li>
          <li>Aligned with human values</li>
          <li>Robustly tested and verified</li>
        </ul>

        <h3>AI Governance and Regulation</h3>
        <p>
          Governments and organizations are developing frameworks to:
        </p>

        <ul>
          <li>Establish safety standards for AI systems</li>
          <li>Address liability and accountability</li>
          <li>Protect against misuse</li>
          <li>Ensure equitable access to AI benefits</li>
          <li>Foster international cooperation</li>
        </ul>

        <h3>Quantum AI</h3>
        <p>
          The convergence of quantum computing and AI could enable breakthroughs in optimization,
          simulation, and cryptography—though practical applications remain years away.
        </p>
      </div>
    )
  },
  {
    title: "Getting Started with AI",
    content: (
      <div>
        <h2>Getting Started with AI: Your Learning Path</h2>

        <p>
          Whether you're a student, professional, or curious enthusiast, there's never been a better
          time to start learning about AI. Here's a roadmap to guide your journey.
        </p>

        <h3>Foundation: Prerequisites</h3>
        <div className="highlight-box">
          <p><strong>Build your foundation in these areas:</strong></p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li><strong>Programming:</strong> Python is the lingua franca of AI/ML</li>
            <li><strong>Mathematics:</strong> Linear algebra, calculus, probability, and statistics</li>
            <li><strong>Data Analysis:</strong> Understanding data structures and manipulation</li>
          </ul>
        </div>

        <h3>Learning Resources</h3>

        <p><strong>Online Courses:</strong></p>
        <ul>
          <li><strong>Coursera:</strong> Andrew Ng's Machine Learning and Deep Learning Specialization</li>
          <li><strong>Fast.ai:</strong> Practical deep learning for coders</li>
          <li><strong>DeepLearning.AI:</strong> Comprehensive AI courses</li>
          <li><strong>Udacity:</strong> AI nanodegree programs</li>
        </ul>

        <p><strong>Books:</strong></p>
        <ul>
          <li>"Hands-On Machine Learning" by Aurélien Géron</li>
          <li>"Deep Learning" by Goodfellow, Bengio, and Courville</li>
          <li>"Pattern Recognition and Machine Learning" by Christopher Bishop</li>
          <li>"Artificial Intelligence: A Modern Approach" by Russell and Norvig</li>
        </ul>

        <p><strong>Interactive Platforms:</strong></p>
        <ul>
          <li><strong>Kaggle:</strong> Competitions, datasets, and community learning</li>
          <li><strong>Google Colab:</strong> Free cloud-based Jupyter notebooks with GPU access</li>
          <li><strong>Hugging Face:</strong> Pre-trained models and tutorials</li>
        </ul>

        <h3>Hands-On Projects</h3>
        <p>
          Build your portfolio with these project ideas:
        </p>

        <ol>
          <li><strong>Beginner:</strong> Image classification (MNIST digits, CIFAR-10)</li>
          <li><strong>Intermediate:</strong> Sentiment analysis on movie reviews or tweets</li>
          <li><strong>Advanced:</strong> Build a chatbot or recommendation system</li>
          <li><strong>Real-world:</strong> Solve a problem in your domain of interest</li>
        </ol>

        <h3>Community and Networking</h3>
        <p>
          Engage with the AI community:
        </p>

        <ul>
          <li>Join AI/ML meetups and conferences</li>
          <li>Participate in Kaggle competitions</li>
          <li>Contribute to open-source AI projects</li>
          <li>Follow AI researchers and practitioners on social media</li>
          <li>Join online forums (Reddit r/MachineLearning, Stack Overflow)</li>
        </ul>

        <h3>Career Paths in AI</h3>
        <div className="highlight-box">
          <p><strong>AI offers diverse career opportunities:</strong></p>
          <ul style={{ marginTop: '1rem', marginBottom: 0 }}>
            <li><strong>Machine Learning Engineer:</strong> Building and deploying ML systems</li>
            <li><strong>Data Scientist:</strong> Extracting insights and building models</li>
            <li><strong>AI Research Scientist:</strong> Advancing the state of the art</li>
            <li><strong>AI Product Manager:</strong> Defining AI-powered products</li>
            <li><strong>AI Ethics Specialist:</strong> Ensuring responsible AI development</li>
          </ul>
        </div>

        <h3>Keep Learning</h3>
        <p>
          AI evolves rapidly. Stay current by:
        </p>

        <ul>
          <li>Reading AI research papers (arXiv.org)</li>
          <li>Following AI conferences (NeurIPS, ICML, CVPR)</li>
          <li>Subscribing to AI newsletters and podcasts</li>
          <li>Experimenting with new tools and techniques</li>
          <li>Teaching others what you've learned</li>
        </ul>

        <div className="key-takeaway">
          <strong>Final Thoughts</strong>
          <p style={{ marginTop: '1rem', marginBottom: 0 }}>
            AI is not just a technology—it's a tool for solving problems and creating value. Focus on
            understanding fundamentals, building practical skills, and applying AI to areas you're
            passionate about. The journey is challenging but immensely rewarding.
          </p>
        </div>
      </div>
    )
  }
];

export default function Home() {
  const [currentPage, setCurrentPage] = useState(0);

  const goToNextPage = () => {
    if (currentPage < ebookPages.length - 1) {
      setCurrentPage(currentPage + 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const goToPreviousPage = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <div className="container">
      <div className="ebook-viewer">
        <div className="ebook-header">
          <h1>Understanding AI & Machine Learning</h1>
          <p>A Comprehensive Guide to the Future of Technology</p>
        </div>

        <div className="page-navigation">
          <div className="page-info">
            Page {currentPage + 1} of {ebookPages.length}
          </div>
          <div className="nav-buttons">
            <button
              className="nav-button"
              onClick={goToPreviousPage}
              disabled={currentPage === 0}
            >
              ← Previous
            </button>
            <button
              className="nav-button"
              onClick={goToNextPage}
              disabled={currentPage === ebookPages.length - 1}
            >
              Next →
            </button>
          </div>
        </div>

        <div className="page-content">
          {ebookPages[currentPage].content}
        </div>

        <div className="page-navigation">
          <div className="page-info">
            Page {currentPage + 1} of {ebookPages.length}
          </div>
          <div className="nav-buttons">
            <button
              className="nav-button"
              onClick={goToPreviousPage}
              disabled={currentPage === 0}
            >
              ← Previous
            </button>
            <button
              className="nav-button"
              onClick={goToNextPage}
              disabled={currentPage === ebookPages.length - 1}
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
