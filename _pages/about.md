---
permalink: /
title: "👋 Hi There, I'm Hrithick"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

🧑‍💻️ I have 3.8YOE as a Data Scientist, where I mostly worked on NLP, Vision, and Multimodal Deep Learning. My journey started with implementing object detection on drones 🛸, then developing Question Answering Systems using Knowledge Graphs 📚, and later Information Extraction from Documents leveraging Pix2Struct, Donut, and LayoutLM models 📄. Most recently, I have been building chatbots leveraging **LLMs, Vector Database** and **Advance RAG pipeline**.

🎓📃 My interests are in Large Language Models (LLMs), Multimodal Deep Learning, and Generative Adversarial Networks (GANs). 🌐

📝 I'm currently reading academic papers on LLMs. 📚 I also believe that software engineering skill is crucial, so I'm actively practicing Data Structures and Algorithms on Leetcode. 💻

## 💻 Work Experience

### 💼 OGMA IT LLC
1. Developed 'The Credit Genius', a RAG-based chatbot designed specifically for performing Question Answering tasks sourced from the client's written material and Video Lectures. Utilized advanced techniques such as Sentence Window Retrieval and Hypothetical Questions (HyDe) within the RAG pipeline to achieve superior answer relevancy, Context Recall, and Faithfulness. Completed the prject end-to-end  by Dockerizing the application and deploying the system on AWS EC2 instance.
   
2. Built a Multimodal DocumentAI system for extracting information from Logistic Tickets, leading to a 91% reduction in the time and cost dedicated to manually verifying trucking tickets. Initially fine-tuned pix2struct-textcaps for information extraction in version 1, but later transitioned to Claude 3-opus.

### 💼 Openstream.ai

1. Collaborated with the team to build and deploy question-answering systems leveraging knowledge graphs (Cypher and Neo4j) and language models models (BERT, T5, DistilBERT).
2. Built a pipeline to automatically extract structured tables from documents into a knowledge graph using Table Transformer (DETR fine-tuned to detect tables) and YOLO.


## 📚 Research Paper Implementations

1. Implemented the Transformer paper titled **[Attention Is All You Need](https://arxiv.org/pdf/1706.03762v1.pdf)** using TensorFlow 2.0. 📄 This project helped me gain a practical understanding of the Transformer model, GPT, and BERT. 🤖 The Transformer model was trained to perform machine translation from English to Spanish. 🇬🇧➡️🇪🇸 The code can be found [here](https://github.com/hrithickcodes/transformer-tf).


2. Implemented Vision Transformer, the paper titled **[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929v1.pdf)**, which had a significant impact on vision models by leveraging transformer models for image classification, achieving state-of-the-art results on ImageNet. The code, implemented by me, provides an easy way to load and fine-tune Vision Transformer models. The code can be found [here](https://github.com/hrithickcodes/vision_transformer_tf).
  ```python
  # Load any vit model
  from vit import viT
  vit_large = viT(vit_size="ViT-LARGE32")
  vit_large.from_pretrained(pretrained_top=True)
  ```


3. Reproduced the inference of the paper titled **[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434v1.pdf)** also known as **DCGAN**, to generate faces. The model was implemented from scratch using TensorFlow 1.x (tf.compat.v1). It was trained for 150 epochs, after which it was able to generate decent fake face images. The code can be found [here](https://github.com/hrithickcodes/Face_generation_using_DCGAN).

<table>
  <tr>
    <td>
      <p align="center">
        <img src="https://raw.githubusercontent.com/hrithickcodes/Face_generation_using_DCGAN/main/generated_faces/fake_faces.jpg" alt="Generated Face Images" title="Generated Fake Faces" width="350"/>
      </p>
    </td>
    <td>
      <p align="center">
        <img src="https://raw.githubusercontent.com/hrithickcodes/Face_generation_using_DCGAN/main/GIF/FaceGan.gif" alt="Animated Gif" title="Animated Gif" width="150"/>
      </p>
    </td>
  </tr>
</table>