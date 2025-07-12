# Meta-Learning for Object Recognition and Simulated Task Generation in Human-Robot Collaboration

This project explores how **meta-learning techniques** can be used for **object recognition** in human-robot collaborative environments, and how the results can be translated into **simulated robotic actions** via **G-code-style command generation**.

> 📍 **University**: Leeds Trinity University  
> 📅 **Project Period**: 2024–2025  
> 🎓 **MSc Data Science and Artificial Intelligence**  
> 👨‍💻 **Author**: [Safa Dana](https://www.linkedin.com/in/safa-dana)

---

## 🧠 Project Overview

In human-robot collaboration (HRC) environments like smart homes or factories, it's important for robots to recognize new objects and adapt quickly. This project proposes a **few-shot learning approach using Prototypical Networks**, which allows robots to recognize new objects using only a few labeled examples. The system then **extracts the object positions and generates simplified G-code commands**, simulating robotic actions such as “move”, “pick”, or “place”.

---

## 🎯 Objectives

- Implement meta-learning model (Prototypical Networks) for few-shot classification.
- Detect object positions from synthetic visual data using OpenCV.
- Generate simplified G-code instructions simulating robotic behavior.
- Evaluate model performance and compare it with a baseline transfer learning approach.

---

## 🛠️ Technologies Used

| Category               | Tools & Libraries                          |
|------------------------|--------------------------------------------|
| Programming Language   | Python                                     |
| Frameworks             | PyTorch, learn2learn                       |
| Image Processing       | OpenCV                                     |
| Notebook Environment   | Jupyter Notebook                           |
| Dataset                | Mini-ImageNet, synthetic object images     |
| Simulation Output      | Plain text `.gcode` commands               |

---

## 📂 Project Structure

```
├── data/                      # Processed and synthetic images
├── models/                    # Meta-learning models
├── gcode_generator/          # Python scripts for G-code logic
├── notebooks/                # Jupyter notebooks for experimentation
├── results/                  # Evaluation metrics and plots
├── utils/                    # Helper functions and tools
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/safaenet/Final_MSc_Project.git
cd Final_MSc_Project
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Open and run the main Jupyter notebook inside the `notebooks/` folder to train the model and simulate G-code generation.

---

## 📊 Evaluation Metrics

- **Classification Accuracy**
- **Precision / Recall / F1-Score**
- **Adaptation Time**
- **Logical correctness of generated G-code**

---

## 🔍 Research Questions

1. How well can Prototypical Networks recognize unseen object categories with few samples?
2. Can we reliably translate those recognitions into meaningful G-code sequences for simulated tasks?

---

## 📈 Results Summary (to be completed)

You will find detailed experiment results and comparison tables inside the `results/` folder.

---

## 🔮 Future Scope

- Integration with real robotic arms and physical execution of generated G-code.
- Human-in-the-loop learning via gesture or voice command interpretation.
- Extension to multimodal HRC scenarios (vision + speech + text).

---

## 📜 License

This project is part of a university MSc program and is intended for academic and educational use only.

---

## 📬 Contact

For questions or collaborations:

**Safa Dana**  
📧 safa.dh@gmail.com  
📞 +44 7777 941862  
🔗 [LinkedIn](https://www.linkedin.com/in/safa-dana)
