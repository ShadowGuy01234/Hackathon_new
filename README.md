# Hackathon_new

This project showcases a document classification pipeline built using Transformer models. It allows classification of documents into predefined categories such as AI/Tech and Finance using files in CSV, JSON, or TXT formats. The pipeline is designed for scalability, flexibility, and iterative improvements.

---

## **Features**

- **Customizable Categories**: Predefined categories such as AI and Finance.
- **Multi-File Support**: Supports document classification from `.csv`, `.json`, and `.txt` files.
- **Transformer Models**: Leverages pre-trained transformer models like `DistilBERT` for classification.
- **Visualization**: Includes graphs to visualize classification results.
- **Scalable Design**: Iterative training and classification for continuous improvements.

---

## **Folder Structure**

```
Hackathon_new/
├── data_loader.py               # Script for loading data from multiple file formats
├── main_classifier.py           # Main script for classifying documents
├── final_model/                 # Trained model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── test_data/                   # Test files for classification
│   ├── test1.json
│   ├── test2.csv
│   ├── test3.txt
├── README.md                    # Project documentation
```

---

## **Setup and Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ShadowGuy01234/Hackathon_new.git
   cd Hackathon_new
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   transformers
   torch
   pandas
   matplotlib
   seaborn
   ```

3. **Prepare Model**:
   Ensure the `final_model` directory contains the trained model files:
   - `pytorch_model.bin`
   - `config.json`
   - `tokenizer.json`

   If you don't have these files, train a model using a separate training script.

4. **Prepare Data**:
   Add test data files (`.csv`, `.json`, `.txt`) to the `test_data` directory.

---

## **Usage**

### **Classifying Documents**
1. Run the classification script:
   ```bash
   python main_classifier.py
   ```

2. **Results**:
   - Displays classification results for each file in `test_data`.
   - Generates a bar graph visualizing document categories.

---

## **Development**

### **Adding New Data**
To classify additional files:
- Add them to the `test_data` directory.
- Supported formats:
  - `.csv` with a `text` column.
  - `.json` with a `text` key in each object.
  - `.txt` with one document per line.

### **Customizing Categories**
Update the `label_mapping` in `main_classifier.py`:
```python
label_mapping = {
    0: "AI",
    1: "Finance",
    # Add more categories here
}
```

---

## **Visualization**
A bar chart is generated to show the distribution of documents across categories. Ensure you have `matplotlib` and `seaborn` installed.

---

## **Contributing**
Contributions are welcome! Feel free to fork the repository and submit a pull request with improvements or bug fixes.

---

## **Contact**
For questions or suggestions, reach out via [GitHub Issues](https://github.com/ShadowGuy01234/Hackathon_new/issues).

---

## **License**
This project is open source under the MIT License.