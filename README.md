# 📊 RiskMetrics: Credit Risk Intelligence Dashboard


## 📁 Project Structure

```
    data-mining-asm/
    ├── .gitignore
    ├── README.md
    ├── Copy_of_Data_Mining.ipynb
    ├── dataset.xls
    ├── app/
    │   ├── app.py
    │   ├── logo-removedbg.png
    │   ├── model/
    │   ├── model_utils.py
    │   └── style.css
    └── train-model/
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
    git clone https://github.com/Clara1243/data-mining-asm.git
    cd data-mining-asm
```

### 2. Create Virtual Environment

```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows:
    .\venv\Scripts\activate

    # Activate on macOS/Linux:
    source venv/bin/activate
```

### 3. Install Dependencies

```bash
    pip install streamlit pandas xgboost plotly openpyxl xlrd
```

## Running the Application

```bash
    streamlit run app.py
```

© 2026 Data Mining Project v3.0