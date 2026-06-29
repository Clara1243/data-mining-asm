# RiskMetrics: Credit Risk Intelligence Dashboard 📊

An enterprise-grade Decision-Support System built for the **CDS6314 Data Mining** project. This application utilizes a Random Forest machine learning pipeline to evaluate credit card applicants, predict default probabilities, and extract global risk insights using Association Rules.

## 📁 Repository Structure

Ensure your repository contains the following core files before deployment:

* `app.py` - The main Streamlit dashboard application.
* `model_utils.py` - Background logic, Explanable AI (SHAP) extraction, and EDA processing.
* `style.css` - Custom UI/UX styling for the dashboard.
* `requirements.txt` - Required Python dependencies for the cloud server.
* `model/` - Directory containing the pre-trained `.joblib` model and transformer files.
* `logo-removedbg.png` - Application branding asset.

---

## 🚀 How to Deploy on Streamlit Community Cloud (Recommended)

This dashboard is designed to run 24/7 on Streamlit Community Cloud, separating the frontend UI from the local training environment.

### Step 1: Prepare GitHub
1. Ensure all files listed in the Repository Structure above are committed and pushed to your `main` branch.
2. Double-check that your `model/` folder was successfully uploaded and is not being blocked by a `.gitignore` file.

### Step 2: Connect to Streamlit
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
2. Click the **"New app"** button in the top right corner.
3. Grant Streamlit authorization to access your GitHub repositories if prompted.

### Step 3: Launch the Application
1. **Repository:** Select your project repository from the dropdown menu.
2. **Branch:** Set this to `main` (or the branch where your latest code lives).
3. **Main file path:** Type exactly `app.py`.
4. Click **Deploy!**

*Note: Streamlit will take 1-3 minutes on the first boot to read your `requirements.txt` and install the necessary dependencies.*

---

## 💻 How to Run Locally (For Development)

If you prefer to run the application on your local machine to test changes:

1. Clone this repository to your local machine.
2. Open your terminal and navigate to the project folder.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Lauch the Streamlit server:
    ```bash
    streamlit run app.py
    ```
5. The dashboard will automatically open in your default web browser at http://localhost:8501.


