# NMF Explainability Web App

## About
This project is an interactive web application for performing **Non-negative Matrix Factorization (NMF)** and generating **interpretable explanations** of the results.

The application provides a complete pipeline:
- selection of the optimal number of latent factors (k)
- execution of the final NMF model
- clustering in the latent space
- generation of **fuzzy explanations** for matrices W and H
- **sample-level interpretation** of latent factors

The goal is to improve the **interpretability of NMF** through fuzzy logic.

## Developing

To run the project, make sure you have:

- **Python 3.x**

## Installation

1. Clone the repository and move into the project folder.

2. (Optional) Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the Required Packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once all dependencies are installed:

```bash
python3 run.py
```

This command will start the server locally, allowing you to access and interact with the application via your web browser at `http://127.0.0.1:5000`.

## Authors

Vito Valentini