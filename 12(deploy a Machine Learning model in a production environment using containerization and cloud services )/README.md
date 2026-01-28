# How to Run the ML Prediction App (Terminal + GUI)


Install dependencies:

    pip install flask numpy scikit-learn

------------------------------------------------------------------------

## Project Files

Your folder should contain:

-   app.py\
-   model.pkl\
-   requirements.txt (optional but recommended)

------------------------------------------------------------------------

## Run the Application

Open terminal in the project folder and run:

    python app.py

You should see output similar to:

    Running on http://127.0.0.1:5000

This means the server is running successfully.

------------------------------------------------------------------------

## Option 1: Use the GUI (Browser Mode)

1.  Open your browser\

2.  Go to:

        http://127.0.0.1:5000

3.  You will see a form with four input fields\

4.  Enter values (for example):

        5.1  
        3.5  
        1.4  
        0.2  

5.  Click **Predict**

You will see output such as:

    The flower belongs to: Setosa

------------------------------------------------------------------------

## Option 2: Use Terminal (API Mode)

You can also send a request using curl.

Run this command in a new terminal:

    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"data\": [5.1, 3.5, 1.4, 0.2]}"

Expected output:

    {"prediction": 0}

Another example:

    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"data\": [2.1, 4.5, 5.4, 6.2]}"

Output:

    {"prediction": 2}

------------------------------------------------------------------------
