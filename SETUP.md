# PathForge — Setup

## 1. Create and activate a virtual environment

**Windows**
    python -m venv venv
    venv\Scripts\activate

**Linux/macOS**
    python3 -m venv venv
    source venv/bin/activate

## 2. Install dependencies
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

## 3. Generate the sample job dataset (already included, but to regenerate)
    python data/generate_sample_jobs.py

## 4. (Optional) Configure Gemini
    cp .env.example .env
    # edit .env and add GEMINI_API_KEY=your_key_here
The app runs fine without this step — it uses a built-in fallback narrative generator.

## 5. Run the app
    streamlit run app.py

## 6. Run tests
    pytest tests/ -v

## 7. Run the evaluation report
    python -m src.evaluation
