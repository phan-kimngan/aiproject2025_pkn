#  Subject: AI PROJECT 2025

Prof: Hyung–Jeong Yang

## Task: Development of AI Text Summarization Feature

Name: Kim Ngan Phan

⛔ ***Only used for demonstration of the subject***
### Overview
As communication becomes increasingly digital and remote, a large amount of voice content is generated in diverse contexts. Con verting this voice con tent into structured and meaningful insights is crucial for effec tive in forma tion retrieval, documen tation , and decision-making. This project focuses on developing an automa ted system that creates informa tive summaries from full- text transcriptions produced by speech-to-text (STT) systems. It is specifically designed for Korean conversations in the con text of personal and relationship topics.
### Set up environment. 
+ Create a python project_ai_env environment using conda or other tools.

+ Activate project_ai_env environment
```bash
conda activate project_ai_env
```
+ Instead packages in requirements.txt
```bash
pip install -r requirements.txt
```
### Project Structure

* Create the following directories to store the original JSON files:

  * **Training/\[라벨]한국어대화요약\_train/\[라벨]한국어대화요약\_train**
  * **Validation/\[라벨]한국어대화요약\_valid/\[라벨]한국어대화요약\_valid**
* Create a **data/** directory to store the preprocessed files.

### Training the Model

1. Run the following command to preprocess the data and save the results in the `data/` folder:

   ```
   python load_data.py
   ```
2. Run the following command to train the model using the preprocessed data. The trained model will be saved in the `fine_tuned_digit82_kobart-2/` directory:

   ```
   python run_model.py
   ```
3. Run the following command to evaluate the performance of the `fine_tuned_digit82_kobart-2` model:

   ```
   python evaluation.py
   ```
4. To launch the application interface, run:

   ```
   streamlit run Main.py
   ```

