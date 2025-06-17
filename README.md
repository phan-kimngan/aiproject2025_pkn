#  Subject: AI PROJECT 2025

Prof: Hyung–Jeong Yang

## Task: Development of AI Text Summarization Feature

Name: Kim Ngan Phan

⛔ ***Only used for demonstration of the subject***
### Overview
As communication becomes increasingly digital and remote, a large amount of voice content is generated in diverse contexts. Converting this voice con tent into structured and meaningful insights is crucial for effective in formation retrieval, documentation , and decision-making. This project focuses on developing an automated system that creates informative summaries from full-text transcriptions produced by speech-to-text (STT) systems. It is specifically designed for Korean conversations in the con text of personal and relationship topics.
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
* Json data is collect from aihub
* Create the following directories to store the original JSON files:

  * **Training/[라벨]한국어대화요약_train/[라벨]한국어대화요약_train**
  * **Validation/[라벨]한국어대화요약_valid/[라벨]한국어대화요약_valid**
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

