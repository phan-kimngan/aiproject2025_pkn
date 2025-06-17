#  Subject: AI PROJECT 2025

Prof: Hyung–Jeong Yang

## Task: Combining Video Class Solution with Face Recognition Solution

Name: Kim Ngan Phan

ID: 24847

⛔ ***Only used for demonstration of the subject***
### Overview
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

