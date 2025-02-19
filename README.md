# Movie Recommendation System

## 📌 Overview
This project is a **content-based movie recommendation system** that suggests similar movies based on a user's input description. It uses **TF-IDF vectorization** and **cosine similarity** to find the most relevant matches from a dataset of top 1000 IMDb movies.

## 🚀 How It Works
1. **Load dataset**: Reads the `imdb_top_1000.csv` file containing movie titles and descriptions.
2. **Preprocess text**: Converts movie descriptions into TF-IDF vectors.
3. **User query processing**: Transforms the user's input into a vector.
4. **Compute similarity**: Uses **cosine similarity** to compare the input query with movie descriptions.
5. **Return top matches**: Outputs the **top 5 most relevant movies**.

---

## 📂 Dataset
- **File**: `imdb_top_1000.csv`
- **Source**: IMDb Top 1000 movies dataset.
- **Columns Used**:
  - `Series_Title`: Movie title.
  - `Overview`: Short movie description.

---

## ⚙️ Setup & Installation
### **1️⃣ Install Dependencies**
Ensure you have Python installed. Then install required libraries:
```bash
pip install -r requirements.txt
```
_(If `requirements.txt` is not available, manually install:_ `pandas`, `numpy`, `scikit-learn` _)_

### **2️⃣ Running the Recommendation System**
Run the script with a **movie preference description**:
```bash
python recommend.py "I love action movies"
```

### **3️⃣ Expected Output**
```
Top Recommendations:
1. The Incredibles (Similarity: 0.1409)
2. Barton Fink (Similarity: 0.1348)
3. Saving Private Ryan (Similarity: 0.1328)
4. Clerks (Similarity: 0.1147)
5. Me and Earl and the Dying Girl (Similarity: 0.0993)
```

---

## 📌 Explanation of Techniques
### **1️⃣ TF-IDF (Term Frequency-Inverse Document Frequency)**
- Converts text into numerical vectors by giving higher importance to **less common but meaningful words**.
- Example: "action" gets a **higher score** if it appears in fewer movie descriptions.

### **2️⃣ Cosine Similarity**
- Measures how **similar** two text descriptions are based on their TF-IDF vectors.
- Values range from **0 (completely different)** to **1 (exact match)**.

---

## 📹 Demo Video
Watch the demo: [Click Here](YOUR_VIDEO_LINK_HERE)

---

## 💰 Salary Expectation
(Expected Salary: **$0 per month**)

---

## 🛠 Future Improvements
- ✅ Improve text preprocessing (lemmatization, stopword removal).
- ✅ Experiment with **Word Embeddings** instead of TF-IDF.
- ✅ Expand dataset for better recommendations.

---

## 📩 Contact
For any questions, reach out at: [yeon971105@icloud.com/www.linkedin.com/in/jewon-yeon-ai-scientist/GitHub].
