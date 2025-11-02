# 🍳 GPT-2 Recipe Generator

An AI-powered recipe generator using fine-tuned GPT-2 with LoRA (PEFT) for parameter-efficient training.

## 📁 Project Structure

```
├── fine-tune-gpt2-recipe.ipynb    # Main training notebook with evaluation
├── streamlit_recipe_app.py        # Interactive web application
├── requirements.txt               # Python dependencies
├── models/                        # Model checkpoints and data
│   ├── model.pkl                 # Fine-tuned model weights
│   ├── *.pkl                     # Training checkpoints
│   └── *.csv                     # Processed datasets
├── tokenizer/                     # Tokenizer files
├── peft_model/                    # LoRA adapter files
└── 3A2M_EXTENDED.csv             # Original dataset
```

## 🚀 Quick Start

### ⚠️ Important: Download Model Files First

The trained model files are stored in GitHub Releases due to their large size:

1. **Go to [Releases](https://github.com/YOUR_USERNAME/YOUR_REPO/releases)**
2. **Download these files from the latest release:**
   - `models.zip` - Main model file and training data  
   - `tokenizer.zip` - Custom tokenizer files
   - `peft_model.zip` - PEFT/LoRA adapter files
3. **Extract all zip files** to this project directory

### Running the Application

### 1. Model Evaluation (Jupyter Notebook)

The notebook contains independent evaluation cells that can be run separately from training:

```bash
# Open the notebook
jupyter notebook fine-tune-gpt2-recipe.ipynb

# Run the evaluation sections (last 6 cells):
# - Cell: "INDEPENDENT MODEL EVALUATION SECTION" (markdown)
# - Cell: "EVALUATION SETUP - Independent Model Loading"
# - Cell: "LOAD FINE-TUNED MODEL FOR EVALUATION" 
# - Cell: "LOAD TEST DATA FOR EVALUATION"
# - Cell: "RECIPE GENERATION FUNCTION"
# - Cell: "COMPREHENSIVE EVALUATION METRICS"
# - Cell: "SAMPLE GENERATION SHOWCASE"
# - Cell: "EVALUATION VISUALIZATION AND FINAL REPORT"
```

### 2. Streamlit Web Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_recipe_app.py
```

The app will be available at `http://localhost:8501`

## 📊 Evaluation Features

The independent evaluation cells provide:

- **✅ Model Loading**: Loads fine-tuned model with LoRA adapters
- **✅ Comprehensive Metrics**: BLEU scores, generation diversity, success rates
- **✅ Sample Generation**: Multiple creativity levels and examples
- **✅ Performance Analysis**: Parameter counts, generation quality assessment
- **✅ Visual Reports**: Detailed evaluation summaries and recommendations

### Key Evaluation Metrics:

- **BLEU Score**: Measures generation quality against references
- **Generation Diversity**: Uniqueness of generated recipes  
- **Success Rate**: Percentage of successful generations
- **Length Statistics**: Analysis of generated recipe lengths
- **Model Performance**: Parameter efficiency and device utilization

## 🎨 Streamlit App Features

- **Interactive Input**: Ingredient lists and recipe titles
- **Creativity Levels**: Conservative, Balanced, Creative, Experimental
- **Real-time Generation**: Instant recipe creation
- **Recipe History**: Save and view previous generations
- **Export Functionality**: Download recipes as Markdown files
- **Advanced Settings**: Fine-tune temperature, top-p, max length

## 🔧 Model Details

- **Base Model**: GPT-2 (124M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) via PEFT
- **Trainable Parameters**: ~0.3M (0.25% of total)
- **Training Data**: Recipe dataset (ingredients → instructions)
- **Training Epochs**: 2 epochs with early stopping

## 📋 Usage Examples

### Evaluation (Notebook)
```python
# The evaluation cells are completely independent
# Just run them in order after the model is saved

# Example generation in evaluation:
prompt = "Ingredients: eggs, flour, milk\nTitle: Pancakes\nRecipe:\n"
recipe = generate_recipe(eval_model, eval_tokenizer, prompt)
```

### Streamlit App
1. Enter ingredients: `"chicken, garlic, herbs, olive oil"`
2. Enter title: `"Herb Roasted Chicken"`
3. Select creativity level: `"Balanced"`
4. Click `"Generate Recipe"`
5. View, save, or download your recipe!

## 🛠️ Technical Requirements

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **Transformers**: 4.30+
- **PEFT**: 0.4+
- **Streamlit**: 1.28+
- **CUDA**: Optional but recommended for faster generation

## 📈 Performance

- **Generation Speed**: ~2-5 seconds per recipe (GPU)
- **Memory Usage**: ~2GB VRAM (with LoRA)
- **Model Size**: ~500MB (base) + ~10MB (LoRA adapters)
- **Quality**: High coherence, contextually appropriate recipes

## 🔍 Troubleshooting

### Common Issues:

1. **Model not found**: Ensure `peft_model/` and `tokenizer/` folders exist
2. **CUDA errors**: Set device to CPU in the code if no GPU available
3. **Import errors**: Run `pip install -r requirements.txt`
4. **Streamlit issues**: Update Streamlit to latest version

### File Locations:
- Model files should be in the same directory as the scripts
- Check that `models/model.pkl` exists for evaluation
- Ensure `peft_model/` contains adapter files

## 🎯 Model Quality Assessment

The evaluation provides a comprehensive quality score based on:
- ✅ Model loading success
- ✅ Evaluation metrics completion  
- ✅ Generation showcase results
- ✅ Test data processing
- ✅ BLEU score performance

Scores of 4-5/5 indicate production-ready models.

## 🔄 Extending the Project

- **Add more metrics**: ROUGE, BERTScore, custom recipe quality metrics
- **Improve UI**: Add more Streamlit features, better styling
- **Model variants**: Try different base models (GPT-Neo, T5)
- **Dataset expansion**: Include more diverse recipe sources
- **Deployment**: Deploy to Streamlit Cloud, Hugging Face Spaces

## 📝 License

This project is for educational and research purposes. Please respect the original dataset licenses and model terms of use.

---

*Happy cooking with AI! 🍳🤖*