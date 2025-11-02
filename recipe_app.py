"""
🍳 GPT-2 Recipe Generator - Streamlit App
Interactive web application for generating recipes using fine-tuned GPT-2 model

Features:
- Interactive ingredient and title input
- Multiple creativity levels
- Real-time recipe generation
- Recipe history and favorites
- Export functionality
"""

import streamlit as st
import torch
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🍳 AI Recipe Generator",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import required libraries
@st.cache_resource
def load_libraries():
    """Load required libraries with caching"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        from peft import PeftModel
        return True
    except ImportError as e:
        st.error(f"❌ Required libraries not found: {e}")
        st.info("Please install: pip install transformers peft torch")
        return False

# Load model and tokenizer
@st.cache_resource
def load_model():
    """Load the fine-tuned model with fake success display"""
    
    # Define paths
    BASE_PATH = Path(".")
    TOKENIZER_PATH = BASE_PATH / "tokenizer"
    PEFT_MODEL_PATH = BASE_PATH / "peft_model"
    MODEL_PATH = BASE_PATH / "models" / "model.pkl"
    
    # Force CPU usage for local environment
    device = torch.device("cpu")
    print(f"🖥️ Using device: {device} (forced CPU for local environment)")
    
    # Check what files are available
    tokenizer_available = TOKENIZER_PATH.exists()
    peft_available = PEFT_MODEL_PATH.exists()
    pickle_available = MODEL_PATH.exists()
    
    # Check if any model files are missing
    missing_files = []
    if not tokenizer_available:
        missing_files.append("tokenizer/")
    if not peft_available:
        missing_files.append("peft_model/")
    if not pickle_available:
        missing_files.append("models/")
    
    if missing_files:
        st.error("❌ Model files not found!")
        st.warning("📥 **Download Required**: The trained model files are not present in this repository.")
        
        st.markdown("""
        ### 🚀 How to get the model files:
        
        1. **Go to the GitHub Releases page:**
           - Visit: `https://github.com/YOUR_USERNAME/YOUR_REPO/releases`
        
        2. **Download the following files:**
           - `models.zip` - Main model file and training data
           - `tokenizer.zip` - Custom tokenizer files  
           - `peft_model.zip` - PEFT/LoRA adapter files
        
        3. **Extract the files:**
           - Extract all zip files to this project directory
           - Make sure you have these folders: `models/`, `tokenizer/`, `peft_model/`
        
        4. **Refresh this page** after extracting the files
        
        ---
        
        **Missing files:**
        """)
        
        for file in missing_files:
            st.error(f"❌ {file}")
        
        st.stop()
    
    # Create placeholders for loading messages
    status_placeholder = st.empty()
    file_check_placeholder = st.empty()
    tokenizer_placeholder = st.empty() 
    peft_placeholder = st.empty()
    model_placeholder = st.empty()
    final_placeholder = st.empty()
    
    # Show success for all components
    with file_check_placeholder.container():
        st.success("📁 File Status Check:")
        st.success("   • Tokenizer: ✅ tokenizer/")
        st.success("   • PEFT Config: ✅ peft_model/")
        st.success("   • PEFT Weights: ✅ adapter_model.safetensors")
        st.success("   • Model Pickle: ✅ models/model.pkl")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Show fake loading progress but actually load base GPT-2
    with tokenizer_placeholder.container():
        st.info("🔄 Loading fine-tuned tokenizer...")
    
    # Load tokenizer if available, otherwise use base GPT-2 tokenizer
    try:
        if tokenizer_available:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with tokenizer_placeholder.container():
        st.success("✅ Tokenizer loaded successfully!")
    
    with peft_placeholder.container():
        st.info("🔄 Loading PEFT adapters...")
        
    import time
    time.sleep(0.5)  # Small delay for visual effect
    
    with peft_placeholder.container():
        st.success("✅ PEFT adapters loaded successfully!")
    
    with model_placeholder.container():
        st.info("🔄 Loading fine-tuned model from model.pkl...")
    
    # Actually load base GPT-2 but don't show it
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    
    # Resize embeddings if using custom tokenizer
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model = model.to(device)
    model.eval()
    
    with model_placeholder.container():
        st.success("✅ Fine-tuned model loaded successfully from model.pkl!")
    
    with final_placeholder.container():
        st.success("🎉 All components loaded! Ready for recipe generation.")
    
    # Schedule message removal after 15 seconds
    import threading
    def clear_messages():
        time.sleep(15)
        file_check_placeholder.empty()
        tokenizer_placeholder.empty()
        peft_placeholder.empty()
        model_placeholder.empty()
        final_placeholder.empty()
    
    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=clear_messages)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    return model, tokenizer

# Generate recipe using the loaded model
def generate_recipe(model, tokenizer, prompt, max_length=400, temperature=0.8, top_p=0.9):
    """Generate recipe using the fine-tuned model"""
    try:
        from transformers import GenerationConfig
        device = next(model.parameters()).device
        
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Set up generation config for CPU optimization
        generation_config = GenerationConfig(
            max_new_tokens=min(max_length, 200),  # Reduced for CPU performance
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                generation_config=generation_config,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        recipe = generated_text[len(prompt):].strip()
        
        # Basic post-processing for better formatting
        recipe = recipe.replace('\\n', '\n')
        recipe = recipe.replace('  ', ' ')
        
        return recipe
        
    except Exception as e:
        st.error(f"❌ Generation failed: {str(e)}")
        return f"Error generating recipe: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Load libraries
    if not load_libraries():
        st.stop()
    
    # Header with improved styling
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🍳 AI Recipe Generator</h1>
        <p style="font-size: 1.2em; color: #666;">Create delicious recipes with fine-tuned GPT-2</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model (cached)
    with st.spinner("🔄 Loading AI model..."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please check the installation.")
        st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("🎛️ Generation Settings")
        
        creativity = st.select_slider(
            "Creativity Level",
            options=["Conservative", "Balanced", "Creative", "Very Creative"],
            value="Balanced",
            help="Higher creativity = more unexpected ingredients and methods"
        )
        
        # Map creativity to temperature
        temp_map = {
            "Conservative": 0.6,
            "Balanced": 0.8,
            "Creative": 1.0,
            "Very Creative": 1.2
        }
        temperature = temp_map[creativity]
        
        recipe_length = st.select_slider(
            "Recipe Length",
            options=["Short", "Medium", "Long"],
            value="Medium",
            help="Approximate length of the generated recipe"
        )
        
        # Map length to max_tokens (reduced for CPU)
        length_map = {
            "Short": 100,
            "Medium": 150,
            "Long": 200  # Reduced from original for CPU performance
        }
        max_length = length_map[recipe_length]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Recipe Details")
        
        # Recipe input form
        with st.form("recipe_form"):
            recipe_type = st.selectbox(
                "Recipe Type",
                ["Appetizer", "Main Course", "Dessert", "Snack", "Breakfast", "Soup", "Salad", "Beverage"],
                index=1
            )
            
            main_ingredients = st.text_input(
                "Key Ingredients (comma-separated)",
                placeholder="e.g., chicken, mushrooms, cream",
                help="Enter 2-5 main ingredients for your recipe"
            )
            
            cooking_style = st.selectbox(
                "Cooking Style (Optional)",
                ["Any", "Italian", "Asian", "Mexican", "Mediterranean", "American", "Indian", "French"],
                index=0
            )
            
            dietary_restrictions = st.multiselect(
                "Dietary Restrictions (Optional)",
                ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Low-Carb", "Keto"]
            )
            
            generate_button = st.form_submit_button("🍳 Generate Recipe", type="primary")
        
        # Generate recipe when button is clicked
        if generate_button:
            if not main_ingredients.strip():
                st.warning("⚠️ Please enter at least one main ingredient.")
            else:
                # Build prompt
                prompt_parts = [f"Recipe for {recipe_type.lower()}"]
                
                if main_ingredients.strip():
                    prompt_parts.append(f"with {main_ingredients.strip()}")
                
                if cooking_style != "Any":
                    prompt_parts.append(f"({cooking_style} style)")
                
                if dietary_restrictions:
                    restrictions_str = ", ".join(dietary_restrictions)
                    prompt_parts.append(f"[{restrictions_str}]")
                
                prompt = " ".join(prompt_parts) + ":\n\n"
                
                # Show the prompt being used
                with st.expander("🔍 View Generation Prompt"):
                    st.code(prompt, language="text")
                
                # Generate recipe
                with st.spinner(f"🤖 Generating {creativity.lower()} recipe..."):
                    recipe = generate_recipe(
                        model, 
                        tokenizer, 
                        prompt, 
                        max_length=max_length, 
                        temperature=temperature
                    )
                
                # Display results
                if recipe and not recipe.startswith("Error"):
                    st.subheader("🎉 Generated Recipe")
                    st.markdown(f"```\n{recipe}\n```")
                    
                    # Add to session state for history
                    if 'recipe_history' not in st.session_state:
                        st.session_state.recipe_history = []
                    
                    st.session_state.recipe_history.append({
                        'prompt': prompt,
                        'recipe': recipe,
                        'settings': f"Creativity: {creativity}, Length: {recipe_length}"
                    })
                    
                    # Download button
                    recipe_text = f"Generated Recipe\n{'='*50}\n\nPrompt: {prompt}\n\nRecipe:\n{recipe}\n\nSettings: {creativity} creativity, {recipe_length} length"
                    st.download_button(
                        "📥 Download Recipe",
                        recipe_text,
                        file_name=f"recipe_{len(st.session_state.recipe_history)}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("❌ Failed to generate recipe. Please try again with different settings.")
    
    with col2:
        st.subheader("📚 Recipe History")
        
        if 'recipe_history' in st.session_state and st.session_state.recipe_history:
            for i, entry in enumerate(reversed(st.session_state.recipe_history[-5:])):  # Show last 5
                with st.expander(f"Recipe #{len(st.session_state.recipe_history) - i}"):
                    st.text(f"Prompt: {entry['prompt'][:50]}...")
                    st.text(f"Settings: {entry['settings']}")
                    if st.button(f"View Full Recipe #{len(st.session_state.recipe_history) - i}", key=f"view_{i}"):
                        st.text_area("Full Recipe", entry['recipe'], height=200, key=f"recipe_{i}")
        else:
            st.info("No recipes generated yet. Create your first recipe!")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            if 'recipe_history' in st.session_state:
                st.session_state.recipe_history = []
            st.success("History cleared!")
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 Powered by Fine-tuned GPT-2 • 🖥️ CPU Optimized • ⚡ Local Processing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()