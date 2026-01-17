import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import io
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .person-card {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DATASET_PATH = 'dataset/faces'
TARGET_SIZE = (92, 112)

# Cache functions for performance
@st.cache_data
def get_person_names():
    """Get list of person names from dataset."""
    if os.path.exists(DATASET_PATH):
        return sorted([name for name in os.listdir(DATASET_PATH) 
                      if os.path.isdir(os.path.join(DATASET_PATH, name))])
    return []

def augment_image(img):
    """Apply data augmentation to increase dataset diversity."""
    augmented_images = []
    augmented_images.append(img)
    
    # Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)
    
    # Small rotation
    rows, cols = img.shape
    for angle in [-3, 3]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(rotated)
    
    # Slight translation
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        translated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(translated)
    
    return augmented_images

@st.cache_data
def load_dataset(augment=False):
    """Loads images from the dataset folder with enhanced preprocessing."""
    images = []
    labels = []
    person_names = []
    person_id = 0
    
    for person_folder in sorted(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person_folder)
        if os.path.isdir(person_path):
            person_names.append(person_folder)
            for filename in sorted(os.listdir(person_path)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, TARGET_SIZE)
                        img_equalized = cv2.equalizeHist(img_resized)
                        img_denoised = cv2.GaussianBlur(img_equalized, (3, 3), 0)
                        img_normalized = img_denoised.astype(np.float32) / 255.0
                        
                        if augment:
                            augmented_imgs = augment_image(img_normalized)
                            for aug_img in augmented_imgs:
                                images.append(aug_img.flatten())
                                labels.append(person_id)
                        else:
                            images.append(img_normalized.flatten())
                            labels.append(person_id)
            person_id += 1
            
    return np.array(images), np.array(labels), person_names

def perform_pca(X_train, k):
    """Performs PCA on the training data to get Eigenfaces using SVD."""
    mean_face = np.mean(X_train, axis=1, keepdims=True)
    delta = X_train - mean_face
    
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    eigenfaces = U[:, :k]
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    signatures_train = eigenfaces.T @ delta
    
    total_variance = np.sum(S**2)
    explained_variance = np.sum((S[:k]**2) / total_variance)
    
    return mean_face, eigenfaces, signatures_train, explained_variance

@st.cache_resource
def train_model(k_value=100):
    """Train the face recognition model."""
    images, labels, person_names = load_dataset(augment=True)
    
    # Use 70-30 split for better training
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    X_train = X_train.T
    X_test = X_test.T
    
    mean_face, eigenfaces, signatures_train, explained_variance = perform_pca(X_train, k=k_value)
    
    X_train_ann = signatures_train.T
    scaler = StandardScaler()
    X_train_ann_scaled = scaler.fit_transform(X_train_ann)
    
    # Improved neural network with better architecture
    ann_classifier = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),  # Deeper network
        activation='relu',
        solver='adam',
        alpha=0.0001,  # Less regularization
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,  # More iterations
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
        batch_size=64
    )
    ann_classifier.fit(X_train_ann_scaled, y_train)
    
    # Test accuracy
    delta_test = X_test - mean_face
    signatures_test = eigenfaces.T @ delta_test
    X_test_ann_scaled = scaler.transform(signatures_test.T)
    predictions = ann_classifier.predict(X_test_ann_scaled)
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        'model': ann_classifier,
        'scaler': scaler,
        'mean_face': mean_face,
        'eigenfaces': eigenfaces,
        'person_names': person_names,
        'accuracy': accuracy,
        'explained_variance': explained_variance,
        'y_test': y_test,
        'predictions': predictions
    }

def detect_and_crop_face(img_array):
    """Detect face in image and crop it."""
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        img_array,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add some padding
        pad = int(0.1 * max(w, h))
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img_array.shape[1] - x, w + 2*pad)
        h = min(img_array.shape[0] - y, h + 2*pad)
        face_crop = img_array[y:y+h, x:x+w]
        return face_crop, True
    
    return img_array, False

def preprocess_uploaded_image(uploaded_file):
    """Preprocess an uploaded image for recognition."""
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    
    # Try to detect and crop face
    face_img, face_detected = detect_and_crop_face(img_array)
    
    # Resize to target size
    img_resized = cv2.resize(face_img, TARGET_SIZE)
    
    # Apply same preprocessing as training
    img_equalized = cv2.equalizeHist(img_resized)
    img_denoised = cv2.GaussianBlur(img_equalized, (3, 3), 0)
    img_normalized = img_denoised.astype(np.float32) / 255.0
    
    return img_normalized, face_detected

def recognize_face(img_normalized, model_data):
    """Recognize a face from preprocessed image."""
    img_vector = img_normalized.flatten().reshape(-1, 1)
    delta = img_vector - model_data['mean_face']
    signature = model_data['eigenfaces'].T @ delta
    signature_scaled = model_data['scaler'].transform(signature.T)
    
    prediction = model_data['model'].predict(signature_scaled)[0]
    probabilities = model_data['model'].predict_proba(signature_scaled)[0]
    confidence = np.max(probabilities) * 100
    
    # Calculate reconstruction error to detect unknown faces
    reconstructed = model_data['eigenfaces'] @ signature + model_data['mean_face']
    reconstruction_error = np.mean((img_vector - reconstructed) ** 2)
    
    return prediction, confidence, probabilities, reconstruction_error

# Main App
def main():
    st.markdown('<h1 class="main-header">👤 Face Recognition System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">PCA + Eigenfaces + Neural Network</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("# 👤")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Recognize Face", "📊 Model Analytics", "👥 Dataset Gallery"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Settings")
    k_value = st.sidebar.slider("Number of Eigenfaces (k)", 50, 200, 100, 25)
    
    # Train model
    with st.spinner("🔄 Loading model... (this may take a moment)"):
        model_data = train_model(k_value)
    
    if page == "🏠 Home":
        show_home(model_data)
    elif page == "🔍 Recognize Face":
        show_recognition(model_data)
    elif page == "📊 Model Analytics":
        show_analytics(model_data)
    elif page == "👥 Dataset Gallery":
        show_gallery()

def show_home(model_data):
    """Home page with overview."""
    st.markdown("---")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", f"{model_data['accuracy']*100:.1f}%")
    with col2:
        st.metric("👥 People", len(model_data['person_names']))
    with col3:
        st.metric("📊 Eigenfaces", model_data['eigenfaces'].shape[1])
    with col4:
        st.metric("📈 Variance", f"{model_data['explained_variance']*100:.1f}%")
    
    st.markdown("---")
    
    # How it works
    st.markdown("## 🔬 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ PCA & Eigenfaces
        - Compute mean face
        - Extract principal components
        - Create eigenfaces basis
        - Reduce dimensionality
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ Feature Extraction
        - Project faces onto eigenfaces
        - Generate compact signatures
        - Scale features
        - Preserve identity info
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ Neural Network
        - Multi-layer perceptron
        - 300-200-100 architecture
        - Adam optimizer
        - Classification output
        """)
    
    st.markdown("---")
    
    # People in dataset
    st.markdown("## 👥 People in Dataset")
    cols = st.columns(9)
    for idx, name in enumerate(model_data['person_names']):
        with cols[idx % 9]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 10px; border-radius: 10px; text-align: center; margin: 5px;">
                <span style="color: white; font-weight: bold;">{name}</span>
            </div>
            """, unsafe_allow_html=True)

def show_recognition(model_data):
    """Face recognition page."""
    st.markdown("## 🔍 Upload Image for Recognition")
    st.markdown("Upload an image of a face to identify the person.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Recognize Face", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    img_normalized, face_detected = preprocess_uploaded_image(uploaded_file)
                    prediction, confidence, probabilities, recon_error = recognize_face(img_normalized, model_data)
                    
                    st.session_state['recognition_result'] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'person_names': model_data['person_names'],
                        'face_detected': face_detected,
                        'recon_error': recon_error
                    }
    
    with col2:
        if 'recognition_result' in st.session_state:
            result = st.session_state['recognition_result']
            predicted_name = result['person_names'][result['prediction']]
            confidence = result['confidence']
            
            # Determine reliability based on confidence and reconstruction error
            is_reliable = confidence > 60 and result['recon_error'] < 0.05
            
            if not result['face_detected']:
                st.warning("⚠️ No face detected in image. Results may be unreliable.")
            
            if result['recon_error'] > 0.05:
                st.warning("⚠️ High reconstruction error. The face may not be in the dataset.")
            
            # Color based on confidence
            if confidence > 70 and is_reliable:
                color_gradient = "linear-gradient(135deg, #28a745 0%, #20c997 100%)"  # Green
                reliability = "High Confidence"
            elif confidence > 50:
                color_gradient = "linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)"  # Orange
                reliability = "Medium Confidence"
            else:
                color_gradient = "linear-gradient(135deg, #dc3545 0%, #c82333 100%)"  # Red
                reliability = "Low Confidence"
            
            st.markdown(f"""
            <div style="background: {color_gradient}; 
                        padding: 2rem; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">Identified Person</h2>
                <h1 style="color: white; font-size: 3rem; margin: 1rem 0;">{predicted_name}</h1>
                <h3 style="color: rgba(255,255,255,0.8);">Confidence: {confidence:.1f}%</h3>
                <p style="color: rgba(255,255,255,0.7);">{reliability}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Confidence Scores")
            
            # Show top 5 matches
            top_indices = np.argsort(result['probabilities'])[-5:][::-1]
            for idx in top_indices:
                name = result['person_names'][idx]
                prob = float(result['probabilities'][idx] * 100)
                st.progress(prob / 100, text=f"{name}: {prob:.1f}%")

def show_analytics(model_data):
    """Model analytics page."""
    st.markdown("## 📊 Model Analytics")
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    cm = confusion_matrix(model_data['y_test'], model_data['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model_data['person_names'],
                yticklabels=model_data['person_names'], ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Per-class accuracy
    st.markdown("### Per-Person Accuracy")
    report = classification_report(model_data['y_test'], model_data['predictions'], 
                                   target_names=model_data['person_names'], output_dict=True)
    
    cols = st.columns(3)
    for idx, name in enumerate(model_data['person_names']):
        with cols[idx % 3]:
            precision = report[name]['precision'] * 100
            st.metric(name, f"{precision:.1f}%")
    
    # Eigenfaces visualization
    st.markdown("### 👁️ Top Eigenfaces Visualization")
    n_eigenfaces = min(8, model_data['eigenfaces'].shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < n_eigenfaces:
            eigenface = model_data['eigenfaces'][:, i].reshape(TARGET_SIZE[1], TARGET_SIZE[0])
            ax.imshow(eigenface, cmap='gray')
            ax.set_title(f'Eigenface {i+1}')
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Mean face
    st.markdown("### 👤 Mean Face")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        mean_face_img = model_data['mean_face'].reshape(TARGET_SIZE[1], TARGET_SIZE[0])
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.imshow(mean_face_img, cmap='gray')
        ax.set_title('Average Face from Dataset')
        ax.axis('off')
        st.pyplot(fig)

def show_gallery():
    """Dataset gallery page."""
    st.markdown("## 👥 Dataset Gallery")
    
    person_names = get_person_names()
    
    if not person_names:
        st.error("Dataset not found!")
        return
    
    selected_person = st.selectbox("Select a person", person_names)
    
    if selected_person:
        person_path = os.path.join(DATASET_PATH, selected_person)
        images = []
        
        for filename in sorted(os.listdir(person_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, filename)
                img = Image.open(img_path)
                images.append((filename, img))
        
        st.markdown(f"### {selected_person}'s Images ({len(images)} total)")
        
        cols = st.columns(5)
        for idx, (filename, img) in enumerate(images[:10]):
            with cols[idx % 5]:
                st.image(img, caption=filename, use_container_width=True)

if __name__ == "__main__":
    main()
