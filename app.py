import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model import UNetMCDropout
from src.uncertainty import mc_predict
from src.active_learning import rank_by_uncertainty

st.set_page_config(page_title="TrustSeg", layout="wide")

st.markdown("""
<style>
div.stButton > button {
    background: radial-gradient(150% 180% at 11% 140%, #000 37%, #08012c 61%, #4e1e40 78%, #70464e 89%, #88394c 100%);
    color: white;
    border: 1px solid rgba(220, 100, 150, 0.3);
    border-radius: 11px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.5s ease;
    min-width: 150px;
}

div.stButton > button:hover {
    background: radial-gradient(120% 103% at 0% 91%, #c96287 0%, #c66c64 8%, #cc7d23 21%, #37140a 71%, #000 85%);
    border-color: rgba(220, 150, 180, 0.6);
    transform: scale(1.02);
}

div.stButton > button:active {
    transform: scale(0.98);
}
</style>
""", unsafe_allow_html=True)

st.title("TrustSeg: Uncertainty-Aware Skin Lesion Segmentation")

st.markdown("""
### What this tool does
- Segments skin lesions using a deep learning model (U-Net)
- Estimates prediction uncertainty using Monte Carlo Dropout
- Highlights unreliable predictions for review

### How to interpret results
- Low uncertainty → high confidence
- High uncertainty → model is unsure
""")

st.markdown("Upload a dermoscopic image to see the predicted segmentation mask and uncertainty map.")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetMCDropout(dropout_p=0.3).to(device)
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
    return model, device

def preprocess(image):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    image_np = np.array(image.convert("RGB"))
    return transform(image=image_np)['image']

tab1, tab2 = st.tabs(["Segmentation Demo", "Active Learning"])

# ---------------------- TAB 1 ----------------------
with tab1:
    uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"])
    use_sample = st.button("Try a sample image")

    if use_sample:
        image = Image.open("sample.png")
    elif uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = None

    if image is not None:
        try:
            model, device = load_model()

            with st.spinner("Running 20 stochastic forward passes..."):
                tensor = preprocess(image)
                mean_pred, uncertainty = mc_predict(model, tensor, n_passes=20, device=device)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Predicted Mask")
                mask_display = (mean_pred > 0.5).astype(np.uint8) * 255
                st.image(mask_display, use_container_width=True, clamp=True)

            with col3:
                st.subheader("Uncertainty Map")
                fig, ax = plt.subplots()
                ax.imshow(uncertainty, cmap='RdYlGn_r')
                ax.axis('off')
                plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
                st.pyplot(fig)
                plt.close()

            with col4:
                st.subheader("Overlay")
                image_resized = np.array(image.convert("RGB").resize((256, 256)))
                mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
                mask_binary = (mean_pred > 0.5)
                mask_rgb[mask_binary] = [255, 0, 0]
                overlay = (0.6 * image_resized + 0.4 * mask_rgb).astype(np.uint8)
                st.image(overlay, use_container_width=True)

            avg_uncertainty = uncertainty.mean()

            st.divider()
            col5, col6 = st.columns(2)

            with col5:
                st.metric("Average Uncertainty", f"{avg_uncertainty:.6f}")

            with col6:
                confidence = "High" if avg_uncertainty < 0.01 else "Medium" if avg_uncertainty < 0.05 else "Low"
                st.metric("Model Confidence", confidence)

            if avg_uncertainty >= 0.03:
                st.warning("⚠️ Low confidence prediction — recommend manual review by a clinician.")

            if avg_uncertainty < 0.01:
                st.success("✅ Model is very confident in this prediction.")
            elif avg_uncertainty < 0.03:
                st.info("ℹ️ Moderate confidence. Review recommended.")
            else:
                st.error("❌ Low confidence. Prediction may be unreliable.")

            st.divider()
            st.subheader("Pixel-Level Uncertainty Distribution")
            st.markdown("This histogram shows how uncertainty is distributed across every pixel in the image. A spike on the left means most pixels are confident. A long right tail means many pixels are uncertain.")

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            uncertainty_flat = uncertainty.flatten()
            ax2.hist(uncertainty_flat, bins=80, color='steelblue', alpha=0.8, edgecolor='none')
            ax2.axvline(avg_uncertainty, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {avg_uncertainty:.4f}')
            ax2.axvline(0.03, color='orange', linestyle='--', linewidth=1.5, label='Review threshold (0.03)')
            ax2.set_xlabel("Uncertainty (Variance)")
            ax2.set_ylabel("Pixel Count")
            ax2.set_title("Distribution of Pixel-Level Uncertainty")
            ax2.legend()
            st.pyplot(fig2)
            plt.close()

            st.divider()
            st.subheader("Model Performance Analysis")
            st.markdown("The plot below shows uncertainty vs Dice score across **800 training images**. Higher uncertainty reliably predicts lower segmentation accuracy — confirming that MC Dropout uncertainty is a meaningful signal.")

            if os.path.exists("uncertainty_vs_dice.png"):
                st.image("uncertainty_vs_dice.png", use_container_width=True, caption="Negative correlation between MC Dropout uncertainty and Dice coefficient (n=800)")
            else:
                st.info("Run src/evaluate.py to generate this plot.")

        except FileNotFoundError:
            st.warning("No trained model found. Please train the model first by running src/train.py")

# ---------------------- TAB 2 ----------------------
with tab2:
    st.subheader("Uncertainty-Guided Active Learning")
    st.markdown("""
    Upload multiple unlabeled images. The model ranks them by uncertainty.
    Label the most uncertain ones first to improve performance efficiently.
    
    This implements **maximum entropy sampling**, a core strategy in active learning research.
    """)

    uploaded_files = st.file_uploader(
        "Upload unlabeled images to rank by uncertainty",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        try:
            model, device = load_model()

            with st.spinner(f"Ranking {len(uploaded_files)} images by uncertainty..."):
                images = [(f.name, Image.open(f)) for f in uploaded_files]
                ranked = rank_by_uncertainty(model, images, device, n_passes=10)

            st.success(f"✅ Ranked {len(ranked)} images. Label the top ones first!")

            st.divider()

            for i, result in enumerate(ranked):
                priority = "🔴 High priority" if result['uncertainty'] > 0.03 else "🟡 Medium priority" if result['uncertainty'] > 0.01 else "🟢 Low priority"

                with st.expander(f"#{i+1} — {result['name']} | Uncertainty: {result['uncertainty']:.6f} | {priority}"):
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.caption("Original image")
                        st.image(result['image'], use_container_width=True)

                    with c2:
                        st.caption("Predicted mask")
                        mask_display = (result['mean_pred'] > 0.5).astype(np.uint8) * 255
                        st.image(mask_display, use_container_width=True, clamp=True)

                    with c3:
                        st.caption("Uncertainty map")
                        fig3, ax3 = plt.subplots()
                        ax3.imshow(result['uncertainty_map'], cmap='RdYlGn_r')
                        ax3.axis('off')
                        st.pyplot(fig3)
                        plt.close()

            st.divider()
            st.subheader("Uncertainty Rankings Summary")
            fig4, ax4 = plt.subplots(figsize=(8, 3))
            names = [r['name'][:15] for r in ranked]
            uncertainties = [r['uncertainty'] for r in ranked]
            colors = ['red' if u > 0.03 else 'orange' if u > 0.01 else 'steelblue' for u in uncertainties]
            ax4.barh(names, uncertainties, color=colors)
            ax4.axvline(0.03, color='red', linestyle='--', linewidth=1, label='Review threshold')
            ax4.set_xlabel("Average Uncertainty")
            ax4.set_title("Images Ranked by Uncertainty (label red ones first)")
            ax4.legend()
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        except FileNotFoundError:
            st.warning("No trained model found. Please train the model first.")